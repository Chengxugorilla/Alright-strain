#!/usr/bin/env python3
"""Resumable, communication-free multi-GPU H1N1 antigenicity scoring.

Each worker deterministically owns a subset of (forecast cutoff, sequence
identity) rows.  Workers communicate only through SQLite's atomic commits;
there is no DDP collective operation.  A completed query summary is the
resume marker, while the individual reference--virus model outputs are
retained in ``pair_scores`` for audit and later re-aggregation.  Besides the
original frozen-serum panels, the scorer can construct past-strain reference
panels for each cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.antigenicity.common import FLUPROFILER_ROOT, IDENTITY_COLUMNS, PROJECT_ROOT

if str(FLUPROFILER_ROOT) not in sys.path:
    sys.path.insert(0, str(FLUPROFILER_ROOT))

from experiments.serum_gate.train_serum_mutation_set import (  # noqa: E402
    align_embedding_to_sequence,
    load_ha_distance_matrix,
    normalize_passage,
)
from src.fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetBatch,
    SerumMutationSetMinusModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=FLUPROFILER_ROOT / "results/H1_HA1_v1.0/SCMS-FiLM-H1/full_data_interpretation/H1N1/checkpoints/epoch_0050.pth",
    )
    parser.add_argument("--distance-matrix", type=Path, default=FLUPROFILER_ROOT / "ha1_distance_no_bias_329.npy")
    parser.add_argument("--query-instances-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv")
    parser.add_argument("--query-registry-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_ha1_registry.csv")
    parser.add_argument("--cutoffs-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels/forecast_cutoffs.csv")
    parser.add_argument("--panels-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels/frozen_serum_panels.csv")
    parser.add_argument("--base-embedding-dir", type=Path, default=FLUPROFILER_ROOT / "data/embedding/files")
    parser.add_argument("--new-embedding-dir", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/embeddings")
    parser.add_argument(
        "--reference-mode",
        choices=["serum", "past_strains"],
        default="serum",
        help="Use the frozen serum panel, or unique past HA1 strains as reference sequences.",
    )
    parser.add_argument(
        "--current-months",
        type=int,
        default=6,
        help="In past_strains mode, query strains are sampled in the final N months before each cutoff.",
    )
    parser.add_argument(
        "--past-years",
        type=int,
        default=5,
        help="In past_strains mode, reference strains are sampled in the preceding N years.",
    )
    parser.add_argument(
        "--reference-passage",
        default="<CELL>",
        help="Passage assigned to past-strain references; used only in past_strains mode.",
    )
    parser.add_argument(
        "--past-panel-size",
        type=int,
        default=0,
        help=(
            "Optional compressed past-strain panel size per cutoff. 0 retains every unique past HA1; "
            "192 selects six-month, sequence-diverse representatives."
        ),
    )
    parser.add_argument(
        "--representative-strata-months",
        type=int,
        default=6,
        help="Temporal stratum width used when --past-panel-size is positive.",
    )
    parser.add_argument(
        "--representative-sentinel-fraction",
        type=float,
        default=0.25,
        help="Fraction of each temporal stratum reserved for sequence-diversity sentinels.",
    )
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        default=None,
        help="Output database. Defaults depend on --reference-mode so serum and past-strain runs remain separate.",
    )
    parser.add_argument("--cutoff", action="append", default=[], help="Optional YYYY-MM-DD cutoff; repeatable.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=0, help="Optional per-rank limit for a small smoke run; 0 means all assigned rows.")
    parser.add_argument("--query-passage", default="<CELL>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-items", type=int, default=128)
    parser.add_argument(
        "--gpu-cache-gb",
        type=float,
        default=36.0,
        help="Per-worker aligned-embedding cache budget in GiB; 0 disables it.",
    )
    return parser.parse_args()


def distributed_context(device_arg: str) -> tuple[int, int, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if device_arg != "auto":
        return rank, world_size, device_arg
    if torch.cuda.is_available():
        return rank, world_size, f"cuda:{local_rank}"
    return rank, world_size, "cpu"


class EmbeddingCache:
    def __init__(self, roots: list[Path], max_items: int) -> None:
        self.roots = roots
        self.max_items = max_items
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, embedding_id: str) -> torch.Tensor:
        value = self.values.pop(embedding_id, None)
        if value is None:
            filename = f"matrix_{embedding_id}.pt"
            path = next((root / filename for root in self.roots if (root / filename).is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Missing embedding {filename} in: {', '.join(map(str, self.roots))}")
            value = torch.as_tensor(torch.load(path, map_location="cpu", weights_only=False)).float()
            if value.ndim != 2:
                raise ValueError(f"Embedding must be 2-D: {path}")
        self.values[embedding_id] = value
        if len(self.values) > self.max_items:
            self.values.popitem(last=False)
        return value


class GpuAlignedCache:
    """Bounded GPU LRU cache that preferentially retains the active reference panel."""

    def __init__(self, cpu_cache: EmbeddingCache, device: torch.device, budget_gb: float) -> None:
        if budget_gb < 0:
            raise ValueError("--gpu-cache-gb must be non-negative")
        self.cpu_cache = cpu_cache
        self.device = device
        self.budget_bytes = int(budget_gb * 1024**3)
        self.used_bytes = 0
        self.values: OrderedDict[tuple[str, str], tuple[torch.Tensor, ...]] = OrderedDict()
        self.active_reference_keys: frozenset[tuple[str, str]] = frozenset()
        self.hits = 0
        self.misses = 0

    def activate_reference_panel(self, panel: pd.DataFrame) -> None:
        """Mark the current cutoff's references as eviction-resistant.

        The same full reference panel is visited for every query in a cutoff.
        Without this distinction, newly arriving query embeddings can evict panel
        entries and turn the next sequential panel scan into a cache-thrashing
        reload.  If the panel itself exceeds the budget, eviction gracefully
        falls back to normal LRU behaviour.
        """
        self.active_reference_keys = frozenset(
            zip(panel["seq_id_a"].astype(str), panel["serum_ha1"].astype(str), strict=True)
        )

    @staticmethod
    def _bytes(value: tuple[torch.Tensor, ...]) -> int:
        return sum(tensor.nelement() * tensor.element_size() for tensor in value)

    def _evict_one(self) -> None:
        # Prefer evicting a stale-query or prior-cutoff embedding.  If every
        # entry is part of the active panel, ordinary LRU is the only safe
        # fallback and preserves correctness with an undersized cache.
        key = next((key for key in self.values if key not in self.active_reference_keys), None)
        if key is None:
            key, evicted = self.values.popitem(last=False)
        else:
            evicted = self.values.pop(key)
        self.used_bytes -= self._bytes(evicted)

    def get(self, embedding_id: str, aligned_sequence: str) -> tuple[torch.Tensor, ...]:
        key = (embedding_id, aligned_sequence)
        value = self.values.pop(key, None)
        if value is not None:
            self.hits += 1
            self.values[key] = value
            return value
        self.misses += 1
        value = tuple(
            tensor.to(self.device, non_blocking=True)
            for tensor in align_embedding_to_sequence(self.cpu_cache.get(embedding_id), aligned_sequence)
        )
        size = self._bytes(value)
        if self.budget_bytes == 0 or size > self.budget_bytes:
            return value
        while self.values and self.used_bytes + size > self.budget_bytes:
            self._evict_one()
        self.values[key] = value
        self.used_bytes += size
        return value

    def stats(self) -> dict[str, float | int]:
        return {
            "gpu_cache_budget_gib": self.budget_bytes / 1024**3,
            "gpu_cache_used_gib": self.used_bytes / 1024**3,
            "gpu_cache_entries": len(self.values),
            "active_reference_entries": len(self.active_reference_keys),
            "gpu_cache_hits": self.hits,
            "gpu_cache_misses": self.misses,
        }


def initialize_database(path: Path, metadata: dict[str, str]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=120000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS pair_scores (
            forecast_cutoff TEXT NOT NULL,
            fasta_sha256 TEXT NOT NULL,
            sequence_index INTEGER NOT NULL,
            sequence_sha256 TEXT NOT NULL,
            query_id TEXT NOT NULL,
            serum_seq_id TEXT NOT NULL,
            serum_passage TEXT NOT NULL,
            serum_name TEXT NOT NULL,
            serum_date TEXT NOT NULL,
            reference_weight REAL NOT NULL DEFAULT 1.0,
            predicted_distance REAL NOT NULL,
            scored_at_utc TEXT NOT NULL,
            PRIMARY KEY (forecast_cutoff, fasta_sha256, sequence_index, sequence_sha256, serum_seq_id, serum_passage, serum_name)
        );
        CREATE TABLE IF NOT EXISTS query_summary (
            forecast_cutoff TEXT NOT NULL,
            fasta_sha256 TEXT NOT NULL,
            sequence_index INTEGER NOT NULL,
            sequence_sha256 TEXT NOT NULL,
            query_id TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            query_passage TEXT NOT NULL,
            panel_size INTEGER NOT NULL,
            mean_distance REAL NOT NULL,
            median_distance REAL NOT NULL,
            p10_distance REAL NOT NULL,
            p90_distance REAL NOT NULL,
            min_distance REAL NOT NULL,
            max_distance REAL NOT NULL,
            scored_at_utc TEXT NOT NULL,
            PRIMARY KEY (forecast_cutoff, fasta_sha256, sequence_index, sequence_sha256)
        );
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS query_summary_cutoff_query_id_idx
        ON query_summary (forecast_cutoff, query_id);
        """
    )
    # Older frozen-serum databases do not have this audit column.  A migration
    # keeps them readable/resumable while compressed panels can record weights.
    pair_columns = {row[1] for row in connection.execute("PRAGMA table_info(pair_scores)")}
    if "reference_weight" not in pair_columns:
        connection.execute("ALTER TABLE pair_scores ADD COLUMN reference_weight REAL NOT NULL DEFAULT 1.0")
    for key, value in metadata.items():
        existing = connection.execute("SELECT value FROM run_metadata WHERE key=?", (key,)).fetchone()
        if existing is not None and existing[0] != value:
            raise ValueError(
                f"Output database metadata conflict for {key!r}: {existing[0]!r} != {value!r}. "
                "Use a separate --output-sqlite for a different scoring definition."
            )
        connection.execute("INSERT OR IGNORE INTO run_metadata(key, value) VALUES (?, ?)", (key, value))
    connection.commit()
    return connection


def stable_owner(cutoff: str, row: pd.Series, world_size: int) -> int:
    identity = "|".join([cutoff, *(str(row[column]) for column in IDENTITY_COLUMNS)])
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16) % world_size


def attach_registry(rows: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Attach sequence and embedding data while validating the registry identity."""

    combined = rows.merge(registry, on="query_id", how="left", validate="many_to_one")
    if "embedding_id" not in combined.columns:
        combined["embedding_id"] = combined["registry_embedding_id"]
    elif not combined["embedding_id"].fillna("").eq(combined["registry_embedding_id"].fillna("")).all():
        raise ValueError("Query-instance embedding_id does not match its HA1 registry embedding_id.")
    combined = combined.drop(columns=["registry_embedding_id"])
    if combined[["ha1_sequence", "embedding_id"]].isna().any().any():
        raise ValueError("At least one query instance has no HA1 registry record.")
    return combined


def deduplicate_strains(rows: pd.DataFrame) -> pd.DataFrame:
    """Retain one representation per HA1 identity within a time window."""

    return rows.sort_values(["collection_date", "query_id"], kind="stable").drop_duplicates("query_id", keep="last")


def sequence_distance_to_many(sequence: str, sequences: np.ndarray) -> np.ndarray:
    """Hamming distances for equal-length HA1 strings, without loading model embeddings."""

    encoded = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    matrix = np.frombuffer("".join(sequences.tolist()).encode("ascii"), dtype=np.uint8).reshape(len(sequences), -1)
    if matrix.shape[1] != len(encoded):
        raise ValueError("Representative selection requires equal-length HA1 sequences.")
    return np.count_nonzero(matrix != encoded, axis=1)


def select_strain_representatives(
    strains: pd.DataFrame,
    target_size: int,
    sentinel_fraction: float,
) -> pd.DataFrame:
    """Build a weighted sequence coreset from one temporal stratum.

    The first representatives are weighted diversity centres; the final
    ``sentinel_fraction`` are deliberately peripheral sequences.  Every
    historical HA1 is assigned to its nearest selected real sequence, so the
    resulting reference weights preserve the original sampling mass.
    """

    if strains.empty or target_size < 1:
        return strains.iloc[0:0].copy()
    ordered = strains.sort_values(["source_weight", "collection_date", "query_id"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    if len(ordered) <= target_size:
        result = ordered.copy()
        result["reference_weight"] = result["source_weight"] / result["source_weight"].sum()
        return result

    sequences = ordered["ha1_sequence"].to_numpy(dtype=str)
    source_weight = ordered["source_weight"].to_numpy(dtype=float)
    sentinel_count = min(target_size - 1, int(round(target_size * sentinel_fraction)))
    centre_count = target_size - sentinel_count
    selected = [0]  # highest-observation deterministic seed
    nearest = sequence_distance_to_many(sequences[0], sequences).astype(float)

    # Main representatives favour regions with both sequence separation and
    # observed sampling mass. sqrt(weight) avoids a single oversampled strain
    # consuming an entire temporal stratum.
    for _ in range(1, centre_count):
        score = nearest * np.sqrt(source_weight)
        score[selected] = -1.0
        choice = int(np.argmax(score))
        selected.append(choice)
        nearest = np.minimum(nearest, sequence_distance_to_many(sequences[choice], sequences))

    # Sentinels explicitly retain antigenically/sequence-distinct edge cases.
    for _ in range(sentinel_count):
        score = nearest.copy()
        score[selected] = -1.0
        choice = int(np.argmax(score))
        selected.append(choice)
        nearest = np.minimum(nearest, sequence_distance_to_many(sequences[choice], sequences))

    selected_array = np.asarray(selected, dtype=int)
    selected_sequences = sequences[selected_array]
    assignment_distance = np.stack(
        [sequence_distance_to_many(sequence, selected_sequences) for sequence in sequences], axis=0
    )
    assignments = assignment_distance.argmin(axis=1)
    weights = np.bincount(assignments, weights=source_weight, minlength=len(selected_array))
    result = ordered.iloc[selected_array].copy().reset_index(drop=True)
    result["reference_weight"] = weights / weights.sum()
    return result


def build_past_strain_panel(
    candidates: pd.DataFrame,
    current_query_ids: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reference_passage: str,
    panel_size: int,
    strata_months: int,
    sentinel_fraction: float,
) -> pd.DataFrame:
    """Convert past HA1 strains to scorer-compatible reference rows."""

    past = candidates.loc[candidates["collection_date"].between(start, end, inclusive="both")].copy()
    # A repeatedly observed HA1 is never used as its own historical reference.
    past = past.loc[~past["query_id"].isin(set(current_query_ids))].copy()
    occurrence_count = past.groupby("query_id", sort=False).size().rename("source_weight")
    past = deduplicate_strains(past).merge(occurrence_count, on="query_id", how="left", validate="one_to_one")
    if past.empty:
        raise ValueError(f"No past strains remain in {start.date()}..{end.date()}.")
    if panel_size:
        if strata_months < 1:
            raise ValueError("--representative-strata-months must be positive")
        if not 0.0 <= sentinel_fraction < 1.0:
            raise ValueError("--representative-sentinel-fraction must be in [0, 1)")
        covered_months = (end.year - start.year) * 12 + (end.month - start.month)
        strata_count = int(np.ceil(max(1, covered_months) / strata_months))
        strata_count = max(1, strata_count)
        if panel_size < strata_count:
            raise ValueError("--past-panel-size must be at least the number of temporal strata")
        # Use exact calendar boundaries anchored at the retrospective-window
        # start (rather than just calendar month numbers).  This keeps every
        # nominal six-month stratum the same width even when start/end dates
        # fall late in a month.
        boundaries = np.asarray(
            [start + pd.DateOffset(months=strata_months * index) for index in range(strata_count)],
            dtype="datetime64[ns]",
        )
        dates = past["collection_date"].to_numpy(dtype="datetime64[ns]")
        past["_stratum"] = np.minimum(np.searchsorted(boundaries, dates, side="right") - 1, strata_count - 1)
        base, remainder = divmod(panel_size, strata_count)
        frames = {int(stratum): frame for stratum, frame in past.groupby("_stratum", sort=True)}
        capacities = {stratum: len(frame) for stratum, frame in frames.items()}
        allocation = {
            stratum: min(capacity, base + (stratum < remainder))
            for stratum, capacity in capacities.items()
        }
        # Sparse early time strata should not silently reduce the whole panel.
        # Reassign their unused places to strata with still-unselected strains,
        # preferring the largest remaining candidate pool deterministically.
        remaining = min(panel_size, len(past)) - sum(allocation.values())
        while remaining:
            eligible = [
                stratum for stratum, capacity in capacities.items()
                if allocation[stratum] < capacity
            ]
            if not eligible:
                break
            selected_stratum = max(eligible, key=lambda stratum: (capacities[stratum] - allocation[stratum], -stratum))
            allocation[selected_stratum] += 1
            remaining -= 1
        selected = []
        for stratum, frame in frames.items():
            target = allocation[stratum]
            selected.append(select_strain_representatives(frame.drop(columns="_stratum"), target, sentinel_fraction))
        past = pd.concat(selected, ignore_index=True)
        past["reference_weight"] = past["reference_weight"] / past["reference_weight"].sum()
    else:
        past["reference_weight"] = past["source_weight"] / past["source_weight"].sum()
    return pd.DataFrame(
        {
            # Legacy serum field names are kept for batch/schema compatibility.
            # In past_strains mode they represent a virus reference, not serum.
            "seq_id_a": past["embedding_id"].astype(str).to_numpy(),
            "serum_ha1": past["ha1_sequence"].astype(str).to_numpy(),
            "serumPassCat": reference_passage,
            "serumName": past["query_id"].astype(str).to_numpy(),
            "serum_date": past["collection_date"].dt.date.astype(str).to_numpy(),
            "reference_weight": past["reference_weight"].astype(float).to_numpy(),
        }
    )


def load_work(args: argparse.Namespace, rank: int, world_size: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    instances = pd.read_csv(args.query_instances_csv)
    registry = pd.read_csv(args.query_registry_csv, usecols=["query_id", "ha1_sequence", "embedding_id"]).rename(
        columns={"embedding_id": "registry_embedding_id"}
    )
    cutoffs = pd.read_csv(args.cutoffs_csv)
    instances["collection_date"] = pd.to_datetime(instances["collection_date"], errors="coerce").dt.normalize()
    if instances["collection_date"].isna().any():
        raise ValueError("Query instances include invalid collection_date values.")
    if args.current_months < 1 or args.past_years < 1:
        raise ValueError("--current-months and --past-years must be positive")
    if args.past_panel_size < 0:
        raise ValueError("--past-panel-size must be non-negative")
    work: list[pd.DataFrame] = []
    selected = set(args.cutoff) if args.cutoff else None
    if args.reference_mode == "serum":
        for cutoff_row in cutoffs.itertuples(index=False):
            cutoff = str(cutoff_row.forecast_cutoff)
            if selected is not None and cutoff not in selected:
                continue
            annual_start = pd.Timestamp(cutoff_row.annual_start)
            annual_end = pd.Timestamp(cutoff_row.annual_end)
            rows = instances.loc[instances["collection_date"].between(annual_start, annual_end, inclusive="both")].copy()
            rows["forecast_cutoff"] = cutoff
            work.append(rows)
        panels = pd.read_csv(args.panels_csv)
        panel_map = {
            str(cutoff): frame.assign(reference_weight=1.0 / len(frame)).reset_index(drop=True)
            for cutoff, frame in panels.groupby("forecast_cutoff", sort=False)
        }
    else:
        candidates = attach_registry(instances, registry)
        panel_map = {}
        for cutoff_row in cutoffs.itertuples(index=False):
            cutoff = str(cutoff_row.forecast_cutoff)
            if selected is not None and cutoff not in selected:
                continue
            cutoff_time = pd.Timestamp(cutoff)
            current_start = cutoff_time - pd.DateOffset(months=args.current_months) + pd.Timedelta(days=1)
            # Keep a fixed retrospective window relative to the cutoff, then
            # exclude the current query period immediately before the cutoff.
            past_start = cutoff_time - pd.DateOffset(years=args.past_years)
            past_end = current_start - pd.Timedelta(days=1)
            current = candidates.loc[candidates["collection_date"].between(current_start, cutoff_time, inclusive="both")]
            current = deduplicate_strains(current).copy()
            if current.empty:
                raise ValueError(f"No current strains remain in {current_start.date()}..{cutoff_time.date()}.")
            current["forecast_cutoff"] = cutoff
            work.append(current)
            panel_map[cutoff] = build_past_strain_panel(
                candidates,
                current["query_id"],
                start=past_start,
                end=past_end,
                reference_passage=args.reference_passage,
                panel_size=args.past_panel_size,
                strata_months=args.representative_strata_months,
                sentinel_fraction=args.representative_sentinel_fraction,
            )
    if not work:
        raise ValueError("No cutoff work items were selected.")
    combined = pd.concat(work, ignore_index=True)
    if args.reference_mode == "serum":
        combined = attach_registry(combined, registry)
        combined = (
            combined.sort_values(["forecast_cutoff", "collection_date", "query_id"], kind="stable")
            .drop_duplicates(["forecast_cutoff", "query_id"], keep="last")
            .reset_index(drop=True)
        )
    owners = combined.apply(lambda row: stable_owner(str(row.forecast_cutoff), row, world_size), axis=1)
    combined = combined.loc[owners.eq(rank)].reset_index(drop=True)
    missing = set(combined["forecast_cutoff"]) - set(panel_map)
    if missing:
        raise ValueError(f"Missing panel(s) for cutoff(s): {sorted(missing)}")
    return combined, panel_map


def make_batch(
    frame: pd.DataFrame,
    cache: GpuAlignedCache,
    passage_to_id: dict[str, int],
    model: SerumMutationSetMinusModel,
    query_passage: str,
    device: torch.device,
) -> SerumMutationSetBatch:
    references = [cache.get(str(row.seq_id_a), str(row.serum_ha1)) for row in frame.itertuples(index=False)]
    queries = [cache.get(str(row.embedding_id), str(row.ha1_sequence)) for row in frame.itertuples(index=False)]
    serum_passage = torch.tensor([passage_to_id[normalize_passage(value)] for value in frame.serumPassCat], device=device)
    query_value = passage_to_id[normalize_passage(query_passage)]
    query_passage_ids = torch.full((len(frame), 1), query_value, dtype=torch.long, device=device)
    return SerumMutationSetBatch(
        reference_ha=torch.stack([item[0] for item in references]),
        query_ha=torch.stack([item[0] for item in queries])[:, None],
        reference_aa=torch.stack([item[3] for item in references]),
        query_aa=torch.stack([item[3] for item in queries])[:, None],
        reference_aligned_mask=torch.stack([item[1] for item in references]),
        query_aligned_mask=torch.stack([item[1] for item in queries])[:, None],
        reference_embedding_mask=torch.stack([item[2] for item in references]),
        query_embedding_mask=torch.stack([item[2] for item in queries])[:, None],
        serum_passage=serum_passage,
        query_passage=query_passage_ids,
        passage_pair=serum_passage[:, None] * model.config.passage_vocab_size + query_passage_ids,
        subtype=torch.zeros(len(frame), dtype=torch.long, device=device),
        query_mask=torch.ones(len(frame), 1, device=device),
    )


def predict_query(
    query: pd.Series, panel: pd.DataFrame, model: SerumMutationSetMinusModel, cache: GpuAlignedCache,
    passage_to_id: dict[str, int], query_passage: str, device: torch.device, batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    pairs = panel.copy()
    pairs["embedding_id"] = query.embedding_id
    pairs["ha1_sequence"] = query.ha1_sequence
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            part = pairs.iloc[start:start + batch_size]
            output = model(make_batch(part, cache, passage_to_id, model, query_passage, device))
            values.append(output["mean"][:, 0].detach().cpu().numpy())
    distances = np.concatenate(values)
    return pairs, distances


def already_completed(connection: sqlite3.Connection, cutoff: str, row: pd.Series) -> bool:
    result = connection.execute(
        "SELECT 1 FROM query_summary WHERE forecast_cutoff=? AND query_id=?",
        (cutoff, row.query_id),
    ).fetchone()
    return result is not None


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Deterministic weighted empirical quantile for compressed panels."""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def persist_query(connection: sqlite3.Connection, cutoff: str, query: pd.Series, pairs: pd.DataFrame, distances: np.ndarray, query_passage: str) -> None:
    now = pd.Timestamp.utcnow().isoformat()
    weights = pairs["reference_weight"].to_numpy(dtype=float)
    rows = [
        (cutoff, query.fasta_sha256, int(query.sequence_index), query.sequence_sha256, query.query_id,
         str(pair.seq_id_a), str(pair.serumPassCat), str(pair.serumName), str(pair.serum_date),
         float(pair.reference_weight), float(distance), now)
        for pair, distance in zip(pairs.itertuples(index=False), distances, strict=True)
    ]
    summary = (
        cutoff, query.fasta_sha256, int(query.sequence_index), query.sequence_sha256, query.query_id,
        query.embedding_id, query_passage, len(distances), float(np.average(distances, weights=weights)),
        weighted_quantile(distances, weights, 0.50), weighted_quantile(distances, weights, 0.10),
        weighted_quantile(distances, weights, 0.90), float(np.min(distances)),
        float(np.max(distances)), now,
    )
    for attempt in range(8):
        try:
            with connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO pair_scores (
                        forecast_cutoff, fasta_sha256, sequence_index, sequence_sha256, query_id,
                        serum_seq_id, serum_passage, serum_name, serum_date, reference_weight,
                        predicted_distance, scored_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.execute("INSERT OR REPLACE INTO query_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", summary)
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower() or attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    rank, world_size, device_name = distributed_context(args.device)
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SerumMutationSetMinusModel(checkpoint["model_config"], load_ha_distance_matrix(args.distance_matrix)).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    passage_to_id = {normalize_passage(key): int(value) for key, value in checkpoint["passage_to_id"].items()}
    normalized_query_passage = normalize_passage(args.query_passage)
    if normalized_query_passage not in passage_to_id:
        raise ValueError(f"Query passage {args.query_passage!r} is absent from checkpoint vocabulary.")
    normalized_reference_passage = normalize_passage(args.reference_passage)
    if args.reference_mode == "past_strains" and normalized_reference_passage not in passage_to_id:
        raise ValueError(f"Reference passage {args.reference_passage!r} is absent from checkpoint vocabulary.")
    work, panels = load_work(args, rank, world_size)
    if args.max_items < 0:
        raise ValueError("--max-items must be non-negative")
    if args.max_items:
        work = work.iloc[:args.max_items].copy()
    if args.output_sqlite is None:
        filename = (
            "antigenicity_scores.sqlite"
            if args.reference_mode == "serum"
            else f"past_strain_{args.past_years}y_excluding_recent{args.current_months}m"
            + (f"_coreset{args.past_panel_size}" if args.past_panel_size else "")
            + ".sqlite"
        )
        output_sqlite = PROJECT_ROOT / "outputs/h1n1_antigenicity" / filename
    else:
        output_sqlite = args.output_sqlite.expanduser().resolve()
    run_metadata = {"reference_mode": args.reference_mode}
    if args.reference_mode == "past_strains":
        run_metadata.update(
            current_months=str(args.current_months),
            past_years=str(args.past_years),
            reference_passage=normalized_reference_passage,
            past_panel_size=str(args.past_panel_size),
            representative_strata_months=str(args.representative_strata_months),
            representative_sentinel_fraction=str(args.representative_sentinel_fraction),
            representative_algorithm="temporal_weighted_farthest_point_with_sentinels_rebalanced_v2" if args.past_panel_size else "none",
        )
    connection = initialize_database(output_sqlite, run_metadata)
    cpu_cache = EmbeddingCache([args.new_embedding_dir.expanduser().resolve(), args.base_embedding_dir.expanduser().resolve()], args.cache_items)
    cache = GpuAlignedCache(cpu_cache, device, args.gpu_cache_gb)
    pending = [row for _, row in work.iterrows() if not already_completed(connection, str(row.forecast_cutoff), row)]
    progress = tqdm(pending, desc=f"rank {rank} antigenicity", dynamic_ncols=True)
    active_cutoff: str | None = None
    for query in progress:
        cutoff = str(query.forecast_cutoff)
        if cutoff != active_cutoff:
            cache.activate_reference_panel(panels[cutoff])
            active_cutoff = cutoff
        pairs, distances = predict_query(query, panels[cutoff], model, cache, passage_to_id, normalized_query_passage, device, args.batch_size)
        persist_query(connection, cutoff, query, pairs, distances, normalized_query_passage)
        progress.set_postfix(panel=len(distances), refresh=False)
    connection.close()
    print(json.dumps({
        "rank": rank,
        "world_size": world_size,
        "device": device_name,
        "reference_mode": args.reference_mode,
        "output_sqlite": str(output_sqlite),
        "assigned": len(work),
        "completed_now": len(pending),
        **cache.stats(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
