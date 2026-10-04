#!/usr/bin/env python3
"""Resumable, communication-free multi-GPU H1N1 antigenicity scoring.

Each worker deterministically owns a subset of (forecast cutoff, sequence
identity) rows.  Workers communicate only through SQLite's atomic commits;
there is no DDP collective operation.  A completed query summary is the
resume marker, while the individual serum--virus model outputs are retained in
``pair_scores`` for audit and later re-aggregation.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLUPROFILER_ROOT = Path("/home/chenyh/workspace/fluProfiler")
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


IDENTITY_COLUMNS = ("fasta_sha256", "sequence_index", "sequence_sha256")


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
    parser.add_argument("--output-sqlite", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/antigenicity_scores.sqlite")
    parser.add_argument("--cutoff", action="append", default=[], help="Optional YYYY-MM-DD cutoff; repeatable.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-passage", default="<CELL>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-items", type=int, default=128)
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


def initialize_database(path: Path) -> sqlite3.Connection:
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
        """
    )
    connection.commit()
    return connection


def stable_owner(cutoff: str, row: pd.Series, world_size: int) -> int:
    identity = "|".join([cutoff, *(str(row[column]) for column in IDENTITY_COLUMNS)])
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16) % world_size


def load_work(args: argparse.Namespace, rank: int, world_size: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    instances = pd.read_csv(args.query_instances_csv)
    registry = pd.read_csv(args.query_registry_csv, usecols=["query_id", "ha1_sequence", "embedding_id"])
    cutoffs = pd.read_csv(args.cutoffs_csv)
    panels = pd.read_csv(args.panels_csv)
    instances["collection_date"] = pd.to_datetime(instances["collection_date"], errors="coerce").dt.normalize()
    if instances["collection_date"].isna().any():
        raise ValueError("Query instances include invalid collection_date values.")
    work: list[pd.DataFrame] = []
    selected = set(args.cutoff) if args.cutoff else None
    for cutoff_row in cutoffs.itertuples(index=False):
        cutoff = str(cutoff_row.forecast_cutoff)
        if selected is not None and cutoff not in selected:
            continue
        annual_start = pd.Timestamp(cutoff_row.annual_start)
        annual_end = pd.Timestamp(cutoff_row.annual_end)
        rows = instances.loc[instances["collection_date"].between(annual_start, annual_end, inclusive="both")].copy()
        rows["forecast_cutoff"] = cutoff
        work.append(rows)
    if not work:
        raise ValueError("No cutoff work items were selected.")
    combined = pd.concat(work, ignore_index=True).merge(registry, on="query_id", how="left", validate="many_to_one")
    if combined[["ha1_sequence", "embedding_id"]].isna().any().any():
        raise ValueError("At least one query instance has no HA1 registry record.")
    owners = combined.apply(lambda row: stable_owner(str(row.forecast_cutoff), row, world_size), axis=1)
    combined = combined.loc[owners.eq(rank)].reset_index(drop=True)
    panel_map = {
        str(cutoff): frame.reset_index(drop=True)
        for cutoff, frame in panels.groupby("forecast_cutoff", sort=False)
    }
    missing = set(combined["forecast_cutoff"]) - set(panel_map)
    if missing:
        raise ValueError(f"Missing panel(s) for cutoff(s): {sorted(missing)}")
    return combined, panel_map


def make_batch(
    frame: pd.DataFrame,
    cache: EmbeddingCache,
    passage_to_id: dict[str, int],
    model: SerumMutationSetMinusModel,
    query_passage: str,
    device: torch.device,
) -> SerumMutationSetBatch:
    references = [align_embedding_to_sequence(cache.get(str(row.seq_id_a)), str(row.serum_ha1)) for row in frame.itertuples(index=False)]
    queries = [align_embedding_to_sequence(cache.get(str(row.embedding_id)), str(row.ha1_sequence)) for row in frame.itertuples(index=False)]
    serum_passage = torch.tensor([passage_to_id[normalize_passage(value)] for value in frame.serumPassCat], device=device)
    query_value = passage_to_id[normalize_passage(query_passage)]
    query_passage_ids = torch.full((len(frame), 1), query_value, dtype=torch.long, device=device)
    return SerumMutationSetBatch(
        reference_ha=torch.stack([item[0] for item in references]).to(device),
        query_ha=torch.stack([item[0] for item in queries]).to(device)[:, None],
        reference_aa=torch.stack([item[3] for item in references]).to(device),
        query_aa=torch.stack([item[3] for item in queries]).to(device)[:, None],
        reference_aligned_mask=torch.stack([item[1] for item in references]).to(device),
        query_aligned_mask=torch.stack([item[1] for item in queries]).to(device)[:, None],
        reference_embedding_mask=torch.stack([item[2] for item in references]).to(device),
        query_embedding_mask=torch.stack([item[2] for item in queries]).to(device)[:, None],
        serum_passage=serum_passage,
        query_passage=query_passage_ids,
        passage_pair=serum_passage[:, None] * model.config.passage_vocab_size + query_passage_ids,
        subtype=torch.zeros(len(frame), dtype=torch.long, device=device),
        query_mask=torch.ones(len(frame), 1, device=device),
    )


def predict_query(
    query: pd.Series, panel: pd.DataFrame, model: SerumMutationSetMinusModel, cache: EmbeddingCache,
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
        "SELECT 1 FROM query_summary WHERE forecast_cutoff=? AND fasta_sha256=? AND sequence_index=? AND sequence_sha256=?",
        (cutoff, row.fasta_sha256, int(row.sequence_index), row.sequence_sha256),
    ).fetchone()
    return result is not None


def persist_query(connection: sqlite3.Connection, cutoff: str, query: pd.Series, pairs: pd.DataFrame, distances: np.ndarray, query_passage: str) -> None:
    now = pd.Timestamp.utcnow().isoformat()
    rows = [
        (cutoff, query.fasta_sha256, int(query.sequence_index), query.sequence_sha256, query.query_id,
         str(pair.seq_id_a), str(pair.serumPassCat), str(pair.serumName), str(pair.serum_date), float(distance), now)
        for pair, distance in zip(pairs.itertuples(index=False), distances, strict=True)
    ]
    summary = (
        cutoff, query.fasta_sha256, int(query.sequence_index), query.sequence_sha256, query.query_id,
        query.embedding_id, query_passage, len(distances), float(np.mean(distances)), float(np.median(distances)),
        float(np.quantile(distances, 0.10)), float(np.quantile(distances, 0.90)), float(np.min(distances)),
        float(np.max(distances)), now,
    )
    for attempt in range(8):
        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO pair_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
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
    work, panels = load_work(args, rank, world_size)
    connection = initialize_database(args.output_sqlite.expanduser().resolve())
    cache = EmbeddingCache([args.new_embedding_dir.expanduser().resolve(), args.base_embedding_dir.expanduser().resolve()], args.cache_items)
    pending = [row for _, row in work.iterrows() if not already_completed(connection, str(row.forecast_cutoff), row)]
    progress = tqdm(pending, desc=f"rank {rank} antigenicity", dynamic_ncols=True)
    for query in progress:
        cutoff = str(query.forecast_cutoff)
        pairs, distances = predict_query(query, panels[cutoff], model, cache, passage_to_id, normalized_query_passage, device, args.batch_size)
        persist_query(connection, cutoff, query, pairs, distances, normalized_query_passage)
        progress.set_postfix(panel=len(distances), refresh=False)
    connection.close()
    print(json.dumps({"rank": rank, "world_size": world_size, "device": device_name, "assigned": len(work), "completed_now": len(pending)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
