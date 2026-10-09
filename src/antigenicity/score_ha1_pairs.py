"""Score direct H1 HA1 reference--query pairs with the antigenicity model."""

from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm.auto import tqdm

from src.antigenicity.common import FLUPROFILER_ROOT, PROJECT_ROOT
from src.antigenicity.score_h1n1_antigenicity_multi_gpu import (
    EmbeddingCache,
    make_batch,
)

if str(FLUPROFILER_ROOT) not in sys.path:
    sys.path.insert(0, str(FLUPROFILER_ROOT))

from experiments.serum_gate.train_serum_mutation_set import (  # noqa: E402
    encode_aligned_sequence,
    load_ha_distance_matrix,
    normalize_passage,
)
from src.fluprofiler.models.serum_mutation_set_model import SerumMutationSetMinusModel  # noqa: E402


_ALIGNMENT_METADATA_ATTR = "_ha1_327_alignment_metadata"


def _reference_ha1() -> str:
    """Consensus of the model's fixed, 327-column training alignment."""
    source = pd.read_csv(
        FLUPROFILER_ROOT / "data/dataset/H1_HA1_v1.0/processed/source.csv",
        usecols=["serumHA", "virusHA"],
    )
    sequences = pd.concat([source["serumHA"], source["virusHA"]]).dropna().astype(str)
    if not sequences.str.len().eq(327).all():
        raise ValueError("The antigenicity training alignment is not fixed at 327 columns.")
    return "".join(
        Counter(sequence[position] for sequence in sequences if sequence[position] != "-").most_common(1)[0][0]
        for position in range(327)
    )


def align_to_327(sequence: str, reference: str) -> tuple[str, tuple[int | None, ...]]:
    """Project a raw HA1 sequence onto the model's 327 reference coordinates."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    alignment = aligner.align(reference, sequence)[0]
    coordinates = alignment.coordinates
    residues = ["-"] * len(reference)
    source_positions: list[int | None] = [None] * len(reference)
    for (ref_start, seq_start), (ref_end, seq_end) in zip(coordinates.T, coordinates.T[1:]):
        ref_width, seq_width = int(ref_end - ref_start), int(seq_end - seq_start)
        if ref_width == seq_width:
            for offset in range(ref_width):
                residues[int(ref_start) + offset] = sequence[int(seq_start) + offset]
                source_positions[int(ref_start) + offset] = int(seq_start) + offset
    return "".join(residues), tuple(source_positions)


def add_327_aligned_columns(
    pairs: pd.DataFrame,
    *,
    query_column: str = "ha1_sequence_recent",
    reference_column: str = "ha1_sequence_past",
    query_aligned_column: str = "ha1_aligned_recent",
    reference_aligned_column: str = "ha1_aligned_past",
    progress: bool = True,
) -> pd.DataFrame:
    """Add fixed-width, gapped HA1 alignments to ``pairs`` before scoring.

    The raw sequence columns are retained because embeddings are indexed in
    their ungapped coordinates.  This function mutates and returns ``pairs``.
    """
    if not {query_column, reference_column}.issubset(pairs.columns):
        raise KeyError(f"pairs must contain {query_column!r} and {reference_column!r}")
    sequences = pd.concat([pairs[query_column], pairs[reference_column]], ignore_index=True)
    sequences = sequences.dropna().astype(str).str.replace("-", "", regex=False).str.upper().drop_duplicates()
    sequences = sequences.loc[sequences.str.len().between(322, 332)]
    reference = _reference_ha1()
    alignments = {
        sequence: align_to_327(sequence, reference)
        for sequence in tqdm(sequences, desc="Aligning HA1", unit="sequence", disable=not progress)
    }
    pairs[query_aligned_column] = pairs[query_column].astype(str).str.replace("-", "", regex=False).str.upper().map(
        {sequence: aligned for sequence, (aligned, _) in alignments.items()}
    )
    pairs[reference_aligned_column] = pairs[reference_column].astype(str).str.replace("-", "", regex=False).str.upper().map(
        {sequence: aligned for sequence, (aligned, _) in alignments.items()}
    )
    pairs.attrs[_ALIGNMENT_METADATA_ATTR] = alignments
    return pairs


class _ReferenceAlignedCache:
    """Align each raw embedding to the model's 327 fixed coordinates."""

    def __init__(
        self,
        cpu_cache: EmbeddingCache,
        device: torch.device,
        reference: str,
        max_items: int | None = 128,
        precomputed_alignments: dict[tuple[str, str], tuple[str, tuple[int | None, ...]]] | None = None,
    ) -> None:
        self.cpu_cache, self.device, self.reference, self.max_items = cpu_cache, device, reference, max_items
        self.precomputed_alignments = precomputed_alignments if precomputed_alignments is not None else {}
        self.values: OrderedDict[tuple[str, str], tuple[torch.Tensor, ...]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, embedding_id: str, sequence: str) -> tuple[torch.Tensor, ...]:
        key = (embedding_id, sequence)
        value = self.values.pop(key, None)
        if value is not None:
            self.hits += 1
            self.values[key] = value
            return value
        self.misses += 1
        alignment = self.precomputed_alignments.get(key)
        if alignment is None:
            aligned, positions = align_to_327(sequence, self.reference)
        else:
            aligned, positions = alignment
        matrix = self.cpu_cache.get(embedding_id)
        if matrix.shape[0] == len(sequence) + 2:
            matrix = matrix[1:-1]
        if matrix.shape[0] != len(sequence):
            raise ValueError(f"Embedding {embedding_id} does not match its HA1 sequence length.")
        matrix = matrix.to(self.device, non_blocking=True)
        projected = matrix.new_zeros((327, matrix.shape[1]))
        embedding_mask = matrix.new_zeros(327)
        for target_position, source_position in enumerate(positions):
            if source_position is not None:
                projected[target_position] = matrix[source_position]
                embedding_mask[target_position] = 1.0
        value = (
            projected,
            matrix.new_ones(327),
            embedding_mask,
            encode_aligned_sequence(aligned).to(self.device),
        )
        self.values[key] = value
        if self.max_items is not None and len(self.values) > self.max_items:
            self.values.popitem(last=False)
        return value

    def stats(self) -> dict[str, int | float | None]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "resident_embeddings": len(self.values),
            "max_items": self.max_items,
        }


class HA1PairScorer:
    """Load the model once, then score one value for every HA1 pair."""

    def __init__(
        self,
        *,
        device: str | None = None,
        gpu_ids: str | Sequence[int] | None = None,
        batch_size: int = 64,
        gpu_cache_items: int | None = 128,
    ) -> None:
        """Create a scorer, optionally splitting each batch across ``gpu_ids``.

        ``gpu_ids`` accepts ``"0,2,3"`` or a sequence such as ``[0, 2, 3]``.
        Set ``gpu_cache_items=None`` to retain every aligned embedding on its
        assigned GPU for this scorer's lifetime; use this only when GPU memory
        can accommodate all unique sequences.
        It is deliberately explicit: CUDA's visible-device configuration is
        left unchanged, so the supplied IDs are the physical CUDA indices.
        """
        if gpu_ids is not None and device is not None:
            raise ValueError("Specify either device or gpu_ids, not both.")
        self.gpu_ids = self._parse_gpu_ids(gpu_ids)
        if self.gpu_ids is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("gpu_ids was provided, but CUDA is unavailable.")
            available = torch.cuda.device_count()
            invalid = [gpu_id for gpu_id in self.gpu_ids if gpu_id >= available]
            if invalid:
                raise ValueError(f"GPU IDs {invalid} are unavailable; CUDA exposes 0 through {available - 1}.")
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        if gpu_cache_items is not None and gpu_cache_items < 0:
            raise ValueError("gpu_cache_items must be non-negative or None.")
        self.gpu_cache_items = gpu_cache_items
        checkpoint_path = FLUPROFILER_ROOT / "results/H1_HA1_v1.0/SCMS-FiLM-H1/full_data_interpretation/H1N1/checkpoints/epoch_0050.pth"
        distance_path = FLUPROFILER_ROOT / "ha1_distance_no_bias_329.npy"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.base_model = SerumMutationSetMinusModel(
            checkpoint["model_config"], load_ha_distance_matrix(distance_path)
        ).to(self.device).eval()
        self.base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Keep one independent model per card.  Unlike DataParallel this does
        # not require NCCL peer communication, which is often unavailable on
        # workstation GPU partitions.
        self.models = [self.base_model]
        self.devices = [self.device]
        for gpu_id in (self.gpu_ids or ())[1:]:
            worker_device = torch.device(f"cuda:{gpu_id}")
            worker_model = SerumMutationSetMinusModel(
                checkpoint["model_config"], load_ha_distance_matrix(distance_path)
            ).to(worker_device).eval()
            worker_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.models.append(worker_model)
            self.devices.append(worker_device)
        self.model = self.base_model
        self.passage_to_id = {
            normalize_passage(key): int(value) for key, value in checkpoint["passage_to_id"].items()
        }
        self.passage = normalize_passage("<CELL>")
        if self.passage not in self.passage_to_id:
            raise ValueError("Checkpoint does not support the <CELL> passage category.")

        self.pair_embedding_root = PROJECT_ROOT / "data/raw/fasta/strain/H1N1/embedding"
        self.embedding_ids = self._embedding_ids()
        self.precomputed_alignments: dict[tuple[str, str], tuple[str, tuple[int | None, ...]]] = {}
        embedding_roots = [self.pair_embedding_root, FLUPROFILER_ROOT / "data/embedding/files"]
        reference = _reference_ha1()
        self.caches = [
            _ReferenceAlignedCache(
                EmbeddingCache(embedding_roots, max_items=512),
                worker_device,
                reference,
                max_items=self.gpu_cache_items,
                precomputed_alignments=self.precomputed_alignments,
            )
            for worker_device in self.devices
        ]
        self.cache = self.caches[0]

    @staticmethod
    def _parse_gpu_ids(gpu_ids: str | Sequence[int] | None) -> tuple[int, ...] | None:
        if gpu_ids is None:
            return None
        values = gpu_ids.split(",") if isinstance(gpu_ids, str) else gpu_ids
        try:
            parsed = tuple(int(value) for value in values)
        except (TypeError, ValueError) as error:
            raise ValueError("gpu_ids must be comma-separated CUDA indices or a sequence of integers.") from error
        if not parsed or any(gpu_id < 0 for gpu_id in parsed) or len(set(parsed)) != len(parsed):
            raise ValueError("gpu_ids must contain one or more distinct non-negative CUDA indices.")
        return parsed

    def _pair_embedding_ids(self, work: pd.DataFrame, query_column: str, reference_column: str) -> dict[str, str]:
        """Map sequences to the ``seq_N`` IDs recorded alongside their embeddings.

        The IDs are assigned when ``unique_ha1.fasta`` is generated, not when a
        subset of pairs is scored.  Re-enumerating ``work`` makes an ID point at
        a different sequence whenever its ordering or membership changes.
        """
        fasta_path = self.pair_embedding_root.parent / "unique_ha1.fasta"
        wanted = set(pd.concat([work[query_column], work[reference_column]], ignore_index=True))
        mapping: dict[str, str] = {}
        sequence_id: str | None = None
        sequence_parts: list[str] = []

        def add_record() -> None:
            if sequence_id is None:
                return
            sequence = "".join(sequence_parts).replace("-", "").upper()
            if (
                sequence in wanted
                and (self.pair_embedding_root / f"matrix_{sequence_id}.pt").is_file()
            ):
                mapping[sequence] = sequence_id

        with fasta_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    add_record()
                    sequence_id = line[1:].split(maxsplit=1)[0]
                    sequence_parts = []
                else:
                    sequence_parts.append(line)
        add_record()
        return mapping

    def precompute_pair_alignments(
        self,
        pairs: pd.DataFrame,
        *,
        query_column: str = "ha1_sequence_recent",
        reference_column: str = "ha1_sequence_past",
        query_aligned_column: str = "ha1_aligned_recent",
        reference_aligned_column: str = "ha1_aligned_past",
        progress: bool = True,
    ) -> int:
        """Pre-align all scorable unique HA1 sequences once in the main process.

        The result is lightweight alignment metadata shared by every GPU cache;
        raw 2560-dimensional embeddings remain loaded lazily on their assigned
        GPU, where the aligned tensor cache handles reuse.
        """
        if not {query_column, reference_column}.issubset(pairs.columns):
            raise KeyError(f"pairs must contain {query_column!r} and {reference_column!r}")
        work = pairs[[query_column, reference_column]].copy()
        for column in (query_column, reference_column):
            work[column] = work[column].astype(str).str.replace("-", "", regex=False).str.upper()
        if {"embedding_id_recent", "embedding_id_past"}.issubset(pairs.columns):
            query_ids = pairs.loc[work.index, "embedding_id_recent"]
            reference_ids = pairs.loc[work.index, "embedding_id_past"]
        else:
            pair_embedding_ids = self._pair_embedding_ids(work, query_column, reference_column)
            query_ids = work[query_column].map(pair_embedding_ids).fillna(work[query_column].map(self.embedding_ids))
            reference_ids = work[reference_column].map(pair_embedding_ids).fillna(work[reference_column].map(self.embedding_ids))
        entries = pd.concat(
            [
                pd.DataFrame({"embedding_id": query_ids, "sequence": work[query_column]}),
                pd.DataFrame({"embedding_id": reference_ids, "sequence": work[reference_column]}),
            ],
            ignore_index=True,
        ).dropna().drop_duplicates()
        entries = entries.loc[entries["sequence"].str.len().between(322, 332)]
        added = 0
        with tqdm(entries.itertuples(index=False), total=len(entries), desc="Pre-aligning HA1", unit="sequence", disable=not progress) as bar:
            for row in bar:
                key = (str(row.embedding_id), str(row.sequence))
                if key not in self.precomputed_alignments:
                    self.precomputed_alignments[key] = align_to_327(key[1], self.cache.reference)
                    added += 1
        # Keep the actual fixed-width alignments with the pair table.  The
        # cache additionally retains source positions, which are required to
        # project a raw embedding when the sequence contains insertions.
        aligned_by_sequence = {
            sequence: aligned
            for (_, sequence), (aligned, _) in self.precomputed_alignments.items()
        }
        pairs[query_aligned_column] = work[query_column].map(aligned_by_sequence)
        pairs[reference_aligned_column] = work[reference_column].map(aligned_by_sequence)
        return added

    @staticmethod
    def _embedding_ids() -> dict[str, str]:
        """Map every available 327-aa HA1 sequence to its embedding ID."""
        query_registry = pd.read_csv(
            PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_ha1_registry.csv",
            usecols=["ha1_sequence", "embedding_id"],
        )
        base_registry = pd.read_csv(
            FLUPROFILER_ROOT / "data/embedding/registry/sequences.csv",
            usecols=["seq_id", "segment", "sequence"],
        )
        base_registry = base_registry.loc[base_registry["segment"].eq("HA")].copy()
        base_registry["ha1_sequence"] = base_registry["sequence"].str.replace("-", "", regex=False).str.upper()
        base_registry = base_registry.loc[base_registry["ha1_sequence"].str.len().eq(327), ["ha1_sequence", "seq_id"]]
        base_registry = base_registry.rename(columns={"seq_id": "embedding_id"})
        registry = pd.concat([query_registry, base_registry], ignore_index=True)
        registry["ha1_sequence"] = registry["ha1_sequence"].str.replace("-", "", regex=False).str.upper()
        embedding_roots = [
            PROJECT_ROOT / "data/raw/fasta/strain/H1N1/embedding",
            FLUPROFILER_ROOT / "data/embedding/files",
        ]
        registry = registry.loc[
            registry["embedding_id"].map(
                lambda embedding_id: any((root / f"matrix_{embedding_id}.pt").is_file() for root in embedding_roots)
            )
        ]
        registry = registry.drop_duplicates("ha1_sequence", keep="first")
        return dict(zip(registry["ha1_sequence"], registry["embedding_id"], strict=True))

    def score_pairs(
        self,
        pairs: pd.DataFrame,
        *,
        query_column: str = "ha1_sequence_recent",
        reference_column: str = "ha1_sequence_past",
        progress: bool = True,
    ) -> pd.Series:
        """Return one predicted antigenic distance per input row.

        ``query_column`` is the recent HA1 and ``reference_column`` is the
        historical HA1. Raw 322--332-aa HA1 sequences are globally aligned to
        the model's fixed 327-position training coordinates before scoring.
        """
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not {query_column, reference_column}.issubset(pairs.columns):
            raise KeyError(f"pairs must contain {query_column!r} and {reference_column!r}")

        work = pairs[[query_column, reference_column]].copy()
        work[query_column] = work[query_column].astype(str).str.replace("-", "", regex=False).str.upper()
        work[reference_column] = work[reference_column].astype(str).str.replace("-", "", regex=False).str.upper()
        # Build this before removing invalid lengths: unique_ha1.fasta was numbered
        # from every non-null HA1 sequence in the two source cohorts.
        invalid = ~work[query_column].str.len().between(322, 332) | ~work[reference_column].str.len().between(322, 332)
        result = pd.Series(float("nan"), index=pairs.index, name="antigenicity_distance")
        work = work.loc[~invalid].copy()
        if work.empty:
            return result

        if {"embedding_id_recent", "embedding_id_past"}.issubset(pairs.columns):
            query_ids = pairs.loc[work.index, "embedding_id_recent"]
            reference_ids = pairs.loc[work.index, "embedding_id_past"]
        else:
            pair_embedding_ids = self._pair_embedding_ids(work, query_column, reference_column)
            query_ids = work[query_column].map(pair_embedding_ids).fillna(work[query_column].map(self.embedding_ids))
            reference_ids = work[reference_column].map(pair_embedding_ids).fillna(work[reference_column].map(self.embedding_ids))
        scorable = query_ids.notna() & reference_ids.notna()
        work, query_ids, reference_ids = work.loc[scorable], query_ids.loc[scorable], reference_ids.loc[scorable]
        if work.empty:
            return result
        self._register_pair_alignment_metadata(pairs, work, query_ids, reference_ids, query_column, reference_column)

        model_input = pd.DataFrame(
            {
                "seq_id_a": reference_ids.to_numpy(),
                "serum_ha1": work[reference_column].to_numpy(),
                "serumPassCat": self.passage,
                "embedding_id": query_ids.to_numpy(),
                "ha1_sequence": work[query_column].to_numpy(),
                "_pair_index": work.index.to_numpy(),
            }
        )
        chunks: list[pd.DataFrame]
        if len(self.models) == 1:
            chunks = [model_input]
        else:
            # A cross join is query-major, so slicing rows makes every GPU load
            # the entire historical panel.  Shard by reference instead: each
            # historical embedding is loaded on only one GPU and stays there.
            chunks = self._reference_shards(model_input)
        total_batches = sum((len(chunk) + self.batch_size - 1) // self.batch_size for chunk in chunks)
        with tqdm(
            total=total_batches,
            desc="Scoring antigenicity",
            unit="batch",
            disable=not progress,
        ) as bar:
            if len(chunks) == 1:
                scored_indices, values = self._score_model_input(
                    chunks[0], self.models[0], self.caches[0], self.devices[0], bar.update
                )
            else:
                with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                    futures = [
                        executor.submit(
                            self._score_model_input,
                            chunk,
                            self.models[index],
                            self.caches[index],
                            self.devices[index],
                            bar.update,
                        )
                        for index, chunk in enumerate(chunks)
                    ]
                    outputs = [future.result() for future in futures]
                    scored_indices = [index for indices, _ in outputs for index in indices]
                    values = [value for _, output_values in outputs for value in output_values]
        result.loc[scored_indices] = values
        return result

    def _register_pair_alignment_metadata(
        self,
        pairs: pd.DataFrame,
        work: pd.DataFrame,
        query_ids: pd.Series,
        reference_ids: pd.Series,
        query_column: str,
        reference_column: str,
    ) -> None:
        """Reuse alignments prepared in the pair-construction notebook cell."""
        metadata = pairs.attrs.get(_ALIGNMENT_METADATA_ATTR, {})
        if not metadata:
            return
        for embedding_ids, sequences in (
            (query_ids, work[query_column]),
            (reference_ids, work[reference_column]),
        ):
            for embedding_id, sequence in zip(embedding_ids, sequences, strict=True):
                alignment = metadata.get(sequence)
                if alignment is not None:
                    self.precomputed_alignments.setdefault((str(embedding_id), sequence), alignment)

    def _reference_shards(self, model_input: pd.DataFrame) -> list[pd.DataFrame]:
        """Balance whole historical-reference groups across GPU workers."""
        groups: list[list[pd.DataFrame]] = [[] for _ in self.models]
        loads = [0] * len(self.models)
        for _, group in model_input.groupby(["seq_id_a", "serum_ha1"], sort=False):
            worker = min(range(len(self.models)), key=loads.__getitem__)
            groups[worker].append(group)
            loads[worker] += len(group)
        return [pd.concat(worker_groups, ignore_index=True) for worker_groups in groups if worker_groups]

    def _score_model_input(
        self,
        model_input: pd.DataFrame,
        model: SerumMutationSetMinusModel,
        cache: _ReferenceAlignedCache,
        device: torch.device,
        on_batch_complete: Callable[[int], object] | None = None,
    ) -> tuple[list[object], list[float]]:
        pair_indices: list[object] = []
        values: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(model_input), self.batch_size):
                part = model_input.iloc[start : start + self.batch_size]
                batch = make_batch(
                    part, cache,
                    self.passage_to_id, model, self.passage, device,
                )
                output = model(batch)
                pair_indices.extend(part["_pair_index"].tolist())
                values.extend(output["mean"][:, 0].detach().cpu().tolist())
                if on_batch_complete is not None:
                    on_batch_complete(1)
        return pair_indices, values

    def cache_stats(self) -> pd.DataFrame:
        """Return per-GPU and aggregate aligned-embedding cache statistics."""
        rows = [{"device": str(device), **cache.stats()} for device, cache in zip(self.devices, self.caches)]
        frame = pd.DataFrame(rows)
        total = int(frame["hits"].sum() + frame["misses"].sum())
        aggregate = {
            "device": "all",
            "hits": int(frame["hits"].sum()),
            "misses": int(frame["misses"].sum()),
            "hit_rate": float(frame["hits"].sum() / total) if total else 0.0,
            "resident_embeddings": int(frame["resident_embeddings"].sum()),
            "max_items": self.gpu_cache_items,
        }
        return pd.concat([frame, pd.DataFrame([aggregate])], ignore_index=True)


_default_scorer: HA1PairScorer | None = None


def score_ha1_pairs(
    pairs: pd.DataFrame,
    *,
    gpu_ids: str | Sequence[int] | None = None,
    batch_size: int = 64,
    gpu_cache_items: int | None = 128,
    progress: bool = True,
) -> pd.Series:
    """Score ``ha1_sequence_recent`` × ``ha1_sequence_past`` pairs."""
    global _default_scorer
    requested_gpu_ids = HA1PairScorer._parse_gpu_ids(gpu_ids)
    if (
        _default_scorer is None
        or _default_scorer.gpu_ids != requested_gpu_ids
        or _default_scorer.batch_size != batch_size
        or _default_scorer.gpu_cache_items != gpu_cache_items
    ):
        _default_scorer = HA1PairScorer(
            gpu_ids=gpu_ids,
            batch_size=batch_size,
            gpu_cache_items=gpu_cache_items,
        )
    return _default_scorer.score_pairs(pairs, progress=progress)


def missing_ha1_embeddings(
    pairs: pd.DataFrame,
    *,
    query_column: str = "ha1_sequence_recent",
    reference_column: str = "ha1_sequence_past",
) -> set[str]:
    """Return unique valid HA1 sequences in ``pairs`` without an embedding file."""
    columns = [query_column, reference_column]
    if not set(columns).issubset(pairs.columns):
        raise KeyError(f"pairs must contain {query_column!r} and {reference_column!r}")
    sequences = pd.concat([pairs[column] for column in columns]).dropna().astype(str)
    sequences = sequences.str.replace("-", "", regex=False).str.upper()
    sequences = set(sequences.loc[sequences.str.len().between(322, 332)])
    return sequences - set(HA1PairScorer._embedding_ids())
