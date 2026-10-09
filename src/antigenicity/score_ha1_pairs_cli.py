#!/usr/bin/env python3
"""Multi-process CLI scorer for a saved direct HA1 pair table."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.antigenicity.score_ha1_pairs import HA1PairScorer


def parse_gpu_ids(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("GPU IDs must be distinct non-negative integers.")
    return result


def normalize_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the direct scorer schema or fluProfiler serum-task CSV."""
    direct = {"ha1_sequence_recent", "ha1_sequence_past"}
    task = {"seq_id_a", "seq_id_c", "serumHA", "virusHA"}
    if direct.issubset(frame.columns):
        return frame
    if not task.issubset(frame.columns):
        raise KeyError(f"pairs must contain either {sorted(direct)} or {sorted(task)}")
    return pd.DataFrame(
        {
            "ha1_sequence_recent": frame["virusHA"].str.replace("-", "", regex=False),
            "ha1_sequence_past": frame["serumHA"].str.replace("-", "", regex=False),
            "embedding_id_recent": frame["seq_id_c"].astype(str),
            "embedding_id_past": frame["seq_id_a"].astype(str),
        },
        index=frame.index,
    )


def worker(input_path: Path, output_path: Path, gpu_id: int, batch_size: int, cache_items: int | None) -> None:
    pairs = normalize_pairs(pd.read_pickle(input_path))
    scorer = HA1PairScorer(device=f"cuda:{gpu_id}", batch_size=batch_size, gpu_cache_items=cache_items)
    # CSV preserves the aligned strings but not pandas.attrs, so recreate the
    # lightweight source-position metadata once per worker before scoring.
    scorer.precompute_pair_alignments(pairs, progress=False)
    scorer.score_pairs(pairs, progress=False).to_pickle(output_path)


def score(args: argparse.Namespace) -> None:
    original_pairs = pd.read_csv(args.pairs)
    pairs = normalize_pairs(original_pairs)
    shards: list[list[pd.DataFrame]] = [[] for _ in args.gpu_ids]
    loads = [0] * len(shards)
    for _, group in pairs.groupby(args.reference_column, sort=False):
        target = min(range(len(shards)), key=loads.__getitem__)
        shards[target].append(group)
        loads[target] += len(group)

    with tempfile.TemporaryDirectory(prefix="ha1_pair_score_") as temporary:
        root = Path(temporary)
        commands: list[subprocess.Popen[str]] = []
        outputs: list[Path] = []
        for worker_index, (gpu_id, groups) in enumerate(zip(args.gpu_ids, shards, strict=True)):
            input_path, output_path = root / f"pairs_{worker_index}.pkl", root / f"scores_{worker_index}.pkl"
            pd.concat(groups).to_pickle(input_path)
            outputs.append(output_path)
            cache_argument = -1 if args.gpu_cache_items is None else args.gpu_cache_items
            commands.append(subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--input", str(input_path), "--output", str(output_path), "--gpu", str(gpu_id),
                "--batch-size", str(args.batch_size), "--gpu-cache-items", str(cache_argument),
            ], cwd=PROJECT_ROOT))
        with tqdm(total=len(commands), desc="GPU workers", unit="worker") as bar:
            pending = set(range(len(commands)))
            while pending:
                completed = [index for index in pending if commands[index].poll() is not None]
                for index in completed:
                    if commands[index].returncode:
                        raise RuntimeError(f"GPU worker {args.gpu_ids[index]} failed with exit code {commands[index].returncode}")
                    pending.remove(index)
                    bar.update(1)
                if pending:
                    time.sleep(1)
        scores = pd.concat([pd.read_pickle(path) for path in outputs])
    original_pairs["antigenicity_distance"] = scores.reindex(original_pairs.index)
    if original_pairs["antigenicity_distance"].isna().any():
        raise RuntimeError("Some pair scores are missing after worker merge.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_pairs.to_csv(args.output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-cache-items", type=int, default=-1, help="-1 retains all embeddings on each worker GPU.")
    parser.add_argument("--query-column", default="ha1_sequence_recent")
    parser.add_argument("--reference-column", default="ha1_sequence_past")
    args = parser.parse_args()
    cache_items = None if args.gpu_cache_items == -1 else args.gpu_cache_items
    if args.worker:
        worker(args.input, args.output, args.gpu, args.batch_size, cache_items)
    else:
        if args.pairs is None or args.output is None or args.gpu_ids is None:
            parser.error("--pairs, --output, and --gpu-ids are required")
        args.gpu_cache_items = cache_items
        score(args)


if __name__ == "__main__":
    main()
