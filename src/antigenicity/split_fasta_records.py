#!/usr/bin/env python3
"""Split a two-line protein FASTA by stable record hash without splitting records."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fasta", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    lines = args.input_fasta.read_text(encoding="ascii").splitlines()
    if len(lines) % 2 or any(not lines[idx].startswith(">") for idx in range(0, len(lines), 2)):
        raise ValueError("Expected exactly one unwrapped sequence line per FASTA header.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f"shard_{idx:02d}.fasta" for idx in range(args.shards)]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Output shard already exists; use --overwrite to replace it.")
    handles = [path.open("w", encoding="ascii", newline="\n") for path in paths]
    counts = [0] * args.shards
    try:
        for idx in range(0, len(lines), 2):
            header, sequence = lines[idx], lines[idx + 1]
            shard = int(hashlib.sha256(header.encode("ascii")).hexdigest(), 16) % args.shards
            handles[shard].write(f"{header}\n{sequence}\n")
            counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()
    print({"shards": args.shards, "records": sum(counts), "records_per_shard": counts})


if __name__ == "__main__":
    main()
