#!/usr/bin/env python3
"""Scripted version of H1N1_score.ipynb (the scoring-prep filtering).

The pipeline is unchanged from the notebook + deduplicate_fasta.py:

  1) exact-sequence deduplication (same rule as deduplicate_fasta.py:
     first occurrence wins, header ignored);
  2) keep records whose header contains the segment tag (default "|HA|");
  3) keep sequences whose length is >= --min-length (default 560);
  4) write one-line-per-sequence FASTA for the scorer.

The only deliberate difference from the raw GISAID files: spaces in headers
are replaced with underscores so that ``header.split()[0]`` downstream stays
a unique, lossless identifier ("A/Hong Kong/..." would otherwise truncate to
"A/Hong" and collide).

Example:
    python src/make_ha_unaligned.py \
      --input "H3N2_protein/H3N2_HA_all.fasta" \
      --output src/ha_unaglined_H3.fasta
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def records(path: Path):
    """Yield (header_without_gt, upper_sequence) from a FASTA file.

    Handles both single-line and 80-char wrapped sequences.
    """

    header, parts = None, []
    with path.open(encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts).upper()
                header, parts = line[1:].strip(), []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts).upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw GISAID protein FASTA (dedup happens here)")
    parser.add_argument("--output", type=Path, required=True, help="one-line FASTA for the scorer")
    parser.add_argument("--min-length", type=int, default=560, help="minimum sequence length in aa (default 560, as in the notebook)")
    parser.add_argument("--segment-tag", default="|HA|", help="header tag required to keep a record")
    args = parser.parse_args()

    started = time.perf_counter()
    seen: set[str] = set()
    total = duplicates = in_ha = kept = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for header, sequence in records(args.input):
            total += 1
            if not sequence or sequence in seen:
                duplicates += 1
                continue
            seen.add(sequence)
            if args.segment_tag not in header:
                continue
            in_ha += 1
            if len(sequence) < args.min_length:
                continue
            kept += 1
            out.write(f">{header.replace(' ', '_')}\n{sequence}\n")
            if kept % 50_000 == 0:
                print(f"  kept {kept:,} ...", flush=True)

    elapsed = time.perf_counter() - started
    print(f"input records           : {total:,}")
    print(f"exact duplicates dropped: {duplicates:,}")
    print(f"unique sequences        : {len(seen):,}")
    print(f"with segment tag        : {in_ha:,}")
    print(f"kept (len >= {args.min_length})   : {kept:,}")
    print(f"output                  : {args.output}")
    print(f"elapsed                 : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
