#!/usr/bin/env python3
"""Report duplicate FASTA headers and sequences."""

import sys
from collections import defaultdict
from pathlib import Path


def records(path):
    header, parts = None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts).upper()
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts).upper()


def duplicate_groups(values):
    groups = defaultdict(list)
    for index, value in enumerate(values, 1):
        groups[value].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


path = Path(sys.argv[1])
items = list(records(path))
headers = [header for header, _ in items]
aligned = [seq for _, seq in items]
ungapped = [seq.replace("-", "").replace(".", "") for seq in aligned]

print(f"records: {len(items):,}")
print(f"file size: {path.stat().st_size:,} bytes")
print(f"empty sequences: {sum(not seq for seq in aligned):,}")
for label, values in [("headers", headers), ("aligned sequences", aligned),
                      ("ungapped sequences", ungapped)]:
    groups = duplicate_groups(values)
    duplicate_records = sum(len(group) - 1 for group in groups)
    print(f"duplicate {label}: {duplicate_records:,} records in {len(groups):,} groups")
    if groups:
        print(f"  first duplicate record numbers: {groups[0]}")
