#!/usr/bin/env python3
import sys
from pathlib import Path


def records(path):
    header, parts = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith('>'):
                if header:
                    yield header, ''.join(parts).upper()
                header, parts = line.rstrip(), []
            else:
                parts.append(line.strip())
    if header:
        yield header, ''.join(parts).upper()


input_path = Path(sys.argv[1])
output_path = input_path.with_name(f'deduplicated.fasta')
seen, kept = set(), 0
with output_path.open('w') as output:
    for header, sequence in records(input_path):
        if sequence and sequence not in seen:
            seen.add(sequence); output.write(f'{header}\n{sequence}\n'); kept += 1
print(f'Kept {kept:,} unique sequences: {output_path}')
