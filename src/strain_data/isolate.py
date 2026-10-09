"""Read isolate metadata from GISAID-style protein FASTA headers."""

from pathlib import Path

import pandas as pd


COLUMNS = [
    "virus_name", "isolate_id", "type", "passage_details", "lineage",
    "clade", "collection_date", "submitter", "sample_id_provider",
    "sample_id_submitting_lab", "last_modified", "originating_lab",
    "submitting_lab", "gene_name", "protein_accession_no",
]


def read_isolate_headers(path: str | Path) -> pd.DataFrame:
    """Stream FASTA headers into a metadata table without loading sequences."""

    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.startswith(">"):
                fields = [value.strip() for value in line[1:].strip().split("|")]
                if len(fields) != len(COLUMNS):
                    raise ValueError(f"Expected {len(COLUMNS)} header fields, got {len(fields)}")
                rows.append(fields)
    result = pd.DataFrame(rows, columns=COLUMNS)
    result["collection_date"] = pd.to_datetime(result["collection_date"], errors="coerce")
    return result
