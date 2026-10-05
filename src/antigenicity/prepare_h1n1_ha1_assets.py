#!/usr/bin/env python3
"""Prepare deduplicated H1N1 HA1 assets for external fluProfiler inference.

This script deliberately performs only local file processing.  It never loads
the antigenic model or LucaVirus model, and it never edits either project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.antigenicity.common import (
    FLUPROFILER_ROOT,
    IDENTITY_COLUMNS,
    PROJECT_ROOT,
    prepare_output_directory,
    require_directory,
    require_file,
    write_json,
)
from src.strain_data.nextclade import BRANCH_COLUMN_PRIORITIES, read_nextclade_table

# LucaVirus explicitly tokenizes these standard and ambiguous protein symbols.
# fluProfiler maps non-canonical residues to its unknown amino-acid ID, so
# retaining them is safer and more reproducible than silently discarding data.
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUOJ")
CANONICAL_AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
EPI_PATTERN = re.compile(r"EPI_ISL_\d+")
UNKNOWN_VALUES = {"", "unknown", "unassigned", "na", "n/a", "none", "null", "-"}
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
DEFAULT_SAMPLE_AA_FASTA = PROJECT_ROOT / "data/raw/nextclade/H1N1/origin_seqs/AA/H1N1_All.fasta"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def translate_nt(sequence: str, frame: int) -> str:
    sequence = sequence.upper().replace("U", "T")
    residues = [CODON_TABLE.get(sequence[index:index + 3], "X") for index in range(frame, len(sequence) - 2, 3)]
    amino_acids = "".join(residues)
    return amino_acids[:-1] if amino_acids.endswith("*") else amino_acids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_epi_isl(value: object) -> str | None:
    match = EPI_PATTERN.search(str(value))
    return match.group(0) if match else None


def normalise_header(value: str) -> str:
    return value.strip().replace(" ", "_")


def normalise_unknown(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in UNKNOWN_VALUES else value


def parse_header(full_header: str) -> dict[str, str | int | None]:
    fields = [field.strip() for field in full_header.split("|")]
    sample_name = fields[0] if fields else full_header
    collection_date = fields[6] if len(fields) > 6 else None
    collection_date = normalise_unknown(collection_date)
    year = None
    if collection_date:
        year = pd.to_datetime(collection_date, errors="coerce")
        year = int(year.year) if not pd.isna(year) else None
    return {
        "sample_name": sample_name,
        "subtype": fields[2] if len(fields) > 2 else None,
        "clade_label": normalise_unknown(fields[3]) if len(fields) > 3 else None,
        "lineage": normalise_unknown(fields[4]) if len(fields) > 4 else None,
        "clade": normalise_unknown(fields[5]) if len(fields) > 5 else None,
        "collection_date": collection_date,
        "year": year,
        "time_source": "collection_date" if year is not None else None,
    }


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts).upper()))
                header, parts = line[1:].strip(), []
            elif header is None:
                raise ValueError(f"{path} starts with sequence content before the first FASTA header.")
            else:
                parts.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(parts).upper()))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-sqlite",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_raw_scores.sqlite",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_metadata.csv",
    )
    parser.add_argument(
        "--fluprofiler-registry",
        type=Path,
        default=FLUPROFILER_ROOT / "data/embedding/registry/sequences.csv",
    )
    parser.add_argument(
        "--fluprofiler-embedding-dir",
        type=Path,
        default=FLUPROFILER_ROOT / "data/embedding/files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input",
    )
    parser.add_argument(
        "--sample-aa-fasta",
        type=Path,
        help=(
            "Optional non-deduplicated full-HA amino-acid FASTA. When provided, "
            "query_instances.csv is expanded to the Nextclade TSV sampling rows "
            "by EPI_ISL, then mapped back to unique scored full-HA sequences. "
            f"If omitted, {DEFAULT_SAMPLE_AA_FASTA} is used when present."
        ),
    )
    parser.add_argument(
        "--sample-nt-fasta",
        type=Path,
        help=(
            "Optional non-deduplicated HA nucleotide FASTA. Records are translated "
            "in all three frames and mapped to unique scored full-HA amino-acid sequences."
        ),
    )
    parser.add_argument(
        "--nextclade-tsv",
        type=Path,
        default=PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv",
        help="Nextclade TSV whose seqName rows define the sampling instances for expansion.",
    )
    parser.add_argument("--label-mode", choices=sorted(BRANCH_COLUMN_PRIORITIES), default="nextclade")
    parser.add_argument(
        "--no-sample-expansion",
        action="store_true",
        help="Keep the legacy one-row-per-scored-unique-sequence query_instances.csv.",
    )
    parser.add_argument("--full-ha-length", type=int, default=566)
    parser.add_argument("--segment-tag", default="|HA|")
    parser.add_argument("--ha1-start", type=int, default=17, help="0-based full-HA start index.")
    parser.add_argument("--ha1-length", type=int, default=327)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scores(path: Path) -> pd.DataFrame:
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT fasta_sha256, sequence_index, sequence_id, sequence_sha256,
                   sequence, valid_length, raw_log_prob_sum, bert_score, scored_at_utc
            FROM sequence_scores
            WHERE bert_score IS NOT NULL
            ORDER BY fasta_sha256, sequence_index
            """,
            connection,
        )
    if frame.empty:
        raise ValueError("No completed BERT scores were found in sequence_scores.")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("SQLite score table contains duplicate sequence identities.")
    return frame


def load_metadata(path: Path) -> pd.DataFrame:
    required = [
        "fasta_sha256", "sequence_index", "sequence_sha256", "full_header",
        "collection_date", "clade", "lineage", "clade_label",
    ]
    frame = pd.read_csv(path, usecols=required)
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Metadata CSV contains duplicate sequence identities.")
    return frame


def sample_sequence_candidates(sequence: str, *, sequence_type: str, full_ha_length: int) -> set[str]:
    if sequence_type == "aa":
        amino_acids = sequence.strip().upper().replace("-", "")
        return {amino_acids} if len(amino_acids) == full_ha_length and set(amino_acids) <= AA_ALPHABET else set()
    if sequence_type == "nt":
        return {
            amino_acids
            for frame in range(3)
            for amino_acids in [translate_nt(sequence, frame)]
            if len(amino_acids) == full_ha_length and set(amino_acids) <= AA_ALPHABET
        }
    raise ValueError(f"Unsupported sample sequence type: {sequence_type}")


def build_sample_sequence_map(
    sample_fasta: Path,
    *,
    sequence_type: str,
    segment_tag: str,
    full_ha_length: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    records = read_fasta(sample_fasta)
    by_header: dict[str, set[str]] = {}
    epi_to_sequences: dict[str, set[str]] = {}
    counts = {
        "sample_records": len(records),
        "sample_segment_records": 0,
        "sample_full_length_records": 0,
        "sample_records_with_epi": 0,
    }
    for header, sequence in records:
        if segment_tag and segment_tag not in header:
            continue
        counts["sample_segment_records"] += 1
        candidates = sample_sequence_candidates(sequence, sequence_type=sequence_type, full_ha_length=full_ha_length)
        if not candidates:
            continue
        counts["sample_full_length_records"] += 1
        by_header.setdefault(header, set()).update(candidates)
        by_header.setdefault(normalise_header(header), set()).update(candidates)
        epi_isl = extract_epi_isl(header)
        if epi_isl:
            counts["sample_records_with_epi"] += 1
            epi_to_sequences.setdefault(epi_isl, set()).update(candidates)

    counts["sample_unique_epi"] = sum(len(sequences) == 1 for sequences in epi_to_sequences.values())
    counts["sample_conflicting_epi"] = sum(len(sequences) > 1 for sequences in epi_to_sequences.values())
    return by_header, epi_to_sequences, counts


def expand_instances_from_samples(
    merged: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    nextclade_tsv: Path,
    sample_fasta: Path,
    sequence_type: str,
    segment_tag: str,
    full_ha_length: int,
    label_mode: str,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    if full_ha_length <= 0:
        raise ValueError("--full-ha-length must be positive.")
    by_header, sequence_by_epi, counts = build_sample_sequence_map(
        sample_fasta,
        sequence_type=sequence_type,
        segment_tag=segment_tag,
        full_ha_length=full_ha_length,
    )
    score_columns = [
        "fasta_sha256", "sequence_index", "sequence_id", "sequence_sha256",
        "valid_length", "raw_log_prob_sum", "bert_score", "scored_at_utc",
        "ha1_sha256", "query_id",
    ]
    scored = merged.copy()
    scored["full_ha_sequence"] = scored["sequence"].astype(str).str.strip().str.upper()
    scored = scored.sort_values(list(IDENTITY_COLUMNS), kind="stable").drop_duplicates("full_ha_sequence", keep="first")
    scored_by_sequence = scored.set_index("full_ha_sequence")[score_columns]
    scored_sequences = set(scored_by_sequence.index)

    nextclade = (
        read_nextclade_table(nextclade_tsv, label_mode=label_mode)
        .reset_index()
        .rename(columns={"index": "nextclade_row"})
    )
    sample_fasta_sha256 = sha256_file(sample_fasta)
    rows = []
    missing_epi = missing_sample_sequence = missing_score = conflicting_score = 0
    for sample_index, row in nextclade.reset_index(drop=True).iterrows():
        full_header = str(row["seqName"])
        epi_isl = extract_epi_isl(full_header)
        if not epi_isl:
            missing_epi += 1
            continue
        candidates = set()
        candidates.update(by_header.get(full_header, set()))
        candidates.update(by_header.get(normalise_header(full_header), set()))
        candidates.update(sequence_by_epi.get(epi_isl, set()))
        if not candidates:
            missing_sample_sequence += 1
            continue
        hits = sorted(candidates & scored_sequences)
        if not hits:
            missing_score += 1
            continue
        if len(hits) > 1:
            conflicting_score += 1
            continue
        sequence = hits[0]
        score = scored_by_sequence.loc[sequence]
        parsed = parse_header(full_header)
        collection_date = row["collection_date"]
        if pd.notna(collection_date):
            collection_date = pd.Timestamp(collection_date).date().isoformat()
        else:
            collection_date = parsed["collection_date"]
        output_row = {
            "fasta_sha256": sample_fasta_sha256,
            "sequence_index": int(sample_index),
            "sequence_id": normalise_header(full_header).split()[0],
            "sequence_sha256": sha256_text(sequence),
            "nextclade_row": int(row["nextclade_row"]),
            "prediction_branch": row["prediction_branch"],
            "score_fasta_sha256": score["fasta_sha256"],
            "score_sequence_index": int(score["sequence_index"]),
            "score_sequence_id": score["sequence_id"],
            "score_sequence_sha256": score["sequence_sha256"],
            "valid_length": score["valid_length"],
            "raw_log_prob_sum": score["raw_log_prob_sum"],
            "bert_score": score["bert_score"],
            "scored_at_utc": score["scored_at_utc"],
            "full_header": full_header,
            "sample_name": parsed["sample_name"],
            "subtype": parsed["subtype"],
            "clade_label": parsed["clade_label"],
            "lineage": parsed["lineage"],
            "clade": parsed["clade"],
            "collection_date": collection_date,
            "year": parsed["year"],
            "time_source": parsed["time_source"],
            "ha1_sha256": score["ha1_sha256"],
            "query_id": score["query_id"],
        }
        rows.append(output_row)

    if not rows:
        raise ValueError(
            "No Nextclade TSV rows could be mapped through the sample AA FASTA to scored unique HA sequences."
        )
    instances = pd.DataFrame(rows)
    instances = instances.merge(
        registry[["query_id", "embedding_id", "embedding_source"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    diagnostics: dict[str, int | str] = {
        **counts,
        "nextclade_rows": int(len(nextclade)),
        "expanded_query_instances": int(len(instances)),
        "nextclade_rows_without_epi_isl": int(missing_epi),
        "nextclade_rows_without_sample_sequence": int(missing_sample_sequence),
        "nextclade_rows_without_scored_sequence": int(missing_score),
        "nextclade_rows_with_conflicting_scored_sequence": int(conflicting_score),
        "nextclade_rows_with_prediction_branch": int(nextclade["prediction_branch"].notna().sum()),
        "sample_sequence_type": sequence_type,
        "sample_fasta": str(sample_fasta),
        "nextclade_tsv": str(nextclade_tsv),
        "label_mode": label_mode,
    }
    return instances, diagnostics


def reusable_embedding_ids(registry_path: Path, embedding_dir: Path) -> dict[str, str]:
    registry = pd.read_csv(registry_path, usecols=["seq_id", "segment", "sequence"])
    registry = registry.loc[registry["segment"].astype(str).eq("HA")].copy()
    registry["sequence"] = registry["sequence"].astype(str).str.replace("-", "", regex=False).str.upper()
    registry = registry.loc[registry["sequence"].str.len().eq(327)]
    registry["exists"] = registry["seq_id"].map(
        lambda seq_id: (embedding_dir / f"matrix_{seq_id}.pt").is_file()
    )
    registry = registry.loc[registry["exists"]].sort_values("seq_id", kind="stable")
    conflicts = registry.duplicated("sequence", keep=False)
    if conflicts.any() and registry.loc[conflicts].groupby("sequence")["seq_id"].nunique().gt(1).any():
        raise ValueError("Multiple existing embedding IDs map to the same 327-aa HA1 sequence.")
    return dict(zip(registry["sequence"], registry["seq_id"], strict=True))


def main() -> None:
    args = parse_args()
    scores_path = require_file(args.scores_sqlite, "Score SQLite")
    metadata_path = require_file(args.metadata_csv, "Metadata CSV")
    registry_path = require_file(args.fluprofiler_registry, "fluProfiler embedding registry")
    embedding_dir = require_directory(args.fluprofiler_embedding_dir, "fluProfiler embedding directory")
    if args.no_sample_expansion and (args.sample_aa_fasta or args.sample_nt_fasta):
        raise ValueError("Do not pass --sample-aa-fasta/--sample-nt-fasta together with --no-sample-expansion.")
    if args.sample_aa_fasta and args.sample_nt_fasta:
        raise ValueError("Use only one of --sample-aa-fasta or --sample-nt-fasta.")
    sample_fasta = None
    sample_sequence_type = None
    sample_aa_fasta = args.sample_aa_fasta
    if not args.no_sample_expansion and sample_aa_fasta is None and args.sample_nt_fasta is None:
        sample_aa_fasta = DEFAULT_SAMPLE_AA_FASTA if DEFAULT_SAMPLE_AA_FASTA.is_file() else None
    if sample_aa_fasta:
        sample_fasta = require_file(sample_aa_fasta, "Non-deduplicated sample AA FASTA")
        sample_sequence_type = "aa"
    elif args.sample_nt_fasta:
        sample_fasta = require_file(args.sample_nt_fasta, "Non-deduplicated sample NT FASTA")
        sample_sequence_type = "nt"
    nextclade_tsv = require_file(args.nextclade_tsv, "Nextclade TSV") if sample_fasta else None
    if args.ha1_start < 0 or args.ha1_length <= 0:
        raise ValueError("--ha1-start must be non-negative and --ha1-length must be positive.")

    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "instances": output_dir / "query_instances.csv",
        "registry": output_dir / "query_ha1_registry.csv",
        "fasta": output_dir / "missing_query_ha1.fasta",
        "manifest": output_dir / "asset_manifest.json",
    }
    output_dir = prepare_output_directory(output_dir, list(outputs.values()), overwrite=args.overwrite)

    scores = load_scores(scores_path)
    metadata = load_metadata(metadata_path)
    merged = scores.merge(metadata, on=IDENTITY_COLUMNS, how="left", validate="one_to_one")
    if merged["full_header"].isna().any():
        raise ValueError(f"{int(merged['full_header'].isna().sum())} scored sequence(s) lack metadata identity matches.")

    merged["sequence"] = merged["sequence"].astype(str).str.strip().str.upper()
    lengths = merged["sequence"].str.len()
    if not lengths.eq(566).all():
        observed = sorted(lengths.value_counts().to_dict().items())
        raise ValueError(f"Expected only 566-aa full HAs before HA1 extraction; observed {observed}.")
    merged["ha1_sequence"] = merged["sequence"].str.slice(args.ha1_start, args.ha1_start + args.ha1_length)
    invalid = ~merged["ha1_sequence"].map(lambda value: len(value) == args.ha1_length and set(value) <= AA_ALPHABET)
    if invalid.any():
        raise ValueError(f"HA1 extraction produced {int(invalid.sum())} invalid sequence(s).")
    ambiguous_character_counts = {
        character: int(sum(sequence.count(character) for sequence in merged["ha1_sequence"]))
        for character in sorted(AA_ALPHABET - CANONICAL_AA_ALPHABET)
    }
    ambiguous_character_counts = {
        character: count for character, count in ambiguous_character_counts.items() if count
    }

    merged["ha1_sha256"] = merged["ha1_sequence"].map(sha256_text)
    merged["query_id"] = "H1HA1_" + merged["ha1_sha256"].str.slice(0, 20)
    if merged.groupby("query_id")["ha1_sha256"].nunique().gt(1).any():
        raise RuntimeError("Truncated HA1 hash collision detected.")

    reusable = reusable_embedding_ids(registry_path, embedding_dir)
    registry = (
        merged[["query_id", "ha1_sha256", "ha1_sequence"]]
        .drop_duplicates("query_id")
        .sort_values("query_id", kind="stable")
        .reset_index(drop=True)
    )
    registry["embedding_id"] = registry["ha1_sequence"].map(reusable).fillna(registry["query_id"])
    registry["embedding_source"] = registry["ha1_sequence"].map(
        lambda sequence: "fluprofiler_existing" if sequence in reusable else "new_required"
    )
    registry["ha1_length"] = registry["ha1_sequence"].str.len()

    expansion_diagnostics: dict[str, int | str] = {"mode": "scored_unique_sequences"}
    if sample_fasta is not None and sample_sequence_type is not None and nextclade_tsv is not None:
        instances, expansion_diagnostics = expand_instances_from_samples(
            merged,
            registry,
            nextclade_tsv=nextclade_tsv,
            sample_fasta=sample_fasta,
            sequence_type=sample_sequence_type,
            segment_tag=args.segment_tag,
            full_ha_length=args.full_ha_length,
            label_mode=args.label_mode,
        )
        expansion_diagnostics["mode"] = f"nextclade_samples_via_full_{sample_sequence_type}"
    else:
        instances = merged.drop(columns=["sequence", "ha1_sequence"]).copy()
        instances = instances.merge(
            registry[["query_id", "embedding_id", "embedding_source"]], on="query_id", how="left", validate="many_to_one"
        )
    instances["collection_date"] = pd.to_datetime(instances["collection_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    instances = instances.sort_values(list(IDENTITY_COLUMNS), kind="stable")

    missing = registry.loc[registry["embedding_source"].eq("new_required")]
    instances.to_csv(outputs["instances"], index=False)
    registry.to_csv(outputs["registry"], index=False)
    with outputs["fasta"].open("w", encoding="ascii", newline="\n") as handle:
        for row in missing.itertuples(index=False):
            handle.write(f">{row.embedding_id}\n{row.ha1_sequence}\n")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "H1N1 full-HA to HA1 assets for external fluProfiler antigenicity inference",
        "input": {
            "scores_sqlite": str(scores_path),
            "metadata_csv": str(metadata_path),
            "fluprofiler_registry": str(registry_path),
            "fluprofiler_embedding_dir": str(embedding_dir),
            "sample_fasta": str(sample_fasta) if sample_fasta else None,
            "sample_sequence_type": sample_sequence_type,
            "nextclade_tsv": str(nextclade_tsv) if nextclade_tsv else None,
        },
        "ha1_extraction": {
            "full_ha_length_required": args.full_ha_length,
            "slice_python": f"sequence[{args.ha1_start}:{args.ha1_start + args.ha1_length}]",
            "ha1_length": args.ha1_length,
            "accepted_ambiguous_residues": sorted(AA_ALPHABET - CANONICAL_AA_ALPHABET),
            "observed_ambiguous_residue_counts": ambiguous_character_counts,
            "validated_against_local_fluProfiler_h1_pairs": True,
        },
        "counts": {
            "scored_unique_full_ha": int(len(merged)),
            "query_instances": int(len(instances)),
            "unique_ha1": int(len(registry)),
            "reused_existing_embeddings": int(registry["embedding_source"].eq("fluprofiler_existing").sum()),
            "new_embeddings_required": int(registry["embedding_source"].eq("new_required").sum()),
            "instances_with_collection_date": int(instances["collection_date"].notna().sum()),
            "unique_query_ids_in_instances": int(instances["query_id"].nunique()),
        },
        "sample_expansion": expansion_diagnostics,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    write_json(outputs["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
