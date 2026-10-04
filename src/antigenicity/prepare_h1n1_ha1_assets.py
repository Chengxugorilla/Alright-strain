#!/usr/bin/env python3
"""Prepare deduplicated H1N1 HA1 assets for external fluProfiler inference.

This script deliberately performs only local file processing.  It never loads
the antigenic model or LucaVirus model, and it never edits either project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLUPROFILER_ROOT = Path("/home/chenyh/workspace/fluProfiler")
# LucaVirus explicitly tokenizes these standard and ambiguous protein symbols.
# fluProfiler maps non-canonical residues to its unknown amino-acid ID, so
# retaining them is safer and more reproducible than silently discarding data.
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUOJ")
CANONICAL_AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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
    parser.add_argument("--ha1-start", type=int, default=17, help="0-based full-HA start index.")
    parser.add_argument("--ha1-length", type=int, default=327)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


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
    identity = ["fasta_sha256", "sequence_index", "sequence_sha256"]
    if frame.duplicated(identity).any():
        raise ValueError("SQLite score table contains duplicate sequence identities.")
    return frame


def load_metadata(path: Path) -> pd.DataFrame:
    required = [
        "fasta_sha256", "sequence_index", "sequence_sha256", "full_header",
        "collection_date", "clade", "lineage", "clade_label",
    ]
    frame = pd.read_csv(path, usecols=required)
    identity = ["fasta_sha256", "sequence_index", "sequence_sha256"]
    if frame.duplicated(identity).any():
        raise ValueError("Metadata CSV contains duplicate sequence identities.")
    return frame


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


def assert_writable(paths: list[Path], overwrite: bool) -> None:
    present = [path for path in paths if path.exists()]
    if present and not overwrite:
        joined = ", ".join(str(path) for path in present)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {joined}. Use --overwrite to replace them.")


def main() -> None:
    args = parse_args()
    scores_path = require_file(args.scores_sqlite, "Score SQLite")
    metadata_path = require_file(args.metadata_csv, "Metadata CSV")
    registry_path = require_file(args.fluprofiler_registry, "fluProfiler embedding registry")
    embedding_dir = args.fluprofiler_embedding_dir.expanduser().resolve()
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"fluProfiler embedding directory does not exist: {embedding_dir}")
    if args.ha1_start < 0 or args.ha1_length <= 0:
        raise ValueError("--ha1-start must be non-negative and --ha1-length must be positive.")

    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "instances": output_dir / "query_instances.csv",
        "registry": output_dir / "query_ha1_registry.csv",
        "fasta": output_dir / "missing_query_ha1.fasta",
        "manifest": output_dir / "asset_manifest.json",
    }
    assert_writable(list(outputs.values()), args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(scores_path)
    metadata = load_metadata(metadata_path)
    identity = ["fasta_sha256", "sequence_index", "sequence_sha256"]
    merged = scores.merge(metadata, on=identity, how="left", validate="one_to_one")
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

    instances = merged.drop(columns=["sequence", "ha1_sequence"]).copy()
    instances = instances.merge(
        registry[["query_id", "embedding_id", "embedding_source"]], on="query_id", how="left", validate="many_to_one"
    )
    instances["collection_date"] = pd.to_datetime(instances["collection_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    instances = instances.sort_values(identity, kind="stable")

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
        },
        "ha1_extraction": {
            "full_ha_length_required": 566,
            "slice_python": f"sequence[{args.ha1_start}:{args.ha1_start + args.ha1_length}]",
            "ha1_length": args.ha1_length,
            "accepted_ambiguous_residues": sorted(AA_ALPHABET - CANONICAL_AA_ALPHABET),
            "observed_ambiguous_residue_counts": ambiguous_character_counts,
            "validated_against_local_fluProfiler_h1_pairs": True,
        },
        "counts": {
            "scored_sequence_instances": int(len(instances)),
            "unique_ha1": int(len(registry)),
            "reused_existing_embeddings": int(registry["embedding_source"].eq("fluprofiler_existing").sum()),
            "new_embeddings_required": int(registry["embedding_source"].eq("new_required").sum()),
            "instances_with_collection_date": int(instances["collection_date"].notna().sum()),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
