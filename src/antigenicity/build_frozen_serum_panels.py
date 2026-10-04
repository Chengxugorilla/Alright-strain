#!/usr/bin/env python3
"""Build cutoff-frozen H1N1 serum panels and rolling-window metadata.

No model inference occurs here.  The resulting CSVs are a small, auditable
contract for the later multi-GPU scorer: each forecast cutoff gets exactly one
serum panel containing only sera dated before that cutoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLUPROFILER_ROOT = Path("/home/chenyh/workspace/fluProfiler")
PANDEMIC_EXCLUDED_TARGET_STARTS = {
    pd.Timestamp("2020-04-01"),
    pd.Timestamp("2020-10-01"),
    pd.Timestamp("2021-04-01"),
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.h1n1_benchmarks import DEFAULT_HALF_YEAR_WINDOWS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-instances-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv",
    )
    parser.add_argument(
        "--serum-source-csv",
        type=Path,
        default=FLUPROFILER_ROOT / "data/dataset/H1_HA1_v1.0/processed/source.csv",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=FLUPROFILER_ROOT / "data/embedding/files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels",
    )
    parser.add_argument(
        "--cutoff",
        action="append",
        default=[],
        help="Optional forecast cutoff YYYY-MM-DD. Repeat for multiple cutoffs.",
    )
    parser.add_argument("--include-pandemic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def resolve_cutoffs(values: list[str], include_pandemic: bool) -> list[pd.Timestamp]:
    if values:
        cutoffs = [pd.Timestamp(value).normalize() for value in values]
    else:
        target_starts = [pd.Timestamp(start).normalize() for start, _ in DEFAULT_HALF_YEAR_WINDOWS]
        if not include_pandemic:
            target_starts = [start for start in target_starts if start not in PANDEMIC_EXCLUDED_TARGET_STARTS]
        cutoffs = [target_start - pd.Timedelta(days=1) for target_start in target_starts]
    if any(value is pd.NaT for value in cutoffs):
        raise ValueError("Invalid --cutoff date.")
    return sorted(set(cutoffs))


def source_serum_conditions(path: Path, embedding_dir: Path) -> pd.DataFrame:
    fields = ["seq_id_a", "serumHA", "serumPassCat", "serumName", "serumDate", "Type"]
    source = pd.read_csv(path, usecols=fields, keep_default_na=False)
    source = source.loc[source["Type"].astype(str).eq("H1N1")].copy()
    source["serum_date"] = pd.to_datetime(source["serumDate"], errors="coerce").dt.normalize()
    source = source.loc[source["serum_date"].notna()].copy()
    source["seq_id_a"] = source["seq_id_a"].astype(str)
    source["serumPassCat"] = source["serumPassCat"].astype(str)
    source["serumName"] = source["serumName"].astype(str)
    source = source.sort_values("serum_date", kind="stable").drop_duplicates(
        ["seq_id_a", "serumPassCat", "serumName"], keep="first"
    )
    source["embedding_file"] = source["seq_id_a"].map(
        lambda value: str(embedding_dir / f"matrix_{value}.pt")
    )
    source["embedding_exists"] = source["embedding_file"].map(lambda value: Path(value).is_file())
    missing = source.loc[~source["embedding_exists"], "seq_id_a"].tolist()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frozen-panel serum embedding(s) are missing; first IDs: {missing[:10]}"
        )
    return source.rename(columns={"serumHA": "serum_ha1", "serumDate": "serum_date_raw"})[
        ["seq_id_a", "serum_ha1", "serumPassCat", "serumName", "serum_date", "serum_date_raw", "embedding_file"]
    ].reset_index(drop=True)


def build_cutoff_table(cutoffs: list[pd.Timestamp], query_instances: pd.DataFrame) -> pd.DataFrame:
    query_dates = pd.to_datetime(query_instances["collection_date"], errors="coerce").dt.normalize()
    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        recent_start = cutoff - pd.DateOffset(months=6) + pd.Timedelta(days=1)
        annual_start = cutoff - pd.DateOffset(months=12) + pd.Timedelta(days=1)
        target_start = cutoff + pd.Timedelta(days=1)
        target_end = target_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        rows.append({
            "forecast_cutoff": cutoff.date().isoformat(),
            "recent_start": recent_start.date().isoformat(),
            "recent_end": cutoff.date().isoformat(),
            "annual_start": annual_start.date().isoformat(),
            "annual_end": cutoff.date().isoformat(),
            "previous_half_start": annual_start.date().isoformat(),
            "previous_half_end": (recent_start - pd.Timedelta(days=1)).date().isoformat(),
            "target_start": target_start.date().isoformat(),
            "target_end": target_end.date().isoformat(),
            "recent_query_instances": int(query_dates.between(recent_start, cutoff, inclusive="both").sum()),
            "annual_query_instances": int(query_dates.between(annual_start, cutoff, inclusive="both").sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    instances_path = require_file(args.query_instances_csv, "Query instance CSV")
    source_path = require_file(args.serum_source_csv, "Serum source CSV")
    embedding_dir = args.embedding_dir.expanduser().resolve()
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {embedding_dir}")
    output_dir = args.output_dir.expanduser().resolve()
    cutoffs_path = output_dir / "forecast_cutoffs.csv"
    panels_path = output_dir / "frozen_serum_panels.csv"
    manifest_path = output_dir / "panel_manifest.json"
    existing = [path for path in (cutoffs_path, panels_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = pd.read_csv(instances_path, usecols=["collection_date"])
    conditions = source_serum_conditions(source_path, embedding_dir)
    cutoffs = resolve_cutoffs(args.cutoff, args.include_pandemic)
    cutoff_table = build_cutoff_table(cutoffs, instances)
    panel_rows: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        panel = conditions.loc[conditions["serum_date"] < cutoff].copy()
        if panel.empty:
            raise ValueError(f"No dated H1N1 serum is available before cutoff {cutoff.date()}.")
        panel.insert(0, "forecast_cutoff", cutoff.date().isoformat())
        panel_rows.append(panel)
    panels = pd.concat(panel_rows, ignore_index=True)
    panel_counts = panels.groupby("forecast_cutoff").size().rename("frozen_serum_conditions")
    cutoff_table = cutoff_table.merge(panel_counts, left_on="forecast_cutoff", right_index=True, how="left", validate="one_to_one")
    cutoff_table.to_csv(cutoffs_path, index=False)
    panels.to_csv(panels_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "serum_panel_rule": "serumDate < forecast_cutoff",
        "cutoff_count": len(cutoffs),
        "cutoffs": [value.date().isoformat() for value in cutoffs],
        "serum_conditions_total": int(len(conditions)),
        "panel_size_by_cutoff": {
            str(key): int(value) for key, value in panel_counts.sort_index().items()
        },
        "query_instances_csv": str(instances_path),
        "serum_source_csv": str(source_path),
        "embedding_dir": str(embedding_dir),
        "outputs": {"cutoffs": str(cutoffs_path), "panels": str(panels_path)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
