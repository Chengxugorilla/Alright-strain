#!/usr/bin/env python3
"""Streaming audit of Nextclade TSV outputs.

The script intentionally emits only aggregate statistics and a few column names;
it never prints sequence strings, mutation lists, or complete input rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ASSIGNMENT_COLUMNS = (
    "clade",
    "proposedSubclade",
    "subclade",
    "short-clade",
    "legacy-clade",
    "legacy-clade-vic",
    "legacy-clade-yam",
    "lineage",
)
METRICS = (
    "qc.overallScore",
    "coverage",
    "totalMissing",
    "totalNonACGTNs",
    "totalFrameShifts",
    "totalUnknownAa",
    "alignmentStart",
    "alignmentEnd",
)


def safe_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def status(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return "blank"
    if value.lower() == "unassigned":
        return "unassigned"
    return "assigned"


def metadata(seq_name: str) -> tuple[str, str, str]:
    parts = seq_name.split("|")
    subtype = parts[2].strip() if len(parts) > 2 else ""
    lineage = parts[5].strip() if len(parts) > 5 else ""
    date_match = re.search(r"(?:^|\|)((?:19|20)\d{2})-\d{2}-\d{2}(?:\||$)", seq_name)
    year = date_match.group(1) if date_match else "unknown"
    return subtype or "unknown", lineage or "unknown", year


def error_category(message: str) -> str:
    text = message.lower()
    if "unknown nucleotide" in text:
        return "unknown_nucleotide"
    if "extracted gene sequence is empty" in text or "unable to find start" in text:
        return "gene_not_found_or_empty"
    if "too short" in text:
        return "sequence_or_gene_too_short"
    if "translation" in text:
        return "translation_error"
    if "alignment" in text:
        return "alignment_error"
    return "other"


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "mean": None}
    return {
        "n": len(values),
        "median": round(statistics.median(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def rate(n: int, d: int) -> float:
    return round(n / d, 6) if d else 0.0


def audit_file(path: Path) -> tuple[dict, dict[str, str]]:
    counts = Counter()
    unique_names: set[str] = set()
    primary_by_qc: dict[str, Counter] = defaultdict(Counter)
    primary_by_year: dict[str, Counter] = defaultdict(Counter)
    primary_by_lineage: dict[str, Counter] = defaultdict(Counter)
    primary_by_subtype: dict[str, Counter] = defaultdict(Counter)
    metrics: dict[str, dict[str, list[float]]] = {
        group: defaultdict(list) for group in ("assigned", "unassigned", "blank")
    }
    assignments: dict[str, str] = {}
    assignment_values: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    error_categories = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = reader.fieldnames or []
        error_columns = [
            h for h in headers
            if "error" in h.lower() or "warning" in h.lower() or h.lower() == "errors"
        ]
        present_assignment_columns = [c for c in ASSIGNMENT_COLUMNS if c in headers]
        for row in reader:
            counts["rows"] += 1
            if None in row:
                counts["rows_with_extra_columns"] += 1
            if any(row.get(h) is None for h in headers):
                counts["rows_with_missing_columns"] += 1
            seq_name = (row.get("seqName") or "").strip()
            if seq_name in unique_names:
                counts["duplicate_seqName_rows"] += 1
            else:
                unique_names.add(seq_name)

            primary = status(row.get("clade"))
            counts[f"primary_{primary}"] += 1
            if seq_name:
                assignments[seq_name] = primary

            subtype, lineage, year = metadata(seq_name)
            qc = (row.get("qc.overallStatus") or "blank").strip() or "blank"
            primary_by_qc[qc][primary] += 1
            primary_by_year[year][primary] += 1
            primary_by_lineage[lineage][primary] += 1
            primary_by_subtype[subtype][primary] += 1

            error_text = (row.get("errors") or "").strip()
            warning_text = (row.get("warnings") or "").strip()
            if error_text:
                counts["rows_with_errors"] += 1
                error_categories[error_category(error_text)] += 1
            if warning_text:
                counts["rows_with_warnings"] += 1
            if error_text or warning_text:
                counts["rows_with_error_or_warning"] += 1
                counts[f"error_primary_{primary}"] += 1

            for col in present_assignment_columns:
                field_status = status(row.get(col))
                counts[f"field::{col}::{field_status}"] += 1
                value = (row.get(col) or "").strip() or "<blank>"
                assignment_values[col][primary][value] += 1
            for col in METRICS:
                value = safe_float(row.get(col))
                if value is not None:
                    metrics[primary][col].append(value)

    rows = counts["rows"]
    result = {
        "file": str(path),
        "rows": rows,
        "unique_seqName": len(unique_names),
        "duplicate_seqName_rows": counts["duplicate_seqName_rows"],
        "tsv_structure": {
            "rows_with_extra_columns": counts["rows_with_extra_columns"],
            "rows_with_missing_columns": counts["rows_with_missing_columns"],
        },
        "primary_clade": {
            group: {
                "n": counts[f"primary_{group}"],
                "rate": rate(counts[f"primary_{group}"], rows),
            }
            for group in ("assigned", "unassigned", "blank")
        },
        "rows_with_error_or_warning": {
            "n": counts["rows_with_error_or_warning"],
            "rate": rate(counts["rows_with_error_or_warning"], rows),
            "by_primary_status": {
                group: counts[f"error_primary_{group}"]
                for group in ("assigned", "unassigned", "blank")
            },
        },
        "rows_with_errors": {"n": counts["rows_with_errors"], "rate": rate(counts["rows_with_errors"], rows)},
        "rows_with_warnings": {"n": counts["rows_with_warnings"], "rate": rate(counts["rows_with_warnings"], rows)},
        "error_categories": dict(error_categories.most_common()),
        "assignment_fields": {
            col: {
                group: counts[f"field::{col}::{group}"]
                for group in ("assigned", "unassigned", "blank")
            }
            for col in present_assignment_columns
        },
        "assignment_value_top": {
            col: {
                group: assignment_values[col][group].most_common(20)
                for group in ("assigned", "unassigned", "blank")
            }
            for col in present_assignment_columns
        },
        "qc_by_primary_status": {k: dict(v) for k, v in sorted(primary_by_qc.items())},
        "metrics_by_primary_status": {
            group: {col: metric_summary(values) for col, values in data.items()}
            for group, data in metrics.items()
        },
        "year_by_primary_status": {k: dict(v) for k, v in sorted(primary_by_year.items())},
        "metadata_lineage_top": [],
        "metadata_subtype": {k: dict(v) for k, v in sorted(primary_by_subtype.items())},
        "columns": {
            "count": len(headers),
            "assignment": present_assignment_columns,
            "error_or_warning": error_columns,
        },
    }

    lineage_rows = []
    for name, c in primary_by_lineage.items():
        total = sum(c.values())
        lineage_rows.append({
            "value": name,
            "n": total,
            "assigned": c["assigned"],
            "unassigned": c["unassigned"],
            "blank": c["blank"],
            "not_assigned_rate": rate(c["unassigned"] + c["blank"], total),
        })
    result["metadata_lineage_top"] = sorted(
        lineage_rows, key=lambda x: (-x["n"], x["value"])
    )[:20]
    return result, assignments


def compare_runs(named_assignments: dict[str, dict[str, str]]) -> list[dict]:
    output = []
    names = sorted(named_assignments)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if Path(left).parent != Path(right).parent:
                continue
            a = named_assignments[left]
            b = named_assignments[right]
            overlap = a.keys() & b.keys()
            transitions = Counter(f"{a[k]}->{b[k]}" for k in overlap)
            output.append({
                "left": left,
                "right": right,
                "overlap": len(overlap),
                "left_only": len(a.keys() - b.keys()),
                "right_only": len(b.keys() - a.keys()),
                "transitions": dict(sorted(transitions.items())),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data/raw/nextclade")
    parser.add_argument("-o", "--output", default="outputs/nextclade_audit/nextclade_audit_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    paths = sorted(root.rglob("*.tsv"))
    reports = []
    assignments = {}
    for path in paths:
        report, file_assignments = audit_file(path)
        reports.append(report)
        assignments[str(path)] = file_assignments

    payload = {
        "files": reports,
        "same_subtype_run_comparisons": compare_runs(assignments),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "files": len(reports),
        "rows": sum(r["rows"] for r in reports),
        "output": args.output,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
