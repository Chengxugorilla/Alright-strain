"""Recover full FASTA headers and summarize score, time, and clade locally."""

import argparse
import csv
import hashlib
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UNKNOWN_VALUES = {"", "unknown", "unassigned", "na", "n/a", "none", "null", "-"}


def read_fasta_with_full_headers(path: Path):
    records = []
    header, sequence_lines = None, []

    def flush():
        if header is None:
            return
        sequence = "".join(sequence_lines).upper()
        if not sequence:
            raise ValueError(f"空 FASTA 记录：{header}")
        records.append((header, sequence))

    with path.open(encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header, sequence_lines = line[1:].strip(), []
            elif header is None:
                raise ValueError("FASTA 首条非空行不是 header。")
            else:
                sequence_lines.append("".join(line.split()))
    flush()
    return records


def sha256_text(value: str):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def normalise_clade(value: str):
    value = value.strip()
    return None if value.lower() in UNKNOWN_VALUES else value


def parse_header(full_header: str):
    """Parse the pipe-delimited GISAID-style fields without discarding raw text."""
    fields = [field.strip() for field in full_header.split("|")]
    sample_name = fields[0] if fields else full_header
    subtype = fields[2] if len(fields) > 2 else None
    clade_label = normalise_clade(fields[3]) if len(fields) > 3 else None
    lineage = normalise_clade(fields[4]) if len(fields) > 4 else None
    clade = normalise_clade(fields[5]) if len(fields) > 5 else None
    collection_date = fields[6] if len(fields) > 6 else None
    year, time_source = None, None
    if collection_date:
        try:
            year = datetime.fromisoformat(collection_date).year
            time_source = "collection_date"
        except ValueError:
            collection_date = None
    if year is None:
        isolate_years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", sample_name)
        if isolate_years:
            year = int(isolate_years[-1])
            time_source = "isolate_name_year"
    return {
        "full_header": full_header,
        "sample_name": sample_name,
        "subtype": subtype,
        "clade_label": clade_label,
        "lineage": lineage,
        "clade": clade,
        "collection_date": collection_date,
        "year": year,
        "time_source": time_source,
    }


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_row(group, values):
    return {
        **group,
        "sequence_count": values["count"],
        "mean_bert_score": values["score_sum"] / values["count"],
        "min_bert_score": values["score_min"],
        "max_bert_score": values["score_max"],
    }


def add_score(summary, key, score):
    values = summary[key]
    values["count"] += 1
    values["score_sum"] += score
    values["score_min"] = score if values["score_min"] is None else min(values["score_min"], score)
    values["score_max"] = score if values["score_max"] is None else max(values["score_max"], score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=PROJECT_ROOT / "src/ha_unaglined.fasta")
    parser.add_argument(
        "--score-table",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_raw_scores.sqlite",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_metadata.csv",
    )
    parser.add_argument(
        "--year-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_year_summary.csv",
    )
    parser.add_argument(
        "--year-clade-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_year_clade_summary.csv",
    )
    parser.add_argument(
        "--clade-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_clade_summary.csv",
    )
    args = parser.parse_args()

    fasta_records = read_fasta_with_full_headers(args.fasta)
    connection = sqlite3.connect(f"file:{args.score_table}?mode=ro", uri=True)
    score_rows = connection.execute(
        """
        SELECT fasta_sha256, sequence_index, sequence_id, sequence_sha256, sequence,
               valid_length, raw_log_prob_sum, bert_score, scored_at_utc
        FROM sequence_scores
        WHERE raw_log_prob_sum IS NOT NULL
        ORDER BY sequence_index
        """
    ).fetchall()
    connection.close()
    if len(fasta_records) != len(score_rows):
        raise RuntimeError(f"FASTA 有 {len(fasta_records)} 条，但已完成得分有 {len(score_rows)} 条。")

    metadata_rows = []
    year_summary = defaultdict(lambda: {"count": 0, "score_sum": 0.0, "score_min": None, "score_max": None})
    year_clade_summary = defaultdict(lambda: {"count": 0, "score_sum": 0.0, "score_min": None, "score_max": None})
    clade_summary = defaultdict(lambda: {"count": 0, "score_sum": 0.0, "score_min": None, "score_max": None, "years": set()})

    for index, ((full_header, fasta_sequence), score_row) in enumerate(zip(fasta_records, score_rows)):
        (
            fasta_sha256, sequence_index, sequence_id, sequence_hash, sequence,
            valid_length, raw_score, bert_score, scored_at_utc,
        ) = score_row
        if sequence_index != index or sequence != fasta_sequence or sequence_hash != sha256_text(fasta_sequence):
            raise RuntimeError(f"FASTA 与得分表在原始索引 {index} 处不匹配。")
        parsed = parse_header(full_header)
        row = {
            "fasta_sha256": fasta_sha256,
            "sequence_index": sequence_index,
            "sequence_id": sequence_id,
            "sequence_sha256": sequence_hash,
            "full_header": full_header,
            "sample_name": parsed["sample_name"],
            "subtype": parsed["subtype"],
            "clade_label": parsed["clade_label"],
            "lineage": parsed["lineage"],
            "clade": parsed["clade"],
            "collection_date": parsed["collection_date"],
            "year": parsed["year"],
            "time_source": parsed["time_source"],
            "valid_length": valid_length,
            "raw_log_prob_sum": raw_score,
            "bert_score": bert_score,
            "scored_at_utc": scored_at_utc,
        }
        metadata_rows.append(row)
        if parsed["year"] is not None:
            add_score(year_summary, parsed["year"], bert_score)
            clade = parsed["clade"] or "unassigned"
            add_score(year_clade_summary, (parsed["year"], clade), bert_score)
            add_score(clade_summary, clade, bert_score)
            clade_summary[clade]["years"].add(parsed["year"])

    write_csv(args.metadata_output, list(metadata_rows[0]), metadata_rows)
    write_csv(
        args.year_output,
        ["year", "sequence_count", "mean_bert_score", "min_bert_score", "max_bert_score"],
        [summary_row({"year": year}, year_summary[year]) for year in sorted(year_summary)],
    )
    write_csv(
        args.year_clade_output,
        ["year", "clade", "sequence_count", "mean_bert_score", "min_bert_score", "max_bert_score"],
        [
            summary_row({"year": year, "clade": clade}, year_clade_summary[(year, clade)])
            for year, clade in sorted(year_clade_summary)
        ],
    )
    write_csv(
        args.clade_output,
        ["clade", "sequence_count", "mean_bert_score", "min_bert_score", "max_bert_score", "first_year", "last_year"],
        [
            {
                **summary_row({"clade": clade}, clade_summary[clade]),
                "first_year": min(clade_summary[clade]["years"]),
                "last_year": max(clade_summary[clade]["years"]),
            }
            for clade in sorted(clade_summary)
        ],
    )
    print(f"元数据：{args.metadata_output}")
    print(f"年度汇总：{args.year_output}")
    print(f"年度-clade 汇总：{args.year_clade_output}")
    print(f"clade 汇总：{args.clade_output}")


if __name__ == "__main__":
    main()
