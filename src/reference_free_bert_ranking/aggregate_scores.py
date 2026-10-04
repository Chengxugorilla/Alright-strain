"""Create ranked CSV and compact metadata from the durable SQLite score table."""

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def connect_read_only(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到 SQLite 原始分数表：{path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def validate_complete(connection):
    total, completed, fasta_count, invalid_scores = connection.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(raw_log_prob_sum IS NOT NULL), 0),
            COUNT(DISTINCT fasta_sha256),
            COALESCE(SUM(
                raw_log_prob_sum IS NOT NULL AND (
                    valid_length IS NULL OR valid_length <= 0 OR bert_score IS NULL OR
                    ABS(bert_score - raw_log_prob_sum * 1.0 / valid_length) > 1e-12
                )
            ), 0)
        FROM sequence_scores
        """
    ).fetchone()
    if total == 0:
        raise RuntimeError("原始分数表为空。")
    if completed != total:
        raise RuntimeError(f"仍有 {total - completed} 条序列未完成，拒绝生成最终排名。")
    if fasta_count != 1:
        raise RuntimeError("原始分数表包含多份 FASTA，需先按输入拆分后再汇总。")
    if invalid_scores:
        raise RuntimeError(f"检测到 {invalid_scores} 条分数与原始累计分数不一致。")


def write_ranking(connection, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "sequence_index",
        "sequence_id",
        "sequence_sha256",
        "sequence",
        "valid_length",
        "raw_log_prob_sum",
        "bert_score",
        "scored_at_utc",
    ]
    cursor = connection.execute(
        """
        SELECT sequence_index, sequence_id, sequence_sha256, sequence,
               valid_length, raw_log_prob_sum, bert_score, scored_at_utc
        FROM sequence_scores
        ORDER BY bert_score DESC, sequence_index ASC
        """
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for rank, row in enumerate(cursor, start=1):
            writer.writerow((rank, *row))


def write_summary(connection, score_table: Path, ranking_output: Path, output_path: Path):
    row = connection.execute(
        """
        SELECT
            MIN(fasta_sha256), COUNT(*),
            MIN(bert_score), MAX(bert_score), AVG(bert_score),
            MIN(raw_log_prob_sum), MAX(raw_log_prob_sum), AVG(raw_log_prob_sum),
            MIN(valid_length), MAX(valid_length), AVG(valid_length),
            MIN(scored_at_utc), MAX(scored_at_utc)
        FROM sequence_scores
        """
    ).fetchone()
    (
        fasta_sha256, count,
        bert_min, bert_max, bert_mean,
        raw_min, raw_max, raw_mean,
        length_min, length_max, length_mean,
        scored_first, scored_last,
    ) = row
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_score_table": str(score_table.resolve()),
        "ranking_csv": str(ranking_output.resolve()),
        "fasta_sha256": fasta_sha256,
        "sequence_count": count,
        "ranking_order": "bert_score descending; sequence_index ascending for ties",
        "bert_score": {"min": bert_min, "max": bert_max, "mean": bert_mean},
        "raw_log_prob_sum": {"min": raw_min, "max": raw_max, "mean": raw_mean},
        "valid_length": {"min": length_min, "max": length_max, "mean": length_mean},
        "scored_at_utc": {"first": scored_first, "last": scored_last},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-table",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_raw_scores.sqlite",
    )
    parser.add_argument(
        "--ranking-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_summary.json",
    )
    args = parser.parse_args()

    with connect_read_only(args.score_table) as connection:
        validate_complete(connection)
        write_ranking(connection, args.ranking_output)
        write_summary(connection, args.score_table, args.ranking_output, args.summary_output)
    print(f"排名 CSV：{args.ranking_output}")
    print(f"汇总 JSON：{args.summary_output}")


if __name__ == "__main__":
    main()
