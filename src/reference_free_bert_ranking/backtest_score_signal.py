"""Test whether clade BERT scores predict next half-year clade prevalence.

The evaluation uses only information available before each target window:
the prior six-month clade mean, median, and maximum BERT score plus clade
frequency. Nextclade counts in the target window define the outcome. It is a
descriptive rolling-origin backtest, not a causal fitness analysis.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.h1n1_benchmarks import DEFAULT_HALF_YEAR_WINDOWS
from src.h1n1_subclade_plot import read_nextclade_table


PANDEMIC_EXCLUDED_WINDOWS = {
    (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-09-30")),
    (pd.Timestamp("2020-10-01"), pd.Timestamp("2021-03-31")),
    (pd.Timestamp("2021-04-01"), pd.Timestamp("2021-09-30")),
}
EPI_PATTERN = r"\|(EPI_ISL_\d+)\|"
SCORE_FEATURES = ("mean_bert_score", "median_bert_score", "max_bert_score")


def extract_unique_epi_ids(frame: pd.DataFrame, header_column: str):
    ids = frame[header_column].astype("string").str.extract(EPI_PATTERN, expand=False)
    result = frame.copy()
    result["epi_isl"] = ids
    duplicate_ids = result["epi_isl"].duplicated(keep=False)
    return result.loc[result["epi_isl"].notna() & ~duplicate_ids].copy()


def window_data(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    return frame.loc[frame["collection_date"].between(start, end, inclusive="both")].copy()


def safe_correlation(data: pd.DataFrame, left: str, right: str, method: str):
    subset = data[[left, right]].dropna()
    return subset[left].corr(subset[right], method=method) if len(subset) >= 3 else np.nan


def top_branch(values: pd.Series, *, ascending: bool = False):
    values = values.dropna()
    if values.empty:
        return None
    ordered = values.sort_values(ascending=ascending, kind="stable")
    return ordered.index[0]


def calculate_window(
    records: pd.DataFrame,
    score_records: pd.DataFrame,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    min_score_sequences: int,
):
    previous_start = target_start - pd.DateOffset(months=6)
    previous_end = target_start - pd.Timedelta(days=1)
    previous_all = window_data(records, previous_start, previous_end)
    target_all = window_data(records, target_start, target_end)
    previous_scored = window_data(score_records, previous_start, previous_end)

    previous_counts = previous_all["prediction_branch"].value_counts()
    target_counts = target_all["prediction_branch"].value_counts()
    score_stats = previous_scored.groupby("prediction_branch")["bert_score"].agg(
        score_sequence_count="count",
        mean_bert_score="mean",
        median_bert_score="median",
        max_bert_score="max",
    )
    branches = previous_counts.index.union(target_counts.index).union(score_stats.index)
    detail = pd.DataFrame(index=branches)
    detail.index.name = "branch"
    detail["previous_count"] = previous_counts.reindex(branches, fill_value=0)
    detail["next_count"] = target_counts.reindex(branches, fill_value=0)
    detail["previous_share"] = detail["previous_count"] / max(len(previous_all), 1)
    detail["next_share"] = detail["next_count"] / max(len(target_all), 1)
    detail["share_change"] = detail["next_share"] - detail["previous_share"]
    detail["score_sequence_count"] = score_stats["score_sequence_count"].reindex(branches)
    for feature in SCORE_FEATURES:
        detail[feature] = score_stats[feature].reindex(branches)
        detail.loc[detail["score_sequence_count"].fillna(0) < min_score_sequences, feature] = np.nan

    actual_top1 = top_branch(detail["next_share"])
    persistence_top1 = top_branch(detail["previous_share"])
    score_top1 = {
        feature: (
            top_branch(detail[feature]),
            top_branch(detail[feature], ascending=True),
        )
        for feature in SCORE_FEATURES
    }
    detail = detail.reset_index()
    for column, value in {
        "target_start": target_start,
        "target_end": target_end,
        "previous_start": previous_start,
        "previous_end": previous_end,
    }.items():
        detail[column] = value

    metrics = {
        "target_start": target_start,
        "target_end": target_end,
        "previous_start": previous_start,
        "previous_end": previous_end,
        "previous_all_sequences": len(previous_all),
        "previous_scored_sequences": len(previous_scored),
        "next_all_sequences": len(target_all),
        "score_eligible_branches": int(detail["mean_bert_score"].notna().sum()),
        "actual_next_top1": actual_top1,
        "persistence_top1": persistence_top1,
        "persistence_top1_hit": persistence_top1 == actual_top1,
    }
    for feature in SCORE_FEATURES:
        feature_name = feature.removesuffix("_bert_score")
        high_top1, low_top1 = score_top1[feature]
        metrics.update({
            f"{feature_name}_vs_next_share_pearson": safe_correlation(detail, feature, "next_share", "pearson"),
            f"{feature_name}_vs_next_share_spearman": safe_correlation(detail, feature, "next_share", "spearman"),
            f"{feature_name}_vs_share_change_pearson": safe_correlation(detail, feature, "share_change", "pearson"),
            f"{feature_name}_vs_share_change_spearman": safe_correlation(detail, feature, "share_change", "spearman"),
            f"high_{feature_name}_top1": high_top1,
            f"low_{feature_name}_top1": low_top1,
            f"high_{feature_name}_top1_hit": high_top1 == actual_top1,
            f"low_{feature_name}_top1_hit": low_top1 == actual_top1,
        })
    return detail, metrics


def json_value(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nextclade-tsv",
        type=Path,
        default=PROJECT_ROOT / "data/clade注释_sorted/H1N1/h1n1_results_MW626062_sorted.tsv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_metadata.csv",
    )
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--min-score-sequences", type=int, default=10)
    parser.add_argument(
        "--cutoff",
        help="Optional feature-window end date (YYYY-MM-DD); evaluate the following six months only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_score_signal_backtest",
    )
    args = parser.parse_args()
    if args.min_score_sequences < 1:
        raise ValueError("--min-score-sequences 必须至少为 1。")

    records = read_nextclade_table(args.nextclade_tsv, label_mode=args.label_mode)
    metadata = pd.read_csv(args.metadata_csv, usecols=["full_header", "bert_score"])
    nextclade_unique = extract_unique_epi_ids(records, "seqName")
    score_unique = extract_unique_epi_ids(metadata, "full_header")
    score_records = nextclade_unique.merge(
        score_unique[["epi_isl", "bert_score"]], on="epi_isl", how="inner", validate="one_to_one"
    )

    if args.cutoff:
        previous_end = pd.Timestamp(args.cutoff).normalize()
        target_start = previous_end + pd.Timedelta(days=1)
        target_end = target_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        windows = [(target_start, target_end)]
    else:
        requested_windows = [
            (pd.Timestamp(start), pd.Timestamp(end)) for start, end in DEFAULT_HALF_YEAR_WINDOWS
        ]
        windows = sorted(window for window in requested_windows if window not in PANDEMIC_EXCLUDED_WINDOWS)
    details, metrics = [], []
    for target_start, target_end in windows:
        detail, metric = calculate_window(
            records, score_records, target_start, target_end, args.min_score_sequences
        )
        details.append(detail)
        metrics.append(metric)
    detail_table = pd.concat(details, ignore_index=True)
    metric_table = pd.DataFrame(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "score_next_window_branch_detail.csv"
    metrics_path = args.output_dir / "score_next_window_metrics.csv"
    summary_path = args.output_dir / "score_next_window_summary.json"
    detail_table.to_csv(detail_path, index=False)
    metric_table.to_csv(metrics_path, index=False)

    correlations = [
        f"{feature.removesuffix('_bert_score')}_vs_{target}_{method}"
        for feature in SCORE_FEATURES
        for target in ("next_share", "share_change")
        for method in ("pearson", "spearman")
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_mode": args.label_mode,
        "min_score_sequences_per_branch": args.min_score_sequences,
        "nextclade_rows": len(records),
        "bert_score_rows": len(metadata),
        "unique_nextclade_epi_ids": len(nextclade_unique),
        "unique_score_epi_ids": len(score_unique),
        "matched_scored_sequences": len(score_records),
        "score_row_match_fraction": len(score_records) / len(metadata),
        "window_count": len(metric_table),
        "cutoff": args.cutoff,
        "mean_window_correlations": {
            name: metric_table[name].mean(skipna=True) for name in correlations
        },
        "top1_hits": {
            "persistence": int(metric_table["persistence_top1_hit"].sum()),
            "windows": len(metric_table),
            **{
                f"high_{feature.removesuffix('_bert_score')}": int(metric_table[f"high_{feature.removesuffix('_bert_score')}_top1_hit"].sum())
                for feature in SCORE_FEATURES
            },
            **{
                f"low_{feature.removesuffix('_bert_score')}": int(metric_table[f"low_{feature.removesuffix('_bert_score')}_top1_hit"].sum())
                for feature in SCORE_FEATURES
            },
        },
        "detail_csv": str(detail_path.resolve()),
        "metrics_csv": str(metrics_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, default=json_value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"逐 clade 明细：{detail_path}")
    print(f"逐窗口指标：{metrics_path}")
    print(f"紧凑汇总：{summary_path}")


if __name__ == "__main__":
    main()
