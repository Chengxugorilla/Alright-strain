#!/usr/bin/env python3
"""Evaluate frozen-panel antigenicity scores in rolling H1N1 windows."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.antigenicity.common import IDENTITY_COLUMNS, PROJECT_ROOT, write_json
from src.strain_data.nextclade import read_nextclade_table  # noqa: E402

EPI_PATTERN = r"\|(EPI_ISL_\d+)\|"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores-sqlite", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/antigenicity_scores.sqlite")
    p.add_argument("--query-instances-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv")
    p.add_argument("--cutoffs-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels/forecast_cutoffs.csv")
    p.add_argument("--nextclade-tsv", type=Path, default=PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv")
    p.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    p.add_argument("--min-score-sequences", type=int, default=10)
    p.add_argument("--allow-partial", action="store_true", help="Write provisional results if some required query summaries are absent.")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/backtest")
    return p.parse_args()


def unique_epi(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    data = frame.copy()
    data["epi_isl"] = data[column].astype("string").str.extract(EPI_PATTERN, expand=False)
    return data.loc[data["epi_isl"].notna() & ~data["epi_isl"].duplicated(keep=False)].copy()


def window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[frame["collection_date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")].copy()


def top(values: pd.Series) -> str | None:
    values = values.dropna()
    return None if values.empty else str(values.sort_values(ascending=False, kind="stable").index[0])


def corr(data: pd.DataFrame, x: str, y: str, method: str) -> float:
    data = data[[x, y]].dropna()
    return float(data[x].corr(data[y], method=method)) if len(data) >= 3 else float("nan")


def json_value(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main():
    args = parse_args()
    if args.min_score_sequences < 1:
        raise ValueError("--min-score-sequences must be positive")
    with sqlite3.connect(args.scores_sqlite) as con:
        scores = pd.read_sql_query("SELECT forecast_cutoff, query_id, mean_distance FROM query_summary", con)
    scores = scores.groupby(["forecast_cutoff", "query_id"], as_index=False, sort=False)["mean_distance"].mean()
    instance_columns = pd.read_csv(args.query_instances_csv, nrows=0).columns
    usecols = ["query_id", "full_header", "collection_date"]
    if "prediction_branch" in instance_columns:
        usecols.append("prediction_branch")
    instances = pd.read_csv(args.query_instances_csv, usecols=usecols)
    instances["collection_date"] = pd.to_datetime(instances["collection_date"], errors="coerce")
    scores = scores.merge(instances, on="query_id", how="left", validate="many_to_many")
    scores = scores.loc[scores["full_header"].notna()].copy()
    if scores.empty:
        raise ValueError("No query summaries have matching query instances.")
    records = read_nextclade_table(args.nextclade_tsv, label_mode=args.label_mode)
    if "prediction_branch" in scores.columns:
        scored = scores.loc[scores["prediction_branch"].notna()].copy()
        scored["collection_date"] = pd.to_datetime(scored["collection_date"], errors="coerce")
    else:
        # The same sequence legitimately appears at more than one forecast cutoff;
        # only Nextclade records must be unique by EPI identifier in legacy assets.
        score_epi = scores.copy()
        score_epi["epi_isl"] = score_epi["full_header"].astype("string").str.extract(EPI_PATTERN, expand=False)
        score_epi = score_epi.loc[score_epi["epi_isl"].notna()].copy()
        records = unique_epi(records, "seqName")
        scored = score_epi.merge(records[["epi_isl", "collection_date", "prediction_branch"]], on="epi_isl", how="inner", suffixes=("_asset", "_nextclade"), validate="many_to_one")
        scored["collection_date"] = pd.to_datetime(scored["collection_date_nextclade"], errors="coerce")
    cutoffs = pd.read_csv(args.cutoffs_csv)
    details, metrics = [], []
    for row in cutoffs.itertuples(index=False):
        cutoff = str(row.forecast_cutoff)
        annual = window(scored.loc[scored.forecast_cutoff.eq(cutoff)], row.annual_start, row.annual_end)
        inferred_annual = window(scores.loc[scores.forecast_cutoff.eq(cutoff)], row.annual_start, row.annual_end)
        expected = window(instances, row.annual_start, row.annual_end)
        missing_inference = len(expected) - len(inferred_annual)
        unmatched_nextclade = len(inferred_annual) - len(annual)
        if missing_inference and not args.allow_partial:
            raise RuntimeError(f"{cutoff}: {missing_inference} required annual query summaries are missing; finish inference or use --allow-partial.")
        recent_scores = window(annual, row.recent_start, row.recent_end)
        previous_scores = window(annual, row.previous_half_start, row.previous_half_end)
        recent_all = window(records, row.recent_start, row.recent_end)
        target_all = window(records, row.target_start, row.target_end)
        annual_mean = recent_scores["mean_distance"].mean() if len(annual) else np.nan
        previous_mean = previous_scores["mean_distance"].mean() if len(previous_scores) else np.nan
        stats = recent_scores.groupby("prediction_branch")["mean_distance"].agg(score_sequence_count="count", current_mean_distance="mean", current_median_distance="median")
        current_counts = recent_all["prediction_branch"].value_counts()
        target_counts = target_all["prediction_branch"].value_counts()
        branches = current_counts.index.union(target_counts.index).union(stats.index)
        detail = pd.DataFrame(index=branches)
        detail.index.name = "branch"
        detail["current_count"] = current_counts.reindex(branches, fill_value=0)
        detail["next_count"] = target_counts.reindex(branches, fill_value=0)
        detail["current_share"] = detail.current_count / max(len(recent_all), 1)
        detail["next_share"] = detail.next_count / max(len(target_all), 1)
        detail["share_change"] = detail.next_share - detail.current_share
        for name in stats.columns:
            detail[name] = stats[name].reindex(branches)
        detail["annual_overall_mean_distance"] = annual_mean
        detail["previous_half_mean_distance"] = previous_mean
        detail["escape_excess_vs_1y_overall"] = detail.current_mean_distance - annual_mean
        detail["escape_excess_vs_previous_half"] = detail.current_mean_distance - previous_mean
        detail.loc[detail.score_sequence_count.fillna(0) < args.min_score_sequences, ["current_mean_distance", "current_median_distance", "escape_excess_vs_1y_overall", "escape_excess_vs_previous_half"]] = np.nan
        actual_top = top(detail.next_share)
        persistent = top(detail.current_share)
        escape_top = top(detail.escape_excess_vs_1y_overall)
        for name, value in {"forecast_cutoff": cutoff, "recent_start": row.recent_start, "recent_end": row.recent_end, "target_start": row.target_start, "target_end": row.target_end}.items():
            detail[name] = value
        details.append(detail.reset_index())
        metrics.append({
            "forecast_cutoff": cutoff, "matched_annual_sequences": len(annual), "missing_inference_annual_sequences": missing_inference,
            "unmatched_nextclade_annual_sequences": unmatched_nextclade,
            "recent_scored_sequences": len(recent_scores), "next_all_sequences": len(target_all),
            "eligible_branches": int(detail.escape_excess_vs_1y_overall.notna().sum()),
            "actual_next_top1": actual_top, "persistence_top1": persistent, "escape_top1": escape_top,
            "persistence_top1_hit": persistent == actual_top, "escape_top1_hit": escape_top == actual_top,
            "escape_vs_next_share_pearson": corr(detail, "escape_excess_vs_1y_overall", "next_share", "pearson"),
            "escape_vs_next_share_spearman": corr(detail, "escape_excess_vs_1y_overall", "next_share", "spearman"),
            "escape_vs_share_change_pearson": corr(detail, "escape_excess_vs_1y_overall", "share_change", "pearson"),
            "escape_vs_share_change_spearman": corr(detail, "escape_excess_vs_1y_overall", "share_change", "spearman"),
        })
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    detail_path, metric_path, summary_path = out / "antigenicity_next_window_branch_detail.csv", out / "antigenicity_next_window_metrics.csv", out / "antigenicity_next_window_summary.json"
    pd.concat(details, ignore_index=True).to_csv(detail_path, index=False)
    metric = pd.DataFrame(metrics); metric.to_csv(metric_path, index=False)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "windows": len(metric), "top1_hits": {"escape": int(metric.escape_top1_hit.sum()), "persistence": int(metric.persistence_top1_hit.sum())}, "mean_correlations": {name: metric[name].mean(skipna=True) for name in ["escape_vs_next_share_pearson", "escape_vs_next_share_spearman", "escape_vs_share_change_pearson", "escape_vs_share_change_spearman"]}, "detail_csv": str(detail_path), "metrics_csv": str(metric_path)}
    write_json(summary_path, summary, default=json_value)
    print(json.dumps(summary, default=json_value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
