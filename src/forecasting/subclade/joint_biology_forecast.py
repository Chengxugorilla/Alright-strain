#!/usr/bin/env python3
"""Small rolling demo: do antigenicity and Fitness improve clade forecasts?

For every forecast cutoff, the script averages sequence scores within each
recent clade.  It predicts that clade's next-six-month share change from:
current share, antigenic escape excess, and reference-free Fitness excess.
Only earlier cutoffs are used to train each forecast.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strain_data.nextclade import read_nextclade_table


IDENTITY = ["fasta_sha256", "sequence_index", "sequence_sha256"]
FEATURES = ["current_share", "escape_excess", "fitness_excess"]
EPI_PATTERN = r"\|(EPI_ISL_\d+)\|"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-sqlite",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/antigenicity_scores.sqlite",
    )
    parser.add_argument(
        "--query-instances-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv",
    )
    parser.add_argument(
        "--cutoffs-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels/forecast_cutoffs.csv",
    )
    parser.add_argument(
        "--nextclade-tsv",
        type=Path,
        default=PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv",
    )
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--min-score-sequences", type=int, default=10)
    parser.add_argument("--min-train-cutoffs", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_joint_forecast_demo",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_scored_sequences(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(args.scores_sqlite) as connection:
        scores = pd.read_sql_query(
            "SELECT forecast_cutoff, query_id, mean_distance FROM query_summary",
            connection,
        )
    scores = scores.groupby(["forecast_cutoff", "query_id"], as_index=False, sort=False)["mean_distance"].mean()
    instance_columns = pd.read_csv(args.query_instances_csv, nrows=0).columns
    usecols = ["query_id", "full_header", "collection_date", "bert_score"]
    if "prediction_branch" in instance_columns:
        usecols.append("prediction_branch")
    instances = pd.read_csv(
        args.query_instances_csv,
        usecols=usecols,
    )
    scores = scores.merge(instances, on="query_id", how="left", validate="many_to_many")
    scores = scores.loc[scores["full_header"].notna()].copy()
    if scores.empty:
        raise ValueError("No query summaries have matching query instance records.")

    records = read_nextclade_table(args.nextclade_tsv, label_mode=args.label_mode)
    if "prediction_branch" in scores.columns:
        scored = scores.loc[scores["prediction_branch"].notna()].copy()
    else:
        scores["epi_isl"] = scores["full_header"].astype("string").str.extract(EPI_PATTERN, expand=False)
        scores = scores.loc[scores["epi_isl"].notna()].copy()
        records = records.copy()
        records["epi_isl"] = records["seqName"].astype("string").str.extract(EPI_PATTERN, expand=False)
        records = records.loc[records["epi_isl"].notna() & ~records["epi_isl"].duplicated(keep=False)].copy()
        scored = scores.merge(
            records[["epi_isl", "collection_date", "prediction_branch"]],
            on="epi_isl",
            how="inner",
            validate="many_to_one",
        )
    scored["collection_date"] = pd.to_datetime(scored["collection_date"], errors="coerce")
    return scored.dropna(subset=["collection_date"]), records


def within(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[frame["collection_date"].between(start, end, inclusive="both")].copy()


def build_features(
    scored: pd.DataFrame,
    records: pd.DataFrame,
    cutoffs: pd.DataFrame,
    min_score_sequences: int,
) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs.itertuples(index=False):
        annual = within(
            scored.loc[scored["forecast_cutoff"].eq(str(cutoff.forecast_cutoff))],
            cutoff.annual_start,
            cutoff.annual_end,
        )
        recent = within(annual, cutoff.recent_start, cutoff.recent_end)
        if annual.empty or recent.empty:
            continue

        clades = recent.groupby("prediction_branch").agg(
            scored_sequences=("mean_distance", "size"),
            mean_distance=("mean_distance", "mean"),
            mean_fitness=("bert_score", "mean"),
        )
        clades = clades.loc[clades["scored_sequences"] >= min_score_sequences].copy()
        if clades.empty:
            continue

        current = within(records, cutoff.recent_start, cutoff.recent_end)["prediction_branch"].value_counts()
        future = within(records, cutoff.target_start, cutoff.target_end)["prediction_branch"].value_counts()
        clades["current_share"] = current.reindex(clades.index, fill_value=0) / max(current.sum(), 1)
        clades["next_share"] = future.reindex(clades.index, fill_value=0) / max(future.sum(), 1)
        clades["share_change"] = clades["next_share"] - clades["current_share"]
        clades["escape_excess"] = clades["mean_distance"] - annual["mean_distance"].mean()
        clades["fitness_excess"] = clades["mean_fitness"] - annual["bert_score"].mean()
        clades["forecast_cutoff"] = str(cutoff.forecast_cutoff)
        rows.append(clades.reset_index(names="branch"))

    if not rows:
        raise RuntimeError("No cutoff has enough scored clade sequences for the requested minimum.")
    return pd.concat(rows, ignore_index=True)


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> np.ndarray:
    mean = train[FEATURES].mean()
    scale = train[FEATURES].std(ddof=0).replace(0, 1.0)
    train_x = ((train[FEATURES] - mean) / scale).to_numpy(dtype=float)
    test_x = ((test[FEATURES] - mean) / scale).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(train_x)), train_x])
    penalty = np.diag([0.0, *([alpha] * len(FEATURES))])
    target = train["share_change"].to_numpy(dtype=float)
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def rolling_predictions(features: pd.DataFrame, min_train_cutoffs: int, alpha: float) -> pd.DataFrame:
    predictions = []
    cutoffs = sorted(features["forecast_cutoff"].unique())
    for index, cutoff in enumerate(cutoffs):
        earlier = cutoffs[:index]
        if len(earlier) < min_train_cutoffs:
            continue
        train = features.loc[features["forecast_cutoff"].isin(earlier)].dropna(subset=[*FEATURES, "share_change"])
        test = features.loc[features["forecast_cutoff"].eq(cutoff)].dropna(subset=FEATURES).copy()
        if len(train) <= len(FEATURES) or test.empty:
            continue
        test["predicted_share_change"] = ridge_predict(train, test, alpha)
        test["persistence_prediction"] = 0.0
        test["train_cutoff_count"] = len(earlier)
        predictions.append(test)
    if not predictions:
        raise RuntimeError("No rolling test window remains; lower --min-train-cutoffs or check input coverage.")
    return pd.concat(predictions, ignore_index=True)


def model_metrics(predictions: pd.DataFrame, prediction_column: str) -> dict[str, float | int | None]:
    error = predictions[prediction_column] - predictions["share_change"]
    hits = 0
    for _, group in predictions.groupby("forecast_cutoff", sort=False):
        actual = group.loc[group["next_share"].idxmax(), "branch"]
        predicted_share = group["current_share"] + group[prediction_column]
        predicted = group.loc[predicted_share.idxmax(), "branch"]
        hits += predicted == actual
    spearman = predictions[[prediction_column, "share_change"]].corr(method="spearman").iloc[0, 1]
    return {
        "rows": int(len(predictions)),
        "cutoffs": int(predictions["forecast_cutoff"].nunique()),
        "share_change_mae": float(error.abs().mean()),
        "share_change_spearman": float(spearman) if pd.notna(spearman) else None,
        "top1_hits": int(hits),
    }


def main() -> None:
    args = parse_args()
    if args.min_score_sequences < 1 or args.min_train_cutoffs < 1 or args.ridge_alpha < 0:
        raise ValueError("Minimum counts must be positive and --ridge-alpha must be non-negative.")

    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "features": output_dir / "clade_features.csv",
        "predictions": output_dir / "rolling_predictions.csv",
        "summary": output_dir / "summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    scored, records = load_scored_sequences(args)
    features = build_features(scored, records, pd.read_csv(args.cutoffs_csv), args.min_score_sequences)
    predictions = rolling_predictions(features, args.min_train_cutoffs, args.ridge_alpha)
    features.to_csv(outputs["features"], index=False)
    predictions.to_csv(outputs["predictions"], index=False)

    summary = {
        "model": "ridge regression of next-six-month share change",
        "features": FEATURES,
        "test_policy": "each cutoff is trained only on earlier cutoffs",
        "test_cutoffs": sorted(predictions["forecast_cutoff"].unique()),
        "eligible_feature_rows": int(len(features)),
        "joint_model": model_metrics(predictions, "predicted_share_change"),
        "persistence_baseline": model_metrics(predictions, "persistence_prediction"),
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
