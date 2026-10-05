#!/usr/bin/env python3
"""Forecast future H1N1 clade change from count trends plus biological features.

This script is intentionally separate from ``c_b2_biological_correction_demo``.
Instead of applying a post-hoc multiplier to a C-B2 trajectory, it fits a
rolling multinomial model whose features explicitly describe the recent count
trend of each clade and optionally add antigenicity/Fitness features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.subclade.joint_biology_forecast import build_features, load_scored_sequences
from src.forecasting.subclade.benchmarks import build_window_aligned_counts, evaluate_frequency_forecast, run_six_model_backtest


COUNT_DIAGNOSTIC_FEATURES = [
    "c_b2_log_growth",
    "log_current_share",
    "log_recent_count",
    "log_count_growth",
    "log_share_growth",
    "logit_presence",
    "composition_slope",
]
BIOLOGY_FEATURES = ["escape_excess", "fitness_excess", "escape_fitness_interaction", "bio_available"]
FEATURE_SETS = {
    "Count trend offset": [],
    "Count trend offset + antigenicity": ["escape_excess", "bio_available"],
    "Count trend offset + Fitness": ["fitness_excess", "bio_available"],
    "Count trend offset + antigenicity + Fitness": BIOLOGY_FEATURES,
    "Count trend offset + count diagnostics + biology": [*COUNT_DIAGNOSTIC_FEATURES, *BIOLOGY_FEATURES],
}


@dataclass
class ForecastWindow:
    cutoff: str
    rows: pd.DataFrame
    c_b2_share: pd.Series
    c_b2_count: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-sqlite",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/past_strain_3y_full_excluding_recent6m.sqlite",
    )
    parser.add_argument(
        "--query-instances-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv",
    )
    parser.add_argument(
        "--cutoffs-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels_2017_2025_past_strain_3y/forecast_cutoffs.csv",
    )
    parser.add_argument(
        "--nextclade-tsv",
        type=Path,
        default=PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv",
    )
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--min-score-sequences", type=int, default=10)
    parser.add_argument("--min-train-cutoffs", type=int, default=3)
    parser.add_argument("--lookback-bins", type=int, default=26)
    parser.add_argument("--recent-bins", type=int, default=13)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument(
        "--nonnegative-antigenicity",
        action="store_true",
        help="Constrain the escape_excess coefficient to be non-negative when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/h1n1_count_trend_biology_forecast",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max()
    weights = np.exp(values)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(weights, 1.0 / len(weights), dtype=float)
    return weights / total


def safe_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float).clip(lower=0.0)
    total = values.sum()
    if total <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / total


def composition_slope(history: pd.DataFrame, pseudocount: float) -> pd.Series:
    counts = history.astype(float)
    active = counts.sum(axis=0) > 0
    result = pd.Series(0.0, index=counts.columns)
    if active.sum() < 2 or len(counts) < 2:
        return result
    active_counts = counts.loc[:, active]
    totals = active_counts.sum(axis=1)
    probabilities = (active_counts + pseudocount).div(totals + pseudocount * active.sum(), axis=0)
    log_probabilities = np.log(probabilities.to_numpy(dtype=float))
    log_probabilities -= log_probabilities.mean(axis=1, keepdims=True)
    log_growth = np.diff(log_probabilities, axis=0)
    if not len(log_growth):
        return result
    transition_weight = np.sqrt(np.minimum(totals.iloc[:-1].to_numpy(), totals.iloc[1:].to_numpy()))
    transition_weight = transition_weight * np.linspace(0.25, 1.0, len(log_growth))
    if transition_weight.sum() <= 0:
        growth = log_growth.mean(axis=0)
    else:
        growth = np.average(log_growth, axis=0, weights=transition_weight)
    result.loc[active_counts.columns] = growth
    return result


def count_trend_rows(
    counts: pd.DataFrame,
    cutoff_time: pd.Timestamp,
    horizon_bins: int,
    *,
    lookback_bins: int,
    recent_bins: int,
    pseudocount: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    history = counts.loc[counts.index <= cutoff_time].tail(lookback_bins).copy()
    if history.empty:
        raise ValueError(f"No history exists before {cutoff_time.date()}.")
    future = counts.loc[counts.index > cutoff_time].head(horizon_bins).copy()
    future = future.loc[future.sum(axis=1) > 0]
    if future.empty:
        raise ValueError(f"No non-empty future bins exist after {cutoff_time.date()}.")

    latest = history.iloc[-1].astype(float)
    recent = history.tail(recent_bins).sum(axis=0).astype(float)
    previous = history.iloc[max(0, len(history) - 2 * recent_bins): max(0, len(history) - recent_bins)].sum(axis=0)
    previous = previous.reindex(history.columns, fill_value=0.0).astype(float)
    current_share = safe_normalize(latest)
    recent_share = safe_normalize(recent)
    previous_share = safe_normalize(previous) if previous.sum() > 0 else current_share.copy()
    presence = (history.tail(recent_bins) > 0).mean(axis=0).clip(1e-6, 1 - 1e-6)
    slope = composition_slope(history, pseudocount)

    future_count = future.sum(axis=0).astype(float)
    future_share = safe_normalize(future_count)
    rows = pd.DataFrame(
        {
            "branch": history.columns.astype(str),
            "current_count": latest.to_numpy(dtype=float),
            "recent_count": recent.to_numpy(dtype=float),
            "previous_count": previous.to_numpy(dtype=float),
            "actual_future_count": future_count.reindex(history.columns, fill_value=0.0).to_numpy(dtype=float),
            "actual_future_share": future_share.reindex(history.columns, fill_value=0.0).to_numpy(dtype=float),
            "log_current_share": np.log(current_share.reindex(history.columns).to_numpy(dtype=float) + 1e-9),
            "log_recent_count": np.log1p(recent.reindex(history.columns).to_numpy(dtype=float)),
            "log_count_growth": np.log(recent.reindex(history.columns).to_numpy(dtype=float) + pseudocount)
            - np.log(previous.reindex(history.columns).to_numpy(dtype=float) + pseudocount),
            "log_share_growth": np.log(recent_share.reindex(history.columns).to_numpy(dtype=float) + 1e-9)
            - np.log(previous_share.reindex(history.columns).to_numpy(dtype=float) + 1e-9),
            "logit_presence": np.log(presence.reindex(history.columns).to_numpy(dtype=float))
            - np.log1p(-presence.reindex(history.columns).to_numpy(dtype=float)),
            "composition_slope": slope.reindex(history.columns).to_numpy(dtype=float),
        }
    )
    return rows, future_count, future_share, future.sum(axis=1)


def attach_biology(rows: pd.DataFrame, biology: pd.DataFrame | None) -> pd.DataFrame:
    result = rows.copy()
    if biology is None or biology.empty:
        for column in ["escape_excess", "fitness_excess", "mean_distance", "mean_fitness", "scored_sequences"]:
            result[column] = 0.0
        result["bio_available"] = 0.0
    else:
        bio = biology.set_index("branch")
        result = result.merge(
            bio[["escape_excess", "fitness_excess", "mean_distance", "mean_fitness", "scored_sequences"]],
            left_on="branch",
            right_index=True,
            how="left",
        )
        result["bio_available"] = result["escape_excess"].notna().astype(float)
        fill_values = {
            "escape_excess": 0.0,
            "fitness_excess": 0.0,
            "mean_distance": 0.0,
            "mean_fitness": 0.0,
            "scored_sequences": 0.0,
        }
        result = result.fillna(fill_values)
    result["escape_fitness_interaction"] = result["escape_excess"].clip(lower=0.0) * result["fitness_excess"]
    return result


def build_windows(args: argparse.Namespace) -> list[ForecastWindow]:
    scored, records = load_scored_sequences(args)
    cutoffs = pd.read_csv(args.cutoffs_csv)
    biology_features = build_features(scored, records, cutoffs, args.min_score_sequences)
    biology_by_cutoff = {
        cutoff: frame.copy()
        for cutoff, frame in biology_features.groupby("forecast_cutoff", sort=False)
    }
    windows: list[ForecastWindow] = []
    for cutoff_row in cutoffs.sort_values("target_start", kind="stable").itertuples(index=False):
        cutoff = str(cutoff_row.forecast_cutoff)
        target_start = pd.Timestamp(cutoff_row.target_start)
        target_end = pd.Timestamp(cutoff_row.target_end)
        cutoff_time = target_start - pd.Timedelta(days=1)
        try:
            counts, _, horizon_bins = build_window_aligned_counts(
                records,
                window_start=target_start,
                window_end=target_end,
            )
            rows, _, _, _ = count_trend_rows(
                counts,
                cutoff_time,
                horizon_bins,
                lookback_bins=args.lookback_bins,
                recent_bins=args.recent_bins,
                pseudocount=args.pseudocount,
            )
            frequencies, count_predictions, actual_counts, _, _ = run_six_model_backtest(
                counts,
                cutoff=cutoff_time,
                horizon_bins=horizon_bins,
                lookback_bins=args.lookback_bins,
            )
        except ValueError:
            continue
        rows = attach_biology(rows, biology_by_cutoff.get(cutoff))
        rows.insert(0, "forecast_cutoff", cutoff)
        c_b2_count = count_predictions["C-B2"].reindex(
            index=actual_counts.index, columns=rows["branch"], fill_value=0.0
        ).sum(axis=0)
        c_b2_share = safe_normalize(c_b2_count)
        # Keep this for direct side-by-side reference with the local C-B2
        # frequency trajectory in the same future bins.
        c_b2_frequency_share = safe_normalize(
            frequencies["C-B2"].reindex(index=actual_counts.index, columns=rows["branch"], fill_value=0.0).mean(axis=0)
        )
        rows["c_b2_aggregate_share"] = c_b2_share.reindex(rows["branch"]).to_numpy(dtype=float)
        rows["c_b2_frequency_mean_share"] = c_b2_frequency_share.reindex(rows["branch"]).to_numpy(dtype=float)
        rows["c_b2_log_share"] = np.log(rows["c_b2_aggregate_share"].clip(lower=1e-9))
        rows["c_b2_log_growth"] = rows["c_b2_log_share"] - rows["log_current_share"]
        windows.append(ForecastWindow(cutoff=cutoff, rows=rows, c_b2_share=c_b2_share, c_b2_count=c_b2_count))
    if not windows:
        raise RuntimeError("No valid count-trend windows could be built.")
    return windows


def standardize(train: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    if not columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    values = train[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = values.mean()
    scale = values.std(ddof=0).replace(0.0, 1.0)
    return mean, scale


def fit_multinomial(
    windows: list[ForecastWindow],
    columns: list[str],
    *,
    l2: float,
    nonnegative_antigenicity: bool,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    train_rows = pd.concat([window.rows for window in windows], ignore_index=True)
    mean, scale = standardize(train_rows, columns)
    if not columns:
        return np.empty(0), mean, scale

    def loss(coefficients: np.ndarray) -> float:
        total = 0.0
        for window in windows:
            x = ((window.rows[columns] - mean) / scale).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            logits = window.rows["c_b2_log_share"].to_numpy(dtype=float) + x.to_numpy(dtype=float) @ coefficients
            predicted = softmax(logits)
            observed = window.rows["actual_future_share"].to_numpy(dtype=float)
            total -= float(np.sum(observed * np.log(predicted + 1e-12)))
        return total / len(windows) + l2 * float(np.square(coefficients).sum())

    bounds = None
    if nonnegative_antigenicity and "escape_excess" in columns:
        bounds = [(None, None)] * len(columns)
        bounds[columns.index("escape_excess")] = (0.0, None)
    result = minimize(loss, np.zeros(len(columns)), method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Count-trend biology fit failed: {result.message}")
    return result.x, mean, scale


def predict_window(
    window: ForecastWindow,
    columns: list[str],
    coefficients: np.ndarray,
    mean: pd.Series,
    scale: pd.Series,
) -> pd.Series:
    logits = window.rows["c_b2_log_share"].to_numpy(dtype=float)
    if columns:
        x = ((window.rows[columns] - mean) / scale).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        logits = logits + x.to_numpy(dtype=float) @ coefficients
    values = softmax(logits)
    return pd.Series(values, index=window.rows["branch"].astype(str), name="predicted_share")


def evaluate_share(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    columns = actual.index.union(predicted.index, sort=False)
    actual_values = actual.reindex(columns, fill_value=0.0).to_numpy(dtype=float)
    predicted_values = predicted.reindex(columns, fill_value=0.0).to_numpy(dtype=float)
    actual_values = actual_values / actual_values.sum()
    predicted_values = predicted_values / predicted_values.sum()
    error = predicted_values - actual_values
    midpoint = 0.5 * (actual_values + predicted_values)
    epsilon = 1e-12
    jsd = 0.5 * np.sum(
        actual_values * np.log((actual_values + epsilon) / (midpoint + epsilon))
        + predicted_values * np.log((predicted_values + epsilon) / (midpoint + epsilon))
    )
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "TVD": float(0.5 * np.abs(error).sum()),
        "JSD": float(jsd),
        "top1_accuracy": float(predicted_values.argmax() == actual_values.argmax()),
    }


def evaluate_count(actual_count: pd.Series, predicted_count: pd.Series) -> dict[str, float]:
    columns = actual_count.index.union(predicted_count.index, sort=False)
    actual = actual_count.reindex(columns, fill_value=0.0).to_numpy(dtype=float)
    predicted = predicted_count.reindex(columns, fill_value=0.0).to_numpy(dtype=float)
    error = predicted - actual
    return {
        "count_MAE": float(np.abs(error).mean()),
        "count_RMSE": float(np.sqrt(np.square(error).mean())),
    }


def rolling_test(
    windows: list[ForecastWindow],
    *,
    min_train_cutoffs: int,
    l2: float,
    nonnegative_antigenicity: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for index, test in enumerate(windows):
        train = windows[:index]
        if len(train) < min_train_cutoffs:
            continue
        actual_share = test.rows.set_index("branch")["actual_future_share"]
        actual_count = test.rows.set_index("branch")["actual_future_count"]

        c_b2_count_prediction = test.c_b2_share * max(float(test.c_b2_count.sum()), 1.0)
        c_b2_metrics = evaluate_share(actual_share, test.c_b2_share)
        c_b2_metrics.update(evaluate_count(actual_count, c_b2_count_prediction))
        detail_rows.append({"forecast_cutoff": test.cutoff, "model": "C-B2 aggregate reference", **c_b2_metrics})

        c_b2_frame = pd.DataFrame(
            {
                "forecast_cutoff": test.cutoff,
                "model": "C-B2 aggregate reference",
                "branch": test.c_b2_share.index,
                "predicted_share": test.c_b2_share.to_numpy(dtype=float),
                "actual_future_share": actual_share.reindex(test.c_b2_share.index, fill_value=0.0).to_numpy(dtype=float),
                "predicted_count": c_b2_count_prediction.to_numpy(dtype=float),
                "actual_future_count": actual_count.reindex(test.c_b2_share.index, fill_value=0.0).to_numpy(dtype=float),
            }
        )
        prediction_rows.append(c_b2_frame)

        for model, columns in FEATURE_SETS.items():
            weights, mean, scale = fit_multinomial(
                train,
                columns,
                l2=l2,
                nonnegative_antigenicity=nonnegative_antigenicity,
            )
            predicted_share = predict_window(test, columns, weights, mean, scale)
            predicted_count = predicted_share * max(float(test.c_b2_count.sum()), 1.0)
            metrics = evaluate_share(actual_share, predicted_share)
            metrics.update(evaluate_count(actual_count, predicted_count))
            detail_rows.append({"forecast_cutoff": test.cutoff, "model": model, **metrics})
            coefficient_rows.extend(
                {
                    "forecast_cutoff": test.cutoff,
                    "model": model,
                    "feature": feature,
                    "coefficient": coefficient,
                    "feature_mean": float(mean[feature]),
                    "feature_scale": float(scale[feature]),
                }
                for feature, coefficient in zip(columns, weights, strict=True)
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "forecast_cutoff": test.cutoff,
                        "model": model,
                        "branch": predicted_share.index,
                        "predicted_share": predicted_share.to_numpy(dtype=float),
                        "actual_future_share": actual_share.reindex(predicted_share.index, fill_value=0.0).to_numpy(dtype=float),
                        "predicted_count": predicted_count.to_numpy(dtype=float),
                        "actual_future_count": actual_count.reindex(predicted_share.index, fill_value=0.0).to_numpy(dtype=float),
                    }
                )
            )
    if not detail_rows:
        raise RuntimeError("No rolling test window remains; lower --min-train-cutoffs.")
    return (
        pd.DataFrame(detail_rows),
        pd.DataFrame(coefficient_rows),
        pd.concat(prediction_rows, ignore_index=True),
    )


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby("model", sort=False)
        .agg(
            windows=("forecast_cutoff", "nunique"),
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_TVD=("TVD", "mean"),
            mean_JSD=("JSD", "mean"),
            mean_top1_accuracy=("top1_accuracy", "mean"),
            mean_count_MAE=("count_MAE", "mean"),
            mean_count_RMSE=("count_RMSE", "mean"),
        )
        .reset_index()
        .sort_values("mean_TVD", ignore_index=True)
    )


def main() -> None:
    args = parse_args()
    if (
        args.min_score_sequences < 1
        or args.min_train_cutoffs < 1
        or args.lookback_bins < 3
        or args.recent_bins < 1
        or args.pseudocount <= 0
        or args.l2 < 0
    ):
        raise ValueError("Minimum counts/bins must be positive, --pseudocount > 0, and --l2 >= 0.")

    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "features": output_dir / "feature_rows.csv",
        "detail": output_dir / "rolling_metrics.csv",
        "summary": output_dir / "rolling_summary.csv",
        "coefficients": output_dir / "rolling_coefficients.csv",
        "predictions": output_dir / "rolling_predictions.csv",
        "report": output_dir / "summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = build_windows(args)
    detail, coefficients, predictions = rolling_test(
        windows,
        min_train_cutoffs=args.min_train_cutoffs,
        l2=args.l2,
        nonnegative_antigenicity=args.nonnegative_antigenicity,
    )
    summary = summarize(detail)
    pd.concat([window.rows for window in windows], ignore_index=True).to_csv(outputs["features"], index=False)
    detail.to_csv(outputs["detail"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    predictions.to_csv(outputs["predictions"], index=False)

    report = {
        "test_framework": "rolling multinomial forecast of next-half-year aggregate clade counts",
        "training_policy": "each test cutoff is trained only on earlier cutoffs",
        "windows_built": len(windows),
        "tested_cutoffs": sorted(detail["forecast_cutoff"].unique()),
        "lookback_bins": args.lookback_bins,
        "recent_bins": args.recent_bins,
        "pseudocount": args.pseudocount,
        "l2": args.l2,
        "nonnegative_antigenicity": args.nonnegative_antigenicity,
        "feature_sets": FEATURE_SETS,
        "models": summary.to_dict(orient="records"),
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
