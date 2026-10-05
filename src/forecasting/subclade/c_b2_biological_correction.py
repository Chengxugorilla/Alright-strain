#!/usr/bin/env python3
"""Test whether antigenicity and Fitness improve C-B2 H1N1 forecasts.

This uses the same nine six-month holdouts, 14-day bins, and frequency metrics
as ``h1n1_subclade_processing.ipynb``. At each cutoff, C-B2 is first fitted
from past Nextclade counts. Antigenicity/Fitness are centred by the current
population mean, producing relative fitness that accumulates over the future
trajectory. The coefficients are trained only on earlier holdouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.subclade.joint_biology_forecast import build_features, load_scored_sequences
from src.analysis.plot import plot_subclade_matrix
from src.forecasting.subclade.benchmarks import (
    build_window_aligned_counts,
    evaluate_frequency_forecast,
    run_six_model_backtest,
)


FEATURE_SETS = {
    "C-B2": [],
    "C-B2 + relative antigenicity": ["escape_excess"],
    "C-B2 + antigenicity-gated Fitness": ["escape_excess", "gated_fitness_excess"],
}

FEATURE_COLUMNS = ["escape_excess", "fitness_excess", "gated_fitness_excess"]


@dataclass
class ForecastWindow:
    cutoff: str
    actual: pd.DataFrame
    c_b2: pd.DataFrame
    features: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-sqlite", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/antigenicity_scores.sqlite")
    parser.add_argument("--query-instances-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv")
    parser.add_argument("--cutoffs-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels/forecast_cutoffs.csv")
    parser.add_argument("--nextclade-tsv", type=Path, default=PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv")
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--min-score-sequences", type=int, default=10)
    parser.add_argument("--min-train-cutoffs", type=int, default=3)
    parser.add_argument(
        "--fitness-antigenicity-threshold",
        type=float,
        default=2.0,
        help="Use Fitness only for clades whose recent mean antigenicity distance exceeds this value.",
    )
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument(
        "--objective",
        choices=["top1", "distribution", "poisson"],
        default="top1",
        help="Fit to the observed top branch, frequency distribution, or counts per bin.",
    )
    parser.add_argument(
        "--prediction-target",
        choices=["frequency", "count", "count_to_frequency"],
        default="frequency",
        help="Correct C-B2 frequencies, raw expected counts, or counts normalized to frequencies.",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=["relative_fitness", "static"],
        default="relative_fitness",
        help="Accumulate relative fitness over future bins, or use the former static correction.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/h1n1_c_b2_biological_correction")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for one actual-versus-predicted all-subclade plot per test window and model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def window_forecasts(
    records: pd.DataFrame,
    clade_features: pd.DataFrame,
    cutoffs: pd.DataFrame,
    trajectory_mode: str,
    prediction_target: str,
) -> list[ForecastWindow]:
    feature_lookup = {
        cutoff: frame.set_index("branch")[FEATURE_COLUMNS]
        for cutoff, frame in clade_features.groupby("forecast_cutoff", sort=False)
    }
    forecasts = []
    window_rows = cutoffs.sort_values("target_start", kind="stable")
    for cutoff_row in window_rows.itertuples(index=False):
        start = cutoff_row.target_start
        end = cutoff_row.target_end
        counts, _, horizon = build_window_aligned_counts(records, window_start=start, window_end=end)
        cutoff_time = pd.Timestamp(start) - pd.Timedelta(days=1)
        frequencies, count_predictions, actual_counts, actual_frequencies, _ = run_six_model_backtest(
            counts,
            cutoff=cutoff_time,
            horizon_bins=horizon,
        )
        cutoff = (pd.Timestamp(start) - pd.Timedelta(days=1)).date().isoformat()
        if cutoff not in feature_lookup:
            continue
        if prediction_target in {"count", "count_to_frequency"}:
            actual = actual_counts
            c_b2 = count_predictions["C-B2"]
        else:
            actual = actual_frequencies
            c_b2 = frequencies["C-B2"]
        if prediction_target == "count_to_frequency":
            actual = actual_frequencies
        c_b2 = c_b2.reindex(index=actual.index, columns=actual.columns, fill_value=0.0)
        features = feature_lookup[cutoff].reindex(actual.columns).fillna(0.0)
        if trajectory_mode == "relative_fitness":
            current = counts.loc[counts.index <= cutoff_time].iloc[-1].reindex(actual.columns, fill_value=0.0)
            current = current / current.sum()
            features = features - current @ features
        forecasts.append(ForecastWindow(cutoff, actual, c_b2, features))
    if not forecasts:
        raise RuntimeError("No C-B2 windows could be aligned with antigenicity/Fitness features.")
    return forecasts


def fit_correction(
    windows: list[ForecastWindow],
    columns: list[str],
    l2: float,
    objective: str,
    trajectory_mode: str,
    prediction_target: str,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    if not columns:
        return np.empty(0), pd.Series(dtype=float), pd.Series(dtype=float)
    values = pd.concat([window.features[columns] for window in windows], ignore_index=True)
    mean = values.mean()
    scale = values.std(ddof=0).replace(0, 1.0)

    def loss(coefficients: np.ndarray) -> float:
        total = 0.0
        bins = 0
        for window in windows:
            adjustment = ((window.features[columns] - mean) / scale).to_numpy() @ coefficients
            if trajectory_mode == "relative_fitness":
                adjustment = np.arange(1, len(window.c_b2) + 1)[:, None] * adjustment
            observed = window.actual.to_numpy()
            if prediction_target == "count":
                predicted = window.c_b2.clip(lower=1e-12).to_numpy() * np.exp(adjustment)
                total += np.sum(predicted - observed * np.log(predicted + 1e-12))
            else:
                logits = np.log(window.c_b2.clip(lower=1e-12).to_numpy()) + adjustment
                logits -= logits.max(axis=1, keepdims=True)
                predicted = np.exp(logits)
                predicted /= predicted.sum(axis=1, keepdims=True)
                if objective == "top1":
                    labels = observed.argmax(axis=1)
                    observed = np.eye(observed.shape[1])[labels]
                total -= np.sum(observed * np.log(predicted + 1e-12))
            bins += len(window.actual)
        return total / bins + l2 * np.square(coefficients).sum()

    result = minimize(loss, np.zeros(len(columns)), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"C-B2 correction fit failed: {result.message}")
    return result.x, mean, scale


def apply_correction(
    window: ForecastWindow,
    columns: list[str],
    coefficients: np.ndarray,
    mean: pd.Series,
    scale: pd.Series,
    trajectory_mode: str,
    prediction_target: str,
) -> pd.DataFrame:
    if not columns:
        if prediction_target == "count_to_frequency":
            values = window.c_b2.to_numpy(dtype=float)
            values /= values.sum(axis=1, keepdims=True)
            return pd.DataFrame(values, index=window.c_b2.index, columns=window.c_b2.columns)
        return window.c_b2.copy()
    adjustment = ((window.features[columns] - mean) / scale).to_numpy() @ coefficients
    if trajectory_mode == "relative_fitness":
        adjustment = np.arange(1, len(window.c_b2) + 1)[:, None] * adjustment
    if prediction_target == "count":
        values = window.c_b2.to_numpy() * np.exp(adjustment)
        return pd.DataFrame(values, index=window.c_b2.index, columns=window.c_b2.columns)
    logits = np.log(window.c_b2.clip(lower=1e-12).to_numpy()) + adjustment
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=window.c_b2.index, columns=window.c_b2.columns)


def evaluate(
    window: ForecastWindow, model: str, prediction: pd.DataFrame, prediction_target: str
) -> dict[str, object]:
    if prediction_target == "count":
        observed = window.actual.to_numpy(dtype=float)
        expected = prediction.reindex_like(window.actual).to_numpy(dtype=float).clip(min=1e-12)
        error = expected - observed
        metrics = {
            "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "mean_Poisson_NLL": float(
                (expected - observed * np.log(expected) + gammaln(observed + 1)).mean()
            ),
        }
    else:
        metrics = evaluate_frequency_forecast(window.actual, prediction).to_dict()
    return {
        "forecast_cutoff": window.cutoff,
        "model": model,
        **metrics,
        "top1_accuracy": float((prediction.idxmax(axis=1) == window.actual.idxmax(axis=1)).mean()),
    }


def rolling_test(
    windows: list[ForecastWindow],
    min_train_cutoffs: int,
    l2: float,
    objective: str,
    trajectory_mode: str,
    prediction_target: str,
    plot_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details, coefficients = [], []
    for index, test in enumerate(windows):
        train = windows[:index]
        if len(train) < min_train_cutoffs:
            continue
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            actual_figure = plot_subclade_matrix(
                test.actual.to_numpy()[:, None, :],
                test.actual.index,
                test.actual.columns,
                ["actual"],
            )
            actual_figure.axes[0].set(
                title=f"{test.cutoff}: all subclades · actual",
                ylabel="count" if prediction_target == "count" else "frequency",
            )
            actual_figure.savefig(plot_dir / f"{test.cutoff}_actual.png", dpi=180, bbox_inches="tight")
            plt.close(actual_figure)
        for model, feature_columns in FEATURE_SETS.items():
            weights, mean, scale = fit_correction(
                train, feature_columns, l2, objective, trajectory_mode, prediction_target
            )
            prediction = apply_correction(
                test, feature_columns, weights, mean, scale, trajectory_mode, prediction_target
            )
            details.append(evaluate(test, model, prediction, prediction_target))
            if plot_dir is not None:
                filename = f"{test.cutoff}_{model.replace(' ', '_').replace('+', 'plus')}.png"
                prediction = prediction.reindex_like(test.actual)
                figure = plot_subclade_matrix(
                    prediction.to_numpy()[:, None, :],
                    test.actual.index,
                    test.actual.columns,
                    [model],
                )
                figure.axes[0].set(
                    title=f"{test.cutoff}: all subclades · {model}",
                    ylabel="count" if prediction_target == "count" else "frequency",
                )
                figure.savefig(plot_dir / filename, dpi=180, bbox_inches="tight")
                plt.close(figure)
            coefficients.extend(
                {"forecast_cutoff": test.cutoff, "model": model, "feature": feature, "coefficient": weight}
                for feature, weight in zip(feature_columns, weights, strict=True)
            )
    if not details:
        raise RuntimeError("No test window remains; lower --min-train-cutoffs.")
    return pd.DataFrame(details), pd.DataFrame(coefficients)


def summarize(detail: pd.DataFrame, prediction_target: str) -> pd.DataFrame:
    metrics = {
        "windows": ("forecast_cutoff", "nunique"),
        "mean_MAE": ("MAE", "mean"),
        "mean_RMSE": ("RMSE", "mean"),
        "mean_top1_accuracy": ("top1_accuracy", "mean"),
    }
    if prediction_target == "count":
        metrics["mean_Poisson_NLL"] = ("mean_Poisson_NLL", "mean")
        sort_metric = "mean_Poisson_NLL"
    else:
        metrics["mean_TVD"] = ("mean_TVD", "mean")
        metrics["mean_JSD"] = ("mean_JSD", "mean")
        sort_metric = "mean_TVD"
    return detail.groupby("model", sort=False).agg(**metrics).reset_index().sort_values(sort_metric, ignore_index=True)


def main() -> None:
    args = parse_args()
    if (
        args.min_score_sequences < 1
        or args.min_train_cutoffs < 1
        or args.l2 < 0
        or args.fitness_antigenicity_threshold < 0
    ):
        raise ValueError("Minimum counts must be positive and --l2 must be non-negative.")
    if args.prediction_target == "count" and args.objective != "poisson":
        raise ValueError("Count prediction requires --objective poisson.")
    if args.prediction_target != "count" and args.objective == "poisson":
        raise ValueError("Frequency evaluation requires --objective top1 or distribution.")
    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "detail": output_dir / "rolling_metrics.csv",
        "summary": output_dir / "rolling_summary.csv",
        "full_baseline": output_dir / "c_b2_all_nine_windows.csv",
        "coefficients": output_dir / "rolling_coefficients.csv",
        "report": output_dir / "summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = (args.plot_dir or output_dir / "plots").expanduser().resolve()

    scored, records = load_scored_sequences(args)
    cutoffs = pd.read_csv(args.cutoffs_csv)
    features = build_features(scored, records, cutoffs, args.min_score_sequences)
    features["gated_fitness_excess"] = features["fitness_excess"].where(
        features["mean_distance"] > args.fitness_antigenicity_threshold,
        0.0,
    )
    windows = window_forecasts(records, features, cutoffs, args.trajectory_mode, args.prediction_target)
    detail, coefficients = rolling_test(
        windows,
        args.min_train_cutoffs,
        args.l2,
        args.objective,
        args.trajectory_mode,
        args.prediction_target,
        plot_dir,
    )
    summary = summarize(detail, args.prediction_target)
    full_baseline = pd.DataFrame(
        [evaluate(window, "C-B2", window.c_b2, args.prediction_target) for window in windows]
    )
    detail.to_csv(outputs["detail"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    full_baseline.to_csv(outputs["full_baseline"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    report = {
        "test_framework": f"C-B2 {args.prediction_target} baseline with 14-day rolling holdouts",
        "training_policy": "each correction uses only earlier holdouts",
        "prediction_target": args.prediction_target,
        "correction_objective": args.objective,
        "trajectory_mode": args.trajectory_mode,
        "fitness_antigenicity_threshold": args.fitness_antigenicity_threshold,
        "gated_fitness_rule": "Fitness is used only when clade mean_distance exceeds the threshold.",
        "plot_dir": str(plot_dir),
        "relative_fitness_rule": "C-B2(t, clade) × exp(t × relative_fitness(clade))",
        "c_b2_all_nine_windows": summarize(full_baseline, args.prediction_target).to_dict(orient="records")[0],
        "tested_cutoffs": sorted(detail["forecast_cutoff"].unique()),
        "models": summary.to_dict(orient="records"),
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
