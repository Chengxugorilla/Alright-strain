#!/usr/bin/env python3
"""Evaluate the official Nextstrain MLR in the C-B2 correction framework, both
as a parallel model and as an alternative base trajectory.

This is a companion script to ``src/fitness/c_b2_biological_correction_demo.py``
(it does NOT modify it).  It re-uses that framework's data loading, feature
building, correction fitting and evaluation verbatim, and runs a two-by-four
grid of models over the same nine six-month windows:

* base trajectory: the C-B2 proxy prediction (the original framework) or the
  official Nextstrain forecasts-flu MLR prediction (evofr
  ``MultinomialLogisticRegression``, official MAP+NUTS settings, fitted per
  window on the same counts/cutoff/26-bin lookback as the C-B2 proxies);
* feature correction on top of the base: none, relative antigenicity,
  relative Fitness, or both (``base × exp(t · coefficient · feature)``,
  coefficients trained only on earlier holdouts).

The MLR-base rows are produced by feeding the correction functions a window
view whose ``c_b2`` field holds the MLR prediction, so the correction code
path is byte-identical for both families.  Comparing the families answers
whether antigenicity/Fitness corrections transfer to a different frequency
model, i.e. whether the mechanistic evidence lines are robust.

Outputs (written to --output-dir, default ``outputs/h1n1_c_b2_correction_with_mlr``):
    rolling_metrics.csv        per-cutoff metrics for all eight models
    rolling_summary.csv        model-level means, sorted by mean_TVD
    c_b2_all_nine_windows.csv  both uncorrected bases on all nine windows
    rolling_coefficients.csv   correction coefficients per model family
    summary.json               run report

Requires: ``pip install evofr`` and ``forecasts_flu_mlr.py``.

Example::

    python src/forecasting/subclade/c_b2_correction_with_mlr.py \
      --scores-sqlite outputs/h1n1_antigenicity/past_strain_3y_full_excluding_recent6m.sqlite \
      --cutoffs-csv outputs/h1n1_antigenicity/panels/forecast_cutoffs.csv \
      --output-dir outputs/h1n1_c_b2_correction_with_mlr --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Brother's framework, imported READ-ONLY (his files are not modified).
from src.forecasting.subclade.c_b2_biological_correction import (
    ForecastWindow,
    evaluate,
    fit_correction,
    apply_correction,
    summarize,
)
from src.forecasting.subclade.joint_biology_forecast import build_features, load_scored_sequences
from src.forecasting.subclade.benchmarks import (
    build_window_aligned_counts,
    run_six_model_backtest,
)

MLR_MODEL_NAME = "Nextstrain MLR (B1-official)"
MLR_BIN_DAYS = 14  # the fixed bin width of this backtest

# Two base trajectories x four feature corrections.  The base is either the
# C-B2 proxy prediction (the original framework) or the official Nextstrain
# MLR prediction; the correction machinery multiplies the base by
# exp(t * coefficient * feature), so swapping the base turns the "C-B2 + ..."
# family into an "MLR + ..." family under an identical protocol.
CORRECTION_SUFFIXES = {
    "": [],
    " + relative antigenicity": ["escape_excess"],
    " + relative Fitness": ["fitness_excess"],
    " + relative fitness": ["escape_excess", "fitness_excess"],
}
BASE_NAMES = ("C-B2", MLR_MODEL_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-sqlite", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/antigenicity_scores.sqlite")
    parser.add_argument("--query-instances-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/input/query_instances.csv")
    parser.add_argument("--cutoffs-csv", type=Path, default=PROJECT_ROOT / "outputs/h1n1_antigenicity/panels/forecast_cutoffs.csv")
    parser.add_argument("--nextclade-tsv", type=Path, default=PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv")
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--min-score-sequences", type=int, default=10)
    parser.add_argument("--min-train-cutoffs", type=int, default=3)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--objective", choices=["top1", "distribution"], default="top1")
    # --- MLR options ---
    parser.add_argument("--mlr-lookback-bins", type=int, default=26, help="History bins given to the MLR (default 26, matching the C-B2 proxies).")
    parser.add_argument("--mlr-gen-days", type=float, default=2.7, help="H1N1pdm generation time in days (forecasts-flu config).")
    parser.add_argument("--mlr-inference", choices=["nuts_from_map", "nuts", "map"], default="nuts_from_map",
                        help="MLR inference backend; nuts_from_map mirrors the official forecasts-flu settings.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/h1n1_c_b2_correction_with_mlr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compute_mlr_forecast(
    counts: pd.DataFrame, cutoff_time: pd.Timestamp, horizon_bins: int, args: argparse.Namespace
) -> pd.DataFrame:
    """Fit the official forecasts-flu MLR on the same window-aligned counts."""

    from src.forecasting.subclade.forecasts_flu_mlr import fit_mlr_forecast, slice_backtest_tables

    history, full_horizon_index = slice_backtest_tables(
        counts, cutoff_time, horizon_bins, args.mlr_lookback_bins
    )
    fit = fit_mlr_forecast(
        history,
        bin_days=MLR_BIN_DAYS,
        horizon_dates=full_horizon_index,
        gen_days=args.mlr_gen_days,
        inference=args.mlr_inference,
    )
    return fit["prediction"]


@dataclass
class WindowWithMlr(ForecastWindow):
    mlr: pd.DataFrame | None = None


def window_forecasts(
    records: pd.DataFrame,
    clade_features: pd.DataFrame,
    cutoffs: pd.DataFrame,
    args: argparse.Namespace,
) -> list[WindowWithMlr]:
    feature_lookup = {
        cutoff: frame.set_index("branch")[["escape_excess", "fitness_excess"]]
        for cutoff, frame in clade_features.groupby("forecast_cutoff", sort=False)
    }
    forecasts = []
    window_rows = cutoffs.sort_values("target_start", kind="stable")
    for cutoff_row in window_rows.itertuples(index=False):
        start = cutoff_row.target_start
        end = cutoff_row.target_end
        counts, _, horizon = build_window_aligned_counts(records, window_start=start, window_end=end)
        cutoff_time = pd.Timestamp(start) - pd.Timedelta(days=1)
        frequencies, _, _, actual_frequencies, _ = run_six_model_backtest(
            counts, cutoff=cutoff_time, horizon_bins=horizon
        )
        cutoff = cutoff_time.date().isoformat()
        if cutoff not in feature_lookup:
            continue
        actual = actual_frequencies
        c_b2 = frequencies["C-B2"].reindex(index=actual.index, columns=actual.columns, fill_value=0.0)
        features = feature_lookup[cutoff].reindex(actual.columns).fillna(0.0)
        if args.trajectory_mode == "relative_fitness":
            current = counts.loc[counts.index <= cutoff_time].iloc[-1].reindex(actual.columns, fill_value=0.0)
            current = current / current.sum()
            features = features - current @ features
        mlr_prediction = compute_mlr_forecast(counts, cutoff_time, horizon, args)
        mlr_prediction = mlr_prediction.reindex(index=actual.index, columns=actual.columns, fill_value=0.0)
        forecasts.append(WindowWithMlr(cutoff, actual, c_b2, features, mlr_prediction))
    if not forecasts:
        raise RuntimeError("No C-B2 windows could be aligned with antigenicity/Fitness features.")
    return forecasts


def base_view(window: WindowWithMlr, base_name: str) -> ForecastWindow:
    """Return a window whose ``c_b2`` field carries the chosen base trajectory.

    The correction functions hardcode ``window.c_b2`` as the trajectory being
    corrected, so an MLR-based model is expressed by feeding a view whose
    ``c_b2`` holds the MLR prediction; actuals and features are unchanged.
    """

    if base_name == "C-B2":
        return window
    if window.mlr is None:
        raise ValueError(f"window {window.cutoff} has no MLR prediction attached")
    return ForecastWindow(
        cutoff=window.cutoff, actual=window.actual, c_b2=window.mlr, features=window.features
    )


def rolling_test(
    windows: list[WindowWithMlr], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details, coefficients = [], []
    for index, test in enumerate(windows):
        train = windows[:index]
        if len(train) < args.min_train_cutoffs:
            continue
        for base_name in BASE_NAMES:
            if base_name != "C-B2" and test.mlr is None:
                continue
            test_view = base_view(test, base_name)
            train_views = [base_view(w, base_name) for w in train]
            for suffix, feature_columns in CORRECTION_SUFFIXES.items():
                model = base_name + suffix
                weights, mean, scale = fit_correction(
                    train_views, feature_columns, args.l2, args.objective,
                    args.trajectory_mode, "frequency",
                )
                prediction = apply_correction(
                    test_view, feature_columns, weights, mean, scale,
                    args.trajectory_mode, "frequency",
                )
                details.append(evaluate(test_view, model, prediction, "frequency"))
                coefficients.extend(
                    {"forecast_cutoff": test.cutoff, "model": model, "feature": feature, "coefficient": weight}
                    for feature, weight in zip(feature_columns, weights, strict=True)
                )
    if not details:
        raise RuntimeError("No test window remains; lower --min-train-cutoffs.")
    return pd.DataFrame(details), pd.DataFrame(coefficients)


def main() -> None:
    args = parse_args()
    if args.min_score_sequences < 1 or args.min_train_cutoffs < 1 or args.l2 < 0:
        raise ValueError("Minimum counts must be positive and --l2 must be non-negative.")
    args.trajectory_mode = "relative_fitness"  # fixed, matching the demo notebook
    args.prediction_target = "frequency"

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

    scored, records = load_scored_sequences(args)
    cutoffs = pd.read_csv(args.cutoffs_csv)
    features = build_features(scored, records, cutoffs, args.min_score_sequences)
    windows = window_forecasts(records, features, cutoffs, args)
    detail, coefficients = rolling_test(windows, args)
    summary = summarize(detail, "frequency")
    full_baseline_rows = [evaluate(window, "C-B2", window.c_b2, "frequency") for window in windows]
    full_baseline_rows.extend(
        evaluate(window, MLR_MODEL_NAME, window.mlr, "frequency")
        for window in windows if window.mlr is not None
    )
    full_baseline = pd.DataFrame(full_baseline_rows)

    detail.to_csv(outputs["detail"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    full_baseline.to_csv(outputs["full_baseline"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    report = {
        "test_framework": "two bases (C-B2 proxy, Nextstrain MLR) x four feature corrections, 14-day rolling holdouts",
        "training_policy": "each correction uses only earlier holdouts of its own base family; MLR is trained per window on counts only",
        "prediction_target": "frequency",
        "correction_objective": args.objective,
        "trajectory_mode": args.trajectory_mode,
        "relative_fitness_rule": "C-B2(t, clade) × exp(t × relative_fitness(clade))",
        "mlr": {
            "model": MLR_MODEL_NAME,
            "lookback_bins": args.mlr_lookback_bins,
            "generation_days": args.mlr_gen_days,
            "inference": args.mlr_inference,
        },
        "c_b2_all_nine_windows": summarize(full_baseline, "frequency").to_dict(orient="records")[0],
        "tested_cutoffs": sorted(detail["forecast_cutoff"].unique()),
        "models": summary.to_dict(orient="records"),
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
