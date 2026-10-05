"""Run the official Nextstrain forecasts-flu MLR (benchmark B1) on our data pipeline.

This script wires the Nextclade strain-data adapter into
evofr's ``MultinomialLogisticRegression`` -- the exact model class the official
forecasts-flu workflow (scripts/run-model.py) fits -- and scores it under the
same backtest protocol as ``benchmarks`` (same 14-day bins, cutoff,
horizon, lookback and metrics), so it is directly comparable with the
B0/B1-proxy/B2 rows produced by ``run_six_model_backtest``.

Protocol notes
--------------
* Counts are aggregated with ``build_binned_tables`` (14-day bins anchored at
  the pipeline start date); the model consumes the same bins.
* Model time is indexed in bins, so the generation time is converted to bin
  units exactly like the official config does for its weekly bins
  (``generation_time: 0.39`` = 2.7 days / 7-day bin for H1N1pdm, Lessler et
  al., via forecasts-flu config/mlr/h1n1pdm.yaml).
* Inference mirrors the official settings: MAP initialisation via SVI followed
  by NUTS with dense mass (``InferNUTS_from_MAP``).
* The pivot (reference branch, GA denominator) is the most frequent branch in
  the last ``--pivot-window`` bins; frequency forecasts are pivot-invariant.

Examples
--------
Single cutoff (same protocol as the notebook's six-model backtest)::

    python src/forecasting/subclade/forecasts_flu_mlr.py

Rolling backtest over the 9 analysis windows::

    python src/forecasting/subclade/forecasts_flu_mlr.py --rolling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evofr as ef
from evofr.infer.InferMCMC import InferNUTS_from_MAP

from src.forecasting.subclade.benchmarks import (
    DEFAULT_HALF_YEAR_WINDOWS,
    SIX_MODEL_LABELS,
    build_window_aligned_counts,
    evaluate_frequency_forecast,
    run_six_model_backtest,
)
from src.forecasting.run_metadata import load_json_config, write_run_manifest
from src.strain_data.nextclade import (
    DEFAULT_START_DATE,
    build_binned_tables,
    read_nextclade_table,
)

DEFAULT_INPUT_TSV = (
    PROJECT_ROOT / "data/raw/nextclade/H1N1/h1n1_results_MW626062_sorted.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/h1n1_demo/forecasts_flu_mlr"
MODEL_LABEL = "B1-official forecasts-flu MLR"
OFFICIAL_MODEL_ID = "B1-official"

# forecasts-flu h1n1pdm.yaml: 2.7 days (Lessler et al.) divided by that
# workflow's 7-day aggregation.  We keep the day figure and rescale per bin.
H1N1_GENERATION_DAYS = 2.7

EXCLUDE_FROM_PIVOT = {"Unassigned"}
METRIC_COLUMNS = ["MAE", "RMSE", "mean_TVD", "mean_JSD", "top1_accuracy"]


def collapse_rare_branches(
    active: pd.DataFrame, min_seqs: float
) -> pd.DataFrame:
    """Merge branches with fewer than ``min_seqs`` sequences into ``other``.

    Mirrors the official forecasts-flu step (collapse_haplotype_counts +
    prepare-data ``clade_min_seq``) so low-count branches cannot inject noise
    through their own softmax logits.
    """

    totals = active.sum(axis=0)
    small = totals < min_seqs
    if not small.any() or int((~small).sum()) < 2:
        return active
    collapsed = active.loc[:, ~small].copy()
    collapsed["other"] = active.loc[:, small].sum(axis=1)
    return collapsed


def fit_mlr_forecast(
    history_counts: pd.DataFrame,
    *,
    bin_days: int,
    horizon_dates: pd.Index,
    gen_days: float = H1N1_GENERATION_DAYS,
    inference: str = "nuts_from_map",
    num_warmup: int = 200,
    num_samples: int = 200,
    svi_iters: int = 50_000,
    svi_lr: float = 4e-4,
    pivot_window: int = 3,
    collapse_min_seqs: int = 0,
) -> dict:
    """Fit the forecasts-flu MLR on binned counts and forecast ahead.

    ``history_counts`` is a bin-by-branch count table ending at the cutoff bin.
    ``horizon_dates`` are the bin dates to predict, in order.  Returns mean and
    quantile frequency forecasts plus growth advantages relative to the pivot.
    """

    history = history_counts.copy()
    history.index.name = "bin_start"
    history.columns.name = "variant"

    # Branches with no sequences inside the modelling window carry no
    # likelihood information but would still receive prior-noise mass through
    # the softmax, so they are dropped (forecasts-flu collapses them instead).
    active = history.loc[:, history.sum(axis=0) > 0]
    if collapse_min_seqs > 0:
        active = collapse_rare_branches(active, collapse_min_seqs)
    if active.shape[1] < 2:
        raise ValueError("fewer than two active branches in the lookback window")

    tail_sums = active.tail(pivot_window).sum(axis=0)
    candidates = tail_sums.drop(
        index=[c for c in EXCLUDE_FROM_PIVOT if c in tail_sums.index]
    )
    pivot = str(candidates.idxmax())

    raw_seq = (
        active.rename_axis(index="bin_start", columns="variant")
        .stack()
        .rename("sequences")
        .reset_index()
        .rename(columns={"bin_start": "date"})
    )
    raw_seq["sequences"] = raw_seq["sequences"].astype(int)

    tau = gen_days / bin_days  # generation time in units of model bins
    model = ef.MultinomialLogisticRegression(tau=tau)
    data = ef.VariantFrequencies(
        raw_seq, pivot=pivot, aggregation_frequency=f"{bin_days}D"
    )
    if data.seq_counts.shape != active.shape:
        raise ValueError(
            f"aggregation mismatch: evofr built {data.seq_counts.shape}, "
            f"expected {active.shape}"
        )
    if int(data.seq_counts.sum()) != int(active.to_numpy().sum()):
        raise ValueError("sequence counts were not preserved by aggregation")

    if inference == "nuts_from_map":
        inferer = InferNUTS_from_MAP(
            num_warmup=num_warmup,
            num_samples=num_samples,
            iters=svi_iters,
            lr=svi_lr,
        )
    elif inference == "nuts":
        inferer = ef.InferNUTS(
            num_warmup=num_warmup, num_samples=num_samples, dense_mass=True
        )
    elif inference == "map":
        inferer = ef.InferMAP(iters=svi_iters, lr=svi_lr)
    else:
        raise ValueError(f"unknown inference method: {inference}")

    posterior = inferer.fit(model, data, name=OFFICIAL_MODEL_ID)
    samples = posterior.samples
    model.forecast_frequencies(samples, forecast_L=len(horizon_dates))

    freq_forecast = np.asarray(samples["freq_forecast"])  # (S, L, K)
    freq_mean = freq_forecast.mean(axis=0)
    freq_lo = np.quantile(freq_forecast, 0.025, axis=0)
    freq_hi = np.quantile(freq_forecast, 0.975, axis=0)

    ga_ratio = np.asarray(samples["ga"])  # (S, K-1), relative to the pivot
    ga_pct = ga_ratio - 1.0
    ga_table = pd.DataFrame(
        {
            "variant": [v for v in data.var_names if v != pivot],
            "ga_pct_mean": ga_pct.mean(axis=0),
            "ga_pct_q025": np.quantile(ga_pct, 0.025, axis=0),
            "ga_pct_q975": np.quantile(ga_pct, 0.975, axis=0),
        }
    ).sort_values("ga_pct_mean", ascending=False, ignore_index=True)

    prediction = pd.DataFrame(
        freq_mean,
        columns=list(data.var_names),
        index=pd.Index(horizon_dates, name="bin_start"),
    )
    prediction_lower = pd.DataFrame(
        freq_lo, columns=prediction.columns, index=prediction.index
    )
    prediction_upper = pd.DataFrame(
        freq_hi, columns=prediction.columns, index=prediction.index
    )

    return {
        "pivot": pivot,
        "var_names": list(data.var_names),
        "prediction": prediction,
        "prediction_lower": prediction_lower,
        "prediction_upper": prediction_upper,
        "ga_table": ga_table,
        "tau": tau,
        "n_posterior_samples": int(freq_forecast.shape[0]),
    }


def evaluate_with_top1(
    actual_frequencies: pd.DataFrame,
    prediction: pd.DataFrame,
) -> pd.Series:
    """Score a forecast with the benchmark metrics plus top-1 accuracy."""

    metrics = evaluate_frequency_forecast(actual_frequencies, prediction)
    metrics["top1_accuracy"] = (
        prediction.idxmax(axis=1) == actual_frequencies.idxmax(axis=1)
    ).mean()
    return metrics


def merge_metrics(
    six_metrics: pd.DataFrame,
    six_predictions: dict[str, pd.DataFrame],
    actual_frequencies: pd.DataFrame,
    mlr_metrics: pd.Series,
) -> pd.DataFrame:
    """Rebuild the six-model table with top-1 and append the official MLR row.

    ``run_six_model_backtest`` only reports top-1 accuracy in its rolling
    variant, so it is recomputed here for every model on the same holdout.
    """

    rows: dict[str, pd.Series] = {}
    for model_id, prediction in six_predictions.items():
        rows[SIX_MODEL_LABELS[model_id]] = evaluate_with_top1(
            actual_frequencies, prediction
        )
    rows[MODEL_LABEL] = mlr_metrics
    metrics = pd.DataFrame(rows).T.rename_axis("model")
    metrics.insert(
        0,
        "training_input",
        [
            "frequency" if str(label).startswith("Frequency") else "counts"
            for label in metrics.index
        ],
    )
    return metrics.sort_values("mean_TVD")


def slice_backtest_tables(
    counts: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_bins: int,
    lookback_bins: int,
):
    """Slice one count table into (history, full-horizon index) at a cutoff."""

    index = counts.index
    eligible = np.flatnonzero(index <= cutoff)
    if not len(eligible):
        raise ValueError("cutoff precedes the first bin")
    cut_pos = int(eligible[-1])
    full_horizon_index = index[cut_pos + 1 : cut_pos + 1 + horizon_bins]
    if len(full_horizon_index) < horizon_bins:
        raise ValueError("not enough bins after the cutoff for the horizon")
    history = counts.iloc[max(0, cut_pos + 1 - lookback_bins) : cut_pos + 1]
    return history, full_horizon_index


def plot_comparison(
    proportions: pd.DataFrame,
    actual_frequencies: pd.DataFrame,
    proxy_predictions: dict[str, pd.DataFrame],
    official_predictions: dict[str, pd.DataFrame],
    effective_cutoff: pd.Timestamp,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    top_branches = actual_frequencies.mean().nlargest(6).index.tolist()
    history = proportions.loc[:effective_cutoff].tail(8)
    styles = {
        "F-B1": ("#1f77b4", "--"),
        OFFICIAL_MODEL_ID: ("#000000", "-"),
    }
    labels = {
        "F-B1": "F-B1 trend proxy",
        OFFICIAL_MODEL_ID: "B1 official MLR",
    }

    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, constrained_layout=True)
    for axis, branch in zip(axes.flat, top_branches):
        axis.plot(
            history.index, history[branch], color="black", alpha=0.25, marker="o", label="History"
        )
        axis.plot(
            actual_frequencies.index,
            actual_frequencies[branch],
            color="black",
            linewidth=2.8,
            marker="o",
            label="Actual",
        )
        for model_name, prediction in {**proxy_predictions, **official_predictions}.items():
            if model_name not in styles:
                continue  # keep panels readable: history, actual, proxy, official
            color, linestyle = styles[model_name]
            axis.plot(
                prediction.index,
                prediction[branch],
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
                label=labels.get(model_name, model_name),
            )
        axis.axvline(effective_cutoff, color="black", linestyle=":", alpha=0.6)
        axis.set_title(branch, fontweight="bold")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=45)

    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels_, loc="outside lower center", ncol=8, frameon=False)
    figure.suptitle(
        "H1N1 forecast: official forecasts-flu MLR vs B-series baselines",
        fontweight="bold",
        fontsize=15,
    )
    figure.supylabel("Nextclade subclade proportion")
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_single_cutoff(args: argparse.Namespace) -> None:
    records = read_nextclade_table(args.input_tsv, label_mode=args.label_mode)
    counts, proportions, _ = build_binned_tables(
        records,
        bin_days=args.bin_days,
        start_date=args.start_date,
    )

    six_predictions, _, actual_counts, actual_frequencies, six_metrics = (
        run_six_model_backtest(
            counts,
            cutoff=args.cutoff,
            horizon_bins=args.horizon_bins,
            lookback_bins=args.lookback_bins,
        )
    )
    effective_cutoff = counts.index[counts.index <= pd.Timestamp(args.cutoff)][-1]
    history, full_horizon_index = slice_backtest_tables(
        counts, pd.Timestamp(args.cutoff), args.horizon_bins, args.lookback_bins
    )

    print(
        f"Effective cutoff: {effective_cutoff.date()} | lookback bins: {len(history)}"
        f" | horizon bins: {len(full_horizon_index)}"
    )
    print("Fitting official forecasts-flu MLR (SVI MAP init + NUTS)...")
    started = time.perf_counter()
    fit = fit_mlr_forecast(
        history,
        bin_days=args.bin_days,
        horizon_dates=full_horizon_index,
        gen_days=args.gen_days,
        inference=args.inference,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        svi_iters=args.svi_iters,
        svi_lr=args.svi_lr,
        pivot_window=args.pivot_window,
        collapse_min_seqs=args.collapse_min_seqs,
    )
    print(f"Fit completed in {time.perf_counter() - started:.1f}s; pivot = {fit['pivot']}")

    prediction = fit["prediction"].reindex(
        index=full_horizon_index, columns=fit["prediction"].columns, fill_value=0.0
    )
    mlr_metrics = evaluate_with_top1(actual_frequencies, prediction)
    metrics = merge_metrics(six_metrics, six_predictions, actual_frequencies, mlr_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "b1_official_metrics.csv")
    prediction.to_csv(args.output_dir / "b1_official_frequency_predictions.csv")
    fit["ga_table"].to_csv(args.output_dir / "b1_official_ga.csv", index=False)

    meta = {
        "input_tsv": str(args.input_tsv),
        "label_mode": args.label_mode,
        "bin_days": args.bin_days,
        "start_date": str(args.start_date),
        "cutoff": str(pd.Timestamp(args.cutoff).date()),
        "effective_cutoff": str(effective_cutoff.date()),
        "horizon_bins": args.horizon_bins,
        "lookback_bins": args.lookback_bins,
        "gen_days": args.gen_days,
        "tau_per_bin": fit["tau"],
        "inference": args.inference,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "svi_iters": args.svi_iters,
        "svi_lr": args.svi_lr,
        "pivot": fit["pivot"],
        "modelled_branches": fit["var_names"],
        "collapse_min_seqs": args.collapse_min_seqs,
        "n_posterior_samples": fit["n_posterior_samples"],
    }
    (args.output_dir / "b1_official_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_comparison(
        proportions,
        actual_frequencies,
        six_predictions,
        {OFFICIAL_MODEL_ID: prediction},
        effective_cutoff,
        args.output_dir / "b1_official_plot.png",
    )
    write_run_manifest(
        args.output_dir,
        mode="subclade_mlr_single_cutoff",
        arguments=vars(args),
        input_files={"nextclade_tsv": args.input_tsv},
        config_path=args.config,
    )

    print()
    print(metrics.to_string())
    print()
    print("Growth advantage relative to pivot (per generation):")
    print(fit["ga_table"].head(10).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nOutputs saved to: {args.output_dir}")


def run_rolling(args: argparse.Namespace) -> None:
    records = read_nextclade_table(args.input_tsv, label_mode=args.label_mode)
    windows = DEFAULT_HALF_YEAR_WINDOWS[:-3]  # exclude COVID-era windows
    label_to_id = {
        **{label: name for name, label in SIX_MODEL_LABELS.items()},
        MODEL_LABEL: OFFICIAL_MODEL_ID,
    }
    detail_rows: list[dict[str, object]] = []

    for window_number, (window_start, window_end) in enumerate(windows, start=1):
        start = pd.Timestamp(window_start).normalize()
        aligned_counts, evaluation_end, window_horizon = build_window_aligned_counts(
            records,
            window_start=start,
            window_end=pd.Timestamp(window_end).normalize(),
            train_start=pd.Timestamp(args.start_date).normalize(),
            bin_days=args.bin_days,
        )
        horizon_bins = window_horizon
        if args.max_horizon_bins is not None:
            horizon_bins = min(horizon_bins, args.max_horizon_bins)
        cutoff = start - pd.Timedelta(days=1)
        six_predictions, _, actual_counts, actual_frequencies, six_metrics = (
            run_six_model_backtest(
                aligned_counts,
                cutoff=cutoff,
                horizon_bins=horizon_bins,
                lookback_bins=args.lookback_bins,
            )
        )
        history, full_horizon_index = slice_backtest_tables(
            aligned_counts, cutoff, horizon_bins, args.lookback_bins
        )

        print(
            f"[window {window_number}/{len(windows)}] {start.date()}..{evaluation_end.date()}"
            f" | lookback {len(history)} bins, horizon {horizon_bins} bins"
        )
        try:
            fit = fit_mlr_forecast(
                history,
                bin_days=args.bin_days,
                horizon_dates=full_horizon_index,
                gen_days=args.gen_days,
                inference=args.inference,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                svi_iters=args.svi_iters,
                svi_lr=args.svi_lr,
                pivot_window=args.pivot_window,
                collapse_min_seqs=args.collapse_min_seqs,
            )
        except Exception as error:  # noqa: BLE001 - keep the sweep alive
            print(f"  MLR fit failed: {error}")
            continue

        prediction = fit["prediction"].reindex(
            index=full_horizon_index, columns=fit["prediction"].columns, fill_value=0.0
        )
        mlr_metrics = evaluate_with_top1(actual_frequencies, prediction)
        metrics = merge_metrics(six_metrics, six_predictions, actual_frequencies, mlr_metrics)

        for model_label, metric_values in metrics.iterrows():
            detail_rows.append(
                {
                    "window_number": window_number,
                    "window_start": start,
                    "window_end": evaluation_end,
                    "horizon_bins": horizon_bins,
                    "holdout_sequences": int(actual_counts.to_numpy().sum()),
                    "model_id": label_to_id.get(model_label, model_label),
                    "model": model_label,
                    "training_input": metric_values["training_input"],
                    **{name: metric_values[name] for name in METRIC_COLUMNS},
                }
            )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["model_id", "model", "training_input"], sort=False)
        .agg(
            windows=("window_number", "nunique"),
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_TVD=("mean_TVD", "mean"),
            std_TVD=("mean_TVD", "std"),
            mean_JSD=("mean_JSD", "mean"),
            mean_top1_accuracy=("top1_accuracy", "mean"),
        )
        .reset_index()
    )
    winners = detail.loc[
        detail.groupby("window_number")["mean_TVD"].transform("min") == detail["mean_TVD"]
    ]["model_id"].value_counts()
    summary["window_wins"] = summary["model_id"].map(winners).fillna(0).astype(int)
    summary = summary.sort_values("mean_TVD", ignore_index=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "rolling_b1_official_detail.csv", index=False)
    summary.to_csv(args.output_dir / "rolling_b1_official_summary.csv", index=False)
    write_run_manifest(
        args.output_dir,
        mode="subclade_mlr_rolling_backtest",
        arguments=vars(args),
        input_files={"nextclade_tsv": args.input_tsv},
        config_path=args.config,
    )
    print()
    print(summary.to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON experiment config; command-line values override its fields",
    )
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--bin-days", type=int, default=14)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--cutoff", default="2024-12-31")
    parser.add_argument("--horizon-bins", type=int, default=12)
    parser.add_argument("--lookback-bins", type=int, default=26)
    parser.add_argument("--gen-days", type=float, default=H1N1_GENERATION_DAYS)
    parser.add_argument(
        "--inference",
        choices=["nuts_from_map", "nuts", "map"],
        default="nuts_from_map",
    )
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--svi-iters", type=int, default=50_000)
    parser.add_argument("--svi-lr", type=float, default=4e-4)
    parser.add_argument("--pivot-window", type=int, default=3)
    parser.add_argument(
        "--collapse-min-seqs",
        type=int,
        default=0,
        help="merge branches with fewer than this many lookback sequences into 'other' (official forecasts-flu uses 30)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="run the 9-window rolling backtest instead of the single cutoff",
    )
    parser.add_argument(
        "--max-horizon-bins",
        type=int,
        default=None,
        help="rolling mode only: truncate each window's horizon to this many bins",
    )
    return parser


def main() -> None:
    parser = build_parser()
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    config_args, _ = bootstrap.parse_known_args()
    if config_args.config is not None:
        allowed_keys = {action.dest for action in parser._actions}
        parser.set_defaults(
            **load_json_config(config_args.config, allowed_keys=allowed_keys)
        )
    args = parser.parse_args()
    if args.rolling:
        run_rolling(args)
    else:
        run_single_cutoff(args)


if __name__ == "__main__":
    main()
