"""Frequency- and count-trained B-series baselines for H1N1 backtesting.

B0 is the persistence baseline.  B1 and B2 are local, dependency-free proxy
implementations for notebook iteration; they are not drop-in replacements for
the official Nextstrain forecasts-flu or RelRe command-line pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "B0": "B0 persistence",
    "B1": "B1 frequency-trend regression (local proxy)",
    "B2": "B2 renewal-selection (local proxy)",
}

SIX_MODEL_LABELS = {
    "F-B0": "Frequency · B0 persistence",
    "F-B1": "Frequency · B1 trend proxy",
    "F-B2": "Frequency · B2 renewal proxy",
    "C-B0": "Counts · B0 persistence",
    "C-B1": "Counts · B1 trend proxy",
    "C-B2": "Counts · B2 renewal proxy",
}

DEFAULT_HALF_YEAR_WINDOWS = (
    ("2025-10-01", "2026-03-31"),
    ("2025-04-01", "2025-09-30"),
    ("2024-10-01", "2025-03-31"),
    ("2024-04-01", "2024-09-30"),
    ("2023-10-01", "2024-03-31"),
    ("2023-04-01", "2023-09-30"),
    ("2022-10-01", "2023-03-31"),
    ("2022-04-01", "2022-09-30"),
    ("2021-10-01", "2022-03-31"),
    ("2021-04-01", "2021-09-30"),
    ("2020-10-01", "2021-03-31"),
    ("2020-04-01", "2020-09-30"),
)


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        raise ValueError("history is empty")
    data = history.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if (data < 0).any().any():
        raise ValueError("history contains negative frequencies")
    row_sums = data.sum(axis=1)
    if not (row_sums > 0).any():
        raise ValueError("history contains no non-empty frequency bins")
    return data.loc[row_sums > 0]


def _normalize_rows(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0
    if empty.any():
        values[empty] = fallback
        totals = values.sum(axis=1, keepdims=True)
    return values / totals


def counts_to_frequencies(counts: pd.DataFrame) -> pd.DataFrame:
    """Normalize each non-empty count row into a frequency distribution."""

    data = _validate_history(counts)
    totals = data.sum(axis=1)
    return data.div(totals, axis=0)


def _forecast_total_counts(
    history_counts: pd.DataFrame,
    horizon: int,
    *,
    total_lookback_bins: int = 4,
) -> np.ndarray:
    if total_lookback_bins <= 0:
        raise ValueError("total_lookback_bins must be positive")
    totals = history_counts.sum(axis=1).tail(total_lookback_bins)
    level = max(float(totals.median()), 1.0)
    return np.repeat(level, horizon)


def _probabilities_to_expected_counts(
    probabilities: np.ndarray,
    history_counts: pd.DataFrame,
    *,
    total_lookback_bins: int = 4,
) -> np.ndarray:
    future_totals = _forecast_total_counts(
        history_counts,
        len(probabilities),
        total_lookback_bins=total_lookback_bins,
    )
    return probabilities * future_totals[:, None]


def forecast_b0_persistence(
    history: pd.DataFrame,
    future_index: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Repeat the most recent observed frequency vector at every horizon."""

    data = _validate_history(history)
    future_index = pd.Index(future_index, name=history.index.name)
    anchor = data.iloc[-1].to_numpy(dtype=float)
    anchor /= anchor.sum()
    values = np.repeat(anchor[None, :], len(future_index), axis=0)
    return pd.DataFrame(values, index=future_index, columns=data.columns)


def forecast_b1_frequency_trend(
    history: pd.DataFrame,
    future_index: Iterable[pd.Timestamp],
    *,
    lookback_bins: int = 26,
) -> pd.DataFrame:
    """Extrapolate a linear trend for each branch, then renormalize.

    This is a frequency-only local proxy for B1.  The official Nextstrain MLR
    uses additional fitness predictors and should be run separately for a
    publication-grade comparison.
    """

    if lookback_bins < 2:
        raise ValueError("lookback_bins must be at least 2")
    data = _validate_history(history).tail(lookback_bins)
    future_index = pd.Index(future_index, name=history.index.name)
    x = np.arange(len(data), dtype=float)
    future_x = len(data) + np.arange(len(future_index), dtype=float)
    weights = np.linspace(0.5, 1.0, len(data))
    values = np.zeros((len(future_index), len(data.columns)), dtype=float)

    for column_number, column in enumerate(data.columns):
        y = data[column].to_numpy(dtype=float)
        if np.allclose(y, 0.0):
            continue
        slope, intercept = np.polyfit(x, y, deg=1, w=weights)
        values[:, column_number] = intercept + slope * future_x

    anchor = data.iloc[-1].to_numpy(dtype=float)
    anchor /= anchor.sum()
    values = _normalize_rows(values, fallback=anchor)
    return pd.DataFrame(values, index=future_index, columns=data.columns)


def forecast_b2_renewal_selection(
    history: pd.DataFrame,
    future_index: Iterable[pd.Timestamp],
    *,
    lookback_bins: int = 26,
    pseudocount: float = 1e-6,
    max_log_growth_per_bin: float = 0.35,
) -> pd.DataFrame:
    """Project frequencies with robust relative-growth selection coefficients.

    Median changes in centered log frequency estimate each branch's relative
    growth.  Future shares follow the discrete renewal/selection update
    ``p_i(t+h) proportional to p_i(t) * exp(s_i * h)``.
    """

    if lookback_bins < 3:
        raise ValueError("lookback_bins must be at least 3")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    if max_log_growth_per_bin <= 0:
        raise ValueError("max_log_growth_per_bin must be positive")

    data = _validate_history(history).tail(lookback_bins)
    future_index = pd.Index(future_index, name=history.index.name)
    frequency = data.to_numpy(dtype=float)
    active = frequency.sum(axis=0) > 0
    active_frequency = frequency[:, active]
    log_frequency = np.log(active_frequency + pseudocount)
    centered_log_frequency = log_frequency - log_frequency.mean(axis=1, keepdims=True)
    growth = np.median(np.diff(centered_log_frequency, axis=0), axis=0)
    growth = np.clip(growth, -max_log_growth_per_bin, max_log_growth_per_bin)

    anchor = active_frequency[-1]
    anchor /= anchor.sum()
    horizons = np.arange(1, len(future_index) + 1, dtype=float)[:, None]
    logits = np.log(anchor + pseudocount)[None, :] + horizons * growth[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    active_values = np.exp(logits)
    active_values /= active_values.sum(axis=1, keepdims=True)
    values = np.zeros((len(future_index), len(data.columns)), dtype=float)
    values[:, active] = active_values
    return pd.DataFrame(values, index=future_index, columns=data.columns)


def forecast_count_b0_persistence(
    history_counts: pd.DataFrame,
    future_index: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Repeat the latest observed count vector at every future horizon."""

    data = _validate_history(history_counts)
    future_index = pd.Index(future_index, name=history_counts.index.name)
    anchor = data.iloc[-1].to_numpy(dtype=float)
    values = np.repeat(anchor[None, :], len(future_index), axis=0)
    return pd.DataFrame(values, index=future_index, columns=data.columns)


def forecast_count_b1_trend(
    history_counts: pd.DataFrame,
    future_index: Iterable[pd.Timestamp],
    *,
    lookback_bins: int = 26,
    pseudocount: float = 0.5,
    total_lookback_bins: int = 4,
) -> pd.DataFrame:
    """Fit count-weighted log-composition trends and predict expected counts."""

    if lookback_bins < 2:
        raise ValueError("lookback_bins must be at least 2")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    data = _validate_history(history_counts).tail(lookback_bins)
    future_index = pd.Index(future_index, name=history_counts.index.name)
    counts = data.to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    active = counts.sum(axis=0) > 0
    active_counts = counts[:, active]
    probabilities = (active_counts + pseudocount) / (
        totals[:, None] + pseudocount * active.sum()
    )
    log_probabilities = np.log(probabilities)
    log_probabilities -= log_probabilities.mean(axis=1, keepdims=True)

    x = np.arange(len(data), dtype=float)
    future_x = len(data) + np.arange(len(future_index), dtype=float)
    sampling_weights = np.sqrt(totals / np.median(totals))
    future_logits = np.zeros((len(future_index), active.sum()), dtype=float)
    for column_number in range(active.sum()):
        slope, intercept = np.polyfit(
            x,
            log_probabilities[:, column_number],
            deg=1,
            w=sampling_weights,
        )
        future_logits[:, column_number] = intercept + slope * future_x

    future_logits -= future_logits.max(axis=1, keepdims=True)
    active_probabilities = np.exp(future_logits)
    active_probabilities /= active_probabilities.sum(axis=1, keepdims=True)
    all_probabilities = np.zeros((len(future_index), len(data.columns)), dtype=float)
    all_probabilities[:, active] = active_probabilities
    expected_counts = _probabilities_to_expected_counts(
        all_probabilities,
        data,
        total_lookback_bins=total_lookback_bins,
    )
    return pd.DataFrame(expected_counts, index=future_index, columns=data.columns)


def forecast_count_b2_renewal(
    history_counts: pd.DataFrame,
    future_index: Iterable[pd.Timestamp],
    *,
    lookback_bins: int = 26,
    pseudocount: float = 0.5,
    max_log_growth_per_bin: float = 0.35,
    total_lookback_bins: int = 4,
) -> pd.DataFrame:
    """Estimate count-weighted relative growth and predict expected counts."""

    if lookback_bins < 3:
        raise ValueError("lookback_bins must be at least 3")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    data = _validate_history(history_counts).tail(lookback_bins)
    future_index = pd.Index(future_index, name=history_counts.index.name)
    counts = data.to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    active = counts.sum(axis=0) > 0
    active_counts = counts[:, active]
    probabilities = (active_counts + pseudocount) / (
        totals[:, None] + pseudocount * active.sum()
    )
    log_probabilities = np.log(probabilities)
    log_probabilities -= log_probabilities.mean(axis=1, keepdims=True)
    log_growth = np.diff(log_probabilities, axis=0)

    transition_sample_size = np.sqrt(np.minimum(totals[:-1], totals[1:]))
    recency_weights = np.linspace(0.25, 1.0, len(log_growth))
    transition_weights = transition_sample_size * recency_weights
    growth = np.average(log_growth, axis=0, weights=transition_weights)
    growth = np.clip(growth, -max_log_growth_per_bin, max_log_growth_per_bin)

    anchor = active_counts[-1] + pseudocount
    anchor /= anchor.sum()
    horizons = np.arange(1, len(future_index) + 1, dtype=float)[:, None]
    logits = np.log(anchor)[None, :] + horizons * growth[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    active_probabilities = np.exp(logits)
    active_probabilities /= active_probabilities.sum(axis=1, keepdims=True)
    all_probabilities = np.zeros((len(future_index), len(data.columns)), dtype=float)
    all_probabilities[:, active] = active_probabilities
    expected_counts = _probabilities_to_expected_counts(
        all_probabilities,
        data,
        total_lookback_bins=total_lookback_bins,
    )
    return pd.DataFrame(expected_counts, index=future_index, columns=data.columns)


def evaluate_frequency_forecast(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.Series:
    """Evaluate aligned frequency forecasts with four distribution metrics."""

    if actual.empty:
        raise ValueError("actual is empty")
    common_index = actual.index.intersection(forecast.index)
    if common_index.empty:
        raise ValueError("actual and forecast have no common dates")

    columns = actual.columns.union(forecast.columns, sort=False)
    observed = actual.reindex(index=common_index, columns=columns, fill_value=0.0)
    predicted = forecast.reindex(index=common_index, columns=columns, fill_value=0.0)
    non_empty = observed.sum(axis=1) > 0
    observed = observed.loc[non_empty].to_numpy(dtype=float)
    predicted = predicted.loc[non_empty].to_numpy(dtype=float)
    if not len(observed):
        raise ValueError("actual contains no non-empty bins")

    observed /= observed.sum(axis=1, keepdims=True)
    predicted = _normalize_rows(predicted, fallback=observed[0])
    error = predicted - observed
    midpoint = 0.5 * (observed + predicted)
    epsilon = 1e-12
    js_divergence = 0.5 * np.sum(
        observed * np.log((observed + epsilon) / (midpoint + epsilon))
        + predicted * np.log((predicted + epsilon) / (midpoint + epsilon)),
        axis=1,
    )
    return pd.Series(
        {
            "MAE": np.abs(error).mean(),
            "RMSE": np.sqrt(np.square(error).mean()),
            "mean_TVD": (0.5 * np.abs(error).sum(axis=1)).mean(),
            "mean_JSD": js_divergence.mean(),
        }
    )


def run_benchmark_backtest(
    proportions: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    horizon_bins: int = 12,
    lookback_bins: int = 26,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Fit all B-series baselines at one cutoff and score the holdout period."""

    if horizon_bins <= 0:
        raise ValueError("horizon_bins must be positive")
    cutoff = pd.Timestamp(cutoff)
    eligible = np.flatnonzero(proportions.index <= cutoff)
    if not len(eligible):
        raise ValueError("cutoff precedes the first frequency bin")
    cutoff_position = int(eligible[-1])
    history = proportions.iloc[: cutoff_position + 1]
    actual = proportions.iloc[
        cutoff_position + 1 : cutoff_position + 1 + horizon_bins
    ]
    actual = actual.loc[actual.sum(axis=1) > 0]
    if actual.empty:
        raise ValueError("no non-empty holdout bins follow cutoff")

    future_index = actual.index
    predictions = {
        "B0": forecast_b0_persistence(history, future_index),
        "B1": forecast_b1_frequency_trend(
            history, future_index, lookback_bins=lookback_bins
        ),
        "B2": forecast_b2_renewal_selection(
            history, future_index, lookback_bins=lookback_bins
        ),
    }
    metrics = pd.DataFrame(
        {
            MODEL_LABELS[name]: evaluate_frequency_forecast(actual, prediction)
            for name, prediction in predictions.items()
        }
    ).T
    metrics.index.name = "model"
    return predictions, actual, metrics.sort_values("mean_TVD")


def run_six_model_backtest(
    counts: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    horizon_bins: int = 12,
    lookback_bins: int = 26,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compare three frequency-trained and three count-trained baselines.

    Returns frequency predictions for all six models, raw expected-count
    predictions for the three count-trained models, actual holdout counts,
    actual holdout frequencies, and a common frequency-metric table.
    """

    if horizon_bins <= 0:
        raise ValueError("horizon_bins must be positive")
    count_data = _validate_history(counts)
    cutoff = pd.Timestamp(cutoff)
    eligible = np.flatnonzero(count_data.index <= cutoff)
    if not len(eligible):
        raise ValueError("cutoff precedes the first count bin")
    cutoff_position = int(eligible[-1])
    history_counts = count_data.iloc[: cutoff_position + 1]
    actual_counts = count_data.iloc[
        cutoff_position + 1 : cutoff_position + 1 + horizon_bins
    ]
    actual_counts = actual_counts.loc[actual_counts.sum(axis=1) > 0]
    if actual_counts.empty:
        raise ValueError("no non-empty holdout bins follow cutoff")

    history_frequencies = counts_to_frequencies(history_counts)
    actual_frequencies = counts_to_frequencies(actual_counts)
    future_index = actual_counts.index
    frequency_predictions = {
        "F-B0": forecast_b0_persistence(history_frequencies, future_index),
        "F-B1": forecast_b1_frequency_trend(
            history_frequencies, future_index, lookback_bins=lookback_bins
        ),
        "F-B2": forecast_b2_renewal_selection(
            history_frequencies, future_index, lookback_bins=lookback_bins
        ),
    }
    count_predictions = {
        "C-B0": forecast_count_b0_persistence(history_counts, future_index),
        "C-B1": forecast_count_b1_trend(
            history_counts, future_index, lookback_bins=lookback_bins
        ),
        "C-B2": forecast_count_b2_renewal(
            history_counts, future_index, lookback_bins=lookback_bins
        ),
    }
    for name, prediction in count_predictions.items():
        frequency_predictions[name] = counts_to_frequencies(prediction)

    metrics = pd.DataFrame(
        {
            SIX_MODEL_LABELS[name]: evaluate_frequency_forecast(
                actual_frequencies, prediction
            )
            for name, prediction in frequency_predictions.items()
        }
    ).T
    metrics.insert(
        0,
        "training_input",
        ["frequency" if name.startswith("Frequency") else "counts" for name in metrics.index],
    )
    metrics.index.name = "model"
    return (
        frequency_predictions,
        count_predictions,
        actual_counts,
        actual_frequencies,
        metrics.sort_values("mean_TVD"),
    )


def build_window_aligned_counts(
    frame: pd.DataFrame,
    *,
    window_start: str | pd.Timestamp,
    window_end: str | pd.Timestamp,
    train_start: str | pd.Timestamp = "2017-01-01",
    bin_days: int = 14,
) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    """Aggregate counts on bins aligned to a calendar holdout window.

    Only complete bins contained in the requested holdout are retained.  The
    same origin is extended backwards through the training period, preventing
    a bin from straddling the train/test boundary.
    """

    required = {"collection_date", "prediction_branch"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    if bin_days <= 0:
        raise ValueError("bin_days must be positive")

    origin = pd.Timestamp(window_start).normalize()
    requested_end = pd.Timestamp(window_end).normalize()
    training_start = pd.Timestamp(train_start).normalize()
    if requested_end < origin:
        raise ValueError("window_end must not precede window_start")
    if training_start >= origin:
        raise ValueError("train_start must precede window_start")

    days_in_window = (requested_end - origin).days + 1
    horizon_bins = days_in_window // bin_days
    if horizon_bins <= 0:
        raise ValueError("window contains no complete bins")
    evaluation_end = origin + pd.Timedelta(days=horizon_bins * bin_days - 1)

    data = frame.loc[
        frame["collection_date"].notna(),
        ["collection_date", "prediction_branch"],
    ].copy()
    data = data.loc[
        data["collection_date"].between(
            training_start, evaluation_end, inclusive="both"
        )
    ]
    if data.empty:
        raise ValueError("no records fall inside the train/evaluation period")

    offsets = (data["collection_date"] - origin).dt.days // bin_days
    data["bin_start"] = origin + pd.to_timedelta(offsets * bin_days, unit="D")
    counts = pd.crosstab(data["bin_start"], data["prediction_branch"])
    first_bin = counts.index.min()
    complete_index = pd.date_range(
        first_bin,
        origin + pd.Timedelta(days=(horizon_bins - 1) * bin_days),
        freq=f"{bin_days}D",
        name="bin_start",
    )
    counts = counts.reindex(index=complete_index, fill_value=0)
    return counts, evaluation_end, horizon_bins


def run_rolling_six_model_backtest(
    frame: pd.DataFrame,
    *,
    windows: Iterable[tuple[str | pd.Timestamp, str | pd.Timestamp]] = DEFAULT_HALF_YEAR_WINDOWS,
    train_start: str | pd.Timestamp = "2017-01-01",
    bin_days: int = 14,
    lookback_bins: int = 26,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the six-model comparison over multiple calendar holdout windows."""

    detail_rows: list[dict[str, object]] = []
    for window_number, (window_start, window_end) in enumerate(windows, start=1):
        start = pd.Timestamp(window_start).normalize()
        requested_end = pd.Timestamp(window_end).normalize()
        aligned_counts, evaluation_end, horizon_bins = build_window_aligned_counts(
            frame,
            window_start=start,
            window_end=requested_end,
            train_start=train_start,
            bin_days=bin_days,
        )
        (
            frequency_predictions,
            _,
            actual_counts,
            actual_frequencies,
            metrics,
        ) = run_six_model_backtest(
            aligned_counts,
            cutoff=start - pd.Timedelta(days=1),
            horizon_bins=horizon_bins,
            lookback_bins=lookback_bins,
        )
        actual_top = actual_frequencies.idxmax(axis=1)
        reverse_labels = {label: name for name, label in SIX_MODEL_LABELS.items()}
        for model_label, metric_values in metrics.iterrows():
            model_id = reverse_labels[model_label]
            predicted_top = frequency_predictions[model_id].idxmax(axis=1)
            detail_rows.append(
                {
                    "window_number": window_number,
                    "window_start": start,
                    "requested_window_end": requested_end,
                    "evaluation_end": evaluation_end,
                    "horizon_bins": horizon_bins,
                    "holdout_sequences": int(actual_counts.to_numpy().sum()),
                    "model_id": model_id,
                    "model": model_label,
                    "training_input": metric_values["training_input"],
                    "MAE": metric_values["MAE"],
                    "RMSE": metric_values["RMSE"],
                    "mean_TVD": metric_values["mean_TVD"],
                    "mean_JSD": metric_values["mean_JSD"],
                    "top1_accuracy": (predicted_top == actual_top).mean(),
                }
            )

    detail = pd.DataFrame(detail_rows)
    winners = detail.loc[
        detail.groupby("window_number")["mean_TVD"].transform("min")
        == detail["mean_TVD"]
    ].groupby("model_id").size()
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
    summary["window_wins"] = summary["model_id"].map(winners).fillna(0).astype(int)
    summary = summary.sort_values("mean_TVD", ignore_index=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return detail, summary
