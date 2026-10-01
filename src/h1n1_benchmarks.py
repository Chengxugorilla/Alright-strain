"""Frequency-only B-series baselines for H1N1 subclade backtesting.

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
    log_frequency = np.log(frequency + pseudocount)
    centered_log_frequency = log_frequency - log_frequency.mean(axis=1, keepdims=True)
    growth = np.median(np.diff(centered_log_frequency, axis=0), axis=0)
    growth = np.clip(growth, -max_log_growth_per_bin, max_log_growth_per_bin)

    anchor = frequency[-1]
    anchor /= anchor.sum()
    horizons = np.arange(1, len(future_index) + 1, dtype=float)[:, None]
    logits = np.log(anchor + pseudocount)[None, :] + horizons * growth[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=future_index, columns=data.columns)


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
