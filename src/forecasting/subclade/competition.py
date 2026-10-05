"""A lightweight change-point, multi-clade frequency forecaster.

The model represents a frequency vector by its centred log-ratio (CLR)
coordinates.  In a stable competitive regime its latent state follows a local
linear process::

    z_t = z_{t-1} + v + epsilon_t

where ``z_t`` is the CLR state and ``v`` is estimated with exponentially
weighted ridge regression.  A composition shift in a recent lookback window
is treated as a change point, so the trend is fitted only after that shift.
Predicted CLR states are mapped back to the probability simplex with softmax.

This deliberately small model is intended for transparent notebook backtests,
not as a replacement for a fully specified Bayesian state-space model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastDiagnostics:
    """Information about the local regime used for one forecast."""

    lookback_start: pd.Timestamp
    change_point: pd.Timestamp | None
    regime_start: pd.Timestamp
    change_score: float
    slope_norm: float
    regime_bins: int


class ChangePointSoftmaxForecaster:
    """Forecast competing subclade frequencies in CLR/softmax coordinates.

    Parameters
    ----------
    lookback_bins:
        Maximum number of non-empty bins considered for the local regime.
    min_regime_bins:
        Minimum number of bins on either side of a candidate change point.
    pseudocount:
        Symmetric Dirichlet pseudo-count applied before the log transform.
    decay:
        Exponential weight assigned to an observation one bin older.  Values
        below one make the fitted slope responsive to recent changes.
    ridge:
        Denominator penalty for the local slope estimate.
    change_threshold:
        Minimum Euclidean CLR mean shift required to reset the regime.
    max_step_norm:
        Upper bound on the Euclidean norm of the one-bin CLR extrapolation.
        This prevents a short, noisy regime from producing extreme softmax
        probabilities many bins into the future.
    trend_damping:
        Geometric damping applied to the local trend at each forecast step.
        A value below one lets a replacement wave continue briefly without
        assuming that its current log-ratio velocity persists for six months.
    """

    def __init__(
        self,
        *,
        lookback_bins: int = 26,
        min_regime_bins: int = 5,
        pseudocount: float = 0.5,
        decay: float = 0.92,
        ridge: float = 1.0,
        change_threshold: float = 0.35,
        max_step_norm: float = 0.18,
        trend_damping: float = 0.85,
    ) -> None:
        if lookback_bins < 2 * min_regime_bins:
            raise ValueError("lookback_bins must be at least twice min_regime_bins")
        if min_regime_bins < 2:
            raise ValueError("min_regime_bins must be at least 2")
        if pseudocount <= 0 or ridge < 0 or max_step_norm <= 0:
            raise ValueError("pseudocount and max_step_norm must be positive; ridge must be non-negative")
        if not 0 < decay <= 1 or not 0 < trend_damping <= 1:
            raise ValueError("decay and trend_damping must be in (0, 1]")
        if change_threshold < 0:
            raise ValueError("change_threshold must be non-negative")
        self.lookback_bins = lookback_bins
        self.min_regime_bins = min_regime_bins
        self.pseudocount = pseudocount
        self.decay = decay
        self.ridge = ridge
        self.change_threshold = change_threshold
        self.max_step_norm = max_step_norm
        self.trend_damping = trend_damping
        self.last_diagnostics: ForecastDiagnostics | None = None

    def predict(
        self,
        counts: pd.DataFrame,
        *,
        horizon: int,
        future_index: Iterable[object] | None = None,
    ) -> pd.DataFrame:
        """Predict ``horizon`` future frequency vectors from a count history."""

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        data = self._validate_counts(counts)
        state = self._clr(data.to_numpy(dtype=float))
        state_index = data.index
        start = max(0, len(data) - self.lookback_bins)
        local_state = state[start:]
        local_index = state_index[start:]
        split, score = self._change_point(local_state)
        regime_state = local_state[split:]
        slope = self._weighted_slope(regime_state)
        slope_norm = float(np.linalg.norm(slope))
        if slope_norm > self.max_step_norm:
            slope *= self.max_step_norm / slope_norm
            slope_norm = self.max_step_norm

        # Sum a damped local velocity: v + phi*v + ... rather than h*v.
        # This is the state-space forecast of a velocity that decays to zero.
        steps = np.cumsum(self.trend_damping ** np.arange(horizon, dtype=float))[:, None]
        predicted = self._softmax(local_state[-1] + steps * slope)
        if future_index is None:
            future_index = pd.RangeIndex(1, horizon + 1, name="forecast_step")
        index = pd.Index(future_index)
        if len(index) != horizon:
            raise ValueError("future_index must have exactly horizon entries")

        self.last_diagnostics = ForecastDiagnostics(
            lookback_start=pd.Timestamp(local_index[0]),
            change_point=pd.Timestamp(local_index[split]) if split else None,
            regime_start=pd.Timestamp(local_index[split]),
            change_score=score,
            slope_norm=slope_norm,
            regime_bins=len(regime_state),
        )
        return pd.DataFrame(predicted, index=index, columns=data.columns)

    def _change_point(self, state: np.ndarray) -> tuple[int, float]:
        """Return the strongest recent composition shift, if it is material."""

        n = len(state)
        if n < 2 * self.min_regime_bins:
            return 0, 0.0
        candidates: list[tuple[float, int]] = []
        for split in range(self.min_regime_bins, n - self.min_regime_bins + 1):
            before = state[max(0, split - self.min_regime_bins) : split].mean(axis=0)
            after = state[split : min(n, split + self.min_regime_bins)].mean(axis=0)
            candidates.append((float(np.linalg.norm(after - before)), split))
        score, split = max(candidates, key=lambda item: item[0])
        return (split, score) if score >= self.change_threshold else (0, score)

    def _weighted_slope(self, state: np.ndarray) -> np.ndarray:
        """Estimate a local CLR velocity with exponentially weighted ridge WLS."""

        n = len(state)
        if n < 2:
            return np.zeros(state.shape[1], dtype=float)
        x = np.arange(n, dtype=float)
        weights = self.decay ** (n - 1 - x)
        x_centered = x - np.average(x, weights=weights)
        y_centered = state - np.average(state, axis=0, weights=weights)
        denominator = float(np.sum(weights * np.square(x_centered)) + self.ridge)
        return (weights * x_centered) @ y_centered / denominator

    def _validate_counts(self, counts: pd.DataFrame) -> pd.DataFrame:
        if counts.empty:
            raise ValueError("counts must contain at least one row")
        data = counts.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if (data < 0).any().any():
            raise ValueError("counts cannot contain negative values")
        data = data.loc[data.sum(axis=1) > 0]
        if len(data) < 2:
            raise ValueError("counts must contain at least two non-empty bins")
        # A branch first seen after the cutoff is not an available state at
        # forecast time.  Dropping it here avoids leaking future categories
        # into the simplex through the pseudo-count.
        data = data.loc[:, data.sum(axis=0) > 0]
        if data.shape[1] < 2:
            raise ValueError("history must contain at least two observed branches")
        return data

    def _clr(self, counts: np.ndarray) -> np.ndarray:
        probabilities = (counts + self.pseudocount) / (
            counts.sum(axis=1, keepdims=True) + self.pseudocount * counts.shape[1]
        )
        log_probabilities = np.log(probabilities)
        return log_probabilities - log_probabilities.mean(axis=1, keepdims=True)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        result = np.exp(shifted)
        return result / result.sum(axis=1, keepdims=True)
