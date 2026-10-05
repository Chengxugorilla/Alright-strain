"""One standard chart for a time × value-series × subclade matrix."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_subclade_matrix(
    values: np.ndarray,
    timepoints: Sequence[object],
    subclades: Sequence[str],
    value_names: Sequence[str],
) -> plt.Figure:
    """Return an all-subclade line chart from a 3D value matrix.

    ``values`` must have shape ``(n_timepoints, n_value_series, n_subclades)``.
    The supplied values are plotted without transformation. Each subclade has
    one color and each value series has one line style.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 3:
        raise ValueError("values must have shape (timepoints, value_series, subclades)")
    expected_shape = (len(timepoints), len(value_names), len(subclades))
    if matrix.shape != expected_shape:
        raise ValueError(f"values has shape {matrix.shape}; expected {expected_shape}")

    figure, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
    colors = plt.get_cmap("tab20", len(subclades))
    styles = ("-", "--", ":", "-.")
    for subclade_index, subclade in enumerate(subclades):
        for value_index, value_name in enumerate(value_names):
            axis.plot(
                timepoints,
                matrix[:, value_index, subclade_index],
                color=colors(subclade_index),
                linestyle=styles[value_index % len(styles)],
                linewidth=2 if value_index == 0 else 1.6,
                label=f"{subclade} · {value_name}",
            )
    axis.set(xlabel="time", ylabel="value")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    return figure
