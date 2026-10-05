"""Read Nextclade H1N1 results and plot collection-time subclade proportions.

The input TSV is treated as read-only.  Only the columns required for the
aggregation are loaded into memory; sequence strings are never printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UNASSIGNED = "Unassigned"
MISSING_BRANCH_VALUES = {"", "unassigned", "unknown", "nan", "none", "<na>"}
DEFAULT_START_DATE = "2017-01-01"
BRANCH_COLUMN_PRIORITIES = {
    "nextclade": ("subclade", "clade", "legacy-clade"),
    "legacy": ("legacy-clade", "subclade", "clade"),
}


def _valid_branch(values: pd.Series) -> pd.Series:
    normalized = values.fillna("").astype("string").str.strip()
    return normalized.where(~normalized.str.lower().isin(MISSING_BRANCH_VALUES))


def extract_collection_date(seq_names: pd.Series) -> pd.Series:
    """Extract the first complete ISO date from each pipe-delimited seqName.

    In the GISAID headers used by this project, the first complete date is the
    collection date and a later complete date is the submission date.  The
    parser does not rely on a fixed pipe-field position because both 17- and
    18-field headers occur in the project.
    """

    date_text = seq_names.astype("string").str.extract(
        r"(?<!\d)((?:19|20)\d{2}-\d{2}-\d{2})(?!\d)", expand=False
    )
    return pd.to_datetime(date_text, errors="coerce")


def read_nextclade_table(
    path: str | Path,
    *,
    label_mode: str = "nextclade",
) -> pd.DataFrame:
    """Load the columns needed for the selected branch-label convention.

    ``nextclade`` follows the competition convention: prefer ``subclade``,
    then ``clade``, and use ``legacy-clade`` only as a final fallback.
    ``legacy`` preserves the demo's former legacy-first behavior.
    """

    if label_mode not in BRANCH_COLUMN_PRIORITIES:
        raise ValueError(
            f"Unknown label_mode {label_mode!r}; expected one of "
            f"{sorted(BRANCH_COLUMN_PRIORITIES)}"
        )
    branch_priority = BRANCH_COLUMN_PRIORITIES[label_mode]

    path = Path(path)
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    required = {"seqName"}
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    branch_columns = [name for name in branch_priority if name in header]
    if not branch_columns:
        raise ValueError(
            "The TSV contains none of the expected branch columns: "
            f"{list(branch_priority)}"
        )

    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["seqName", *branch_columns],
        dtype="string",
        low_memory=False,
    )
    frame["collection_date"] = extract_collection_date(frame["seqName"])

    branch = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in branch_columns:
        branch = branch.fillna(_valid_branch(frame[column]))
    frame["prediction_branch"] = branch.fillna(UNASSIGNED)
    return frame


def build_binned_tables(
    frame: pd.DataFrame,
    *,
    bin_days: int = 14,
    start_date: str | pd.Timestamp | None = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return bin-by-branch counts, proportions, and bin totals."""

    if bin_days <= 0:
        raise ValueError("bin_days must be positive")

    data = frame.loc[frame["collection_date"].notna(), [
        "collection_date",
        "prediction_branch",
    ]].copy()
    if data.empty:
        raise ValueError("No valid collection dates were found.")

    start = (
        pd.Timestamp(start_date).normalize()
        if start_date is not None
        else data["collection_date"].min().normalize()
    )
    end = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else data["collection_date"].max().normalize()
    )
    if end < start:
        raise ValueError("end_date must not precede start_date")

    data = data.loc[data["collection_date"].between(start, end, inclusive="both")]
    if data.empty:
        raise ValueError("No records remain inside the requested date range.")

    offsets = (data["collection_date"] - start).dt.days // bin_days
    data["bin_start"] = start + pd.to_timedelta(offsets * bin_days, unit="D")

    first_seen = data.groupby("prediction_branch")["collection_date"].min()
    branch_order = first_seen.sort_values(kind="stable").index.tolist()
    if UNASSIGNED in branch_order:
        branch_order.remove(UNASSIGNED)
        branch_order.insert(0, UNASSIGNED)

    counts = pd.crosstab(data["bin_start"], data["prediction_branch"])
    complete_bins = pd.date_range(
        start=start,
        end=start + pd.Timedelta(days=((end - start).days // bin_days) * bin_days),
        freq=f"{bin_days}D",
        name="bin_start",
    )
    counts = counts.reindex(index=complete_bins, columns=branch_order, fill_value=0)
    totals = counts.sum(axis=1).rename("total_count")
    proportions = counts.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    return counts, proportions, totals


def _branch_colors(branches: list[str]) -> dict[str, object]:
    import matplotlib.colors as mcolors

    named = [branch for branch in branches if branch != UNASSIGNED]
    anchors = [
        "#f8766d",
        "#e69f00",
        "#b79f00",
        "#39b600",
        "#00ba79",
        "#00bfc4",
        "#00a9e6",
        "#619cff",
        "#9c78ef",
        "#d65ce6",
        "#ff62bc",
        "#ff6f91",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("subclade_spectrum", anchors)
    positions = np.linspace(0, 1, max(len(named), 1))
    colors = {branch: cmap(pos) for branch, pos in zip(named, positions)}
    colors[UNASSIGNED] = "#cfcfcf"
    return colors


def plot_subclade_area(
    proportions: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
    legend_title: str = "Nextclade subclade",
    omit_empty_bins: bool = True,
):
    """Create a stacked area plot matching the supplied reference style."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if proportions.empty:
        raise ValueError("proportions is empty")

    plot_data = proportions
    if omit_empty_bins:
        plot_data = proportions.loc[proportions.sum(axis=1) > 0]
    if plot_data.empty:
        raise ValueError("proportions contains no non-empty bins")

    colors = _branch_colors(plot_data.columns.tolist())
    figure, axis = plt.subplots(figsize=(24, 9), constrained_layout=True)
    axis.stackplot(
        plot_data.index,
        plot_data.to_numpy().T,
        labels=plot_data.columns,
        colors=[colors[name] for name in plot_data.columns],
        edgecolor="white",
        linewidth=0.55,
    )
    axis.set_ylim(0, 1)
    axis.set_xlim(plot_data.index.min(), plot_data.index.max())
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_yticks(np.linspace(0, 1, 5))
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlabel("Collection Date", fontsize=20, fontweight="bold")
    axis.set_ylabel("Proportion", fontsize=20, fontweight="bold")
    if title:
        axis.set_title(title, fontsize=21, fontweight="bold", pad=14)
    axis.tick_params(axis="both", labelsize=15, width=1.5)
    axis.tick_params(axis="x", rotation=45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(1.8)
    legend = axis.legend(
        title=legend_title,
        bbox_to_anchor=(1.015, 1.0),
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=12,
        title_fontsize=17,
        columnspacing=1.2,
        handlelength=1.2,
        handleheight=1.2,
    )
    legend.get_title().set_fontweight("bold")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    return figure, axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--bin-days", type=int, default=14)
    parser.add_argument(
        "--label-mode",
        choices=sorted(BRANCH_COLUMN_PRIORITIES),
        default="nextclade",
        help="Branch-label convention (default: nextclade)",
    )
    args = parser.parse_args()

    frame = read_nextclade_table(args.input_tsv, label_mode=args.label_mode)
    counts, proportions, totals = build_binned_tables(
        frame,
        bin_days=args.bin_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    is_legacy = args.label_mode == "legacy"
    figure, _ = plot_subclade_area(
        proportions,
        output_path=args.output_png,
        title=(
            "H1N1 legacy subclade proportions by collection date"
            if is_legacy
            else "H1N1 Nextclade subclade proportions by collection date"
        ),
        legend_title="Legacy subclade" if is_legacy else "Nextclade subclade",
    )

    import matplotlib.pyplot as plt

    plt.close(figure)
    print(f"rows_loaded={len(frame):,}")
    print(f"valid_collection_dates={frame['collection_date'].notna().sum():,}")
    print(f"date_range={frame['collection_date'].min().date()}..{frame['collection_date'].max().date()}")
    print(f"bins={len(counts):,}")
    print(f"branches={len(proportions.columns):,}")
    print(f"label_mode={args.label_mode}")
    print(f"records_in_plot={int(totals.sum()):,}")
    print(f"output={args.output_png}")


if __name__ == "__main__":
    main()
