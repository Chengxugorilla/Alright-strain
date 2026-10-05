"""Plot local BERT score statistics over collection date."""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_metadata.csv",
    )
    parser.add_argument("--bin-days", type=int, default=14)
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_score_by_date.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/bert_reference_free_ranking_score_by_date.png",
    )
    args = parser.parse_args()
    if args.bin_days <= 0:
        raise ValueError("--bin-days 必须为正数。")

    data = pd.read_csv(args.metadata_csv, usecols=["collection_date", "bert_score"])
    data["collection_date"] = pd.to_datetime(data["collection_date"], errors="coerce")
    data = data.dropna(subset=["collection_date", "bert_score"])
    start = pd.Timestamp(args.start_date)
    data = data.loc[data["collection_date"] >= start].copy()
    if data.empty:
        raise RuntimeError("指定日期范围内没有有效分数。")
    data["bin_start"] = start + pd.to_timedelta(
        ((data["collection_date"] - start).dt.days // args.bin_days) * args.bin_days,
        unit="D",
    )
    summary = (
        data.groupby("bin_start", sort=True)["bert_score"]
        .agg(
            sequence_count="size",
            mean="mean",
            median="median",
            q10=lambda values: values.quantile(0.10),
            q90=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)

    figure, (score_axis, count_axis) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    score_axis.fill_between(
        summary["bin_start"], summary["q10"], summary["q90"],
        color="#4c78a8", alpha=0.20, label="10–90% quantile",
    )
    score_axis.plot(summary["bin_start"], summary["mean"], color="#1f4e79", linewidth=2.2, label="Mean")
    score_axis.plot(summary["bin_start"], summary["median"], color="#e67e22", linewidth=1.5, label="Median")
    score_axis.set_ylabel("BERT score\n(mean log probability)")
    score_axis.set_title("H1N1 masked-language-model score by collection date", fontweight="bold")
    score_axis.grid(alpha=0.25)
    score_axis.legend(frameon=False, ncol=3)

    count_axis.bar(summary["bin_start"], summary["sequence_count"], width=args.bin_days * 0.8, color="#7f7f7f")
    count_axis.set_ylabel("Sequences")
    count_axis.set_xlabel("Collection date")
    count_axis.grid(axis="y", alpha=0.25)
    count_axis.xaxis.set_major_locator(mdates.YearLocator())
    count_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.autofmt_xdate()
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure_output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"日期统计：{args.summary_output}")
    print(f"图：{args.figure_output}")


if __name__ == "__main__":
    main()
