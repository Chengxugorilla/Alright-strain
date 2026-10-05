"""Export binned Nextclade subclade counts for forecasts-flu."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strain_data.nextclade import DEFAULT_START_DATE, build_binned_tables, read_nextclade_table


def export_counts(
    input_tsv: Path,
    output_tsv: Path,
    *,
    label_mode: str = "nextclade",
    bin_days: int = 14,
    start_date: str = DEFAULT_START_DATE,
) -> None:
    """Write a bin-by-subclade count table consumable by forecasts-flu."""

    records = read_nextclade_table(input_tsv, label_mode=label_mode)
    counts, _, _ = build_binned_tables(
        records,
        bin_days=bin_days,
        start_date=start_date,
    )
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_tsv, sep="\t")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("output_tsv", type=Path)
    parser.add_argument("--label-mode", choices=["nextclade", "legacy"], default="nextclade")
    parser.add_argument("--bin-days", type=int, default=14)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    args = parser.parse_args()
    export_counts(
        args.input_tsv,
        args.output_tsv,
        label_mode=args.label_mode,
        bin_days=args.bin_days,
        start_date=args.start_date,
    )
    print(f"output={args.output_tsv}")


if __name__ == "__main__":
    main()
