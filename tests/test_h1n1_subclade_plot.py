import pandas as pd
import pytest

from src.h1n1_benchmarks import (
    evaluate_frequency_forecast,
    forecast_b0_persistence,
    forecast_b1_frequency_trend,
    forecast_b2_renewal_selection,
    run_benchmark_backtest,
)

from src.h1n1_subclade_plot import (
    DEFAULT_START_DATE,
    UNASSIGNED,
    build_binned_tables,
    extract_collection_date,
    read_nextclade_table,
)


def test_extract_collection_date_from_variable_width_headers():
    names = pd.Series(
        [
            "A/Test/1/20|ID1|A_/_H1N1|Original|pdm09|X|2020-01-02|lab|x||2020-02-03",
            "A/Test/2/20|ID2|A_/_H1N1|Original|pdm09|X|Y|2021-04-05|lab|x||2021-05-06",
        ],
        dtype="string",
    )
    result = extract_collection_date(names)
    assert result.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2021-04-05")]


def test_build_14_day_counts_and_proportions():
    frame = pd.DataFrame(
        {
            "collection_date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-15", "2020-01-16"]
            ),
            "prediction_branch": ["A", "A", "A", UNASSIGNED],
        }
    )
    counts, proportions, totals = build_binned_tables(
        frame,
        start_date="2020-01-01",
        end_date="2020-01-28",
        bin_days=14,
    )
    assert totals.tolist() == [2, 2]
    assert counts.loc[pd.Timestamp("2020-01-01"), "A"] == 2
    assert proportions.loc[pd.Timestamp("2020-01-15"), "A"] == 0.5
    assert proportions.loc[pd.Timestamp("2020-01-15"), UNASSIGNED] == 0.5
    assert proportions.sum(axis=1).tolist() == [1.0, 1.0]


def _write_branch_fixture(tmp_path):
    path = tmp_path / "nextclade.tsv"
    path.write_text(
        "\t".join(["seqName", "legacy-clade", "subclade", "clade"]) + "\n"
        + "\n".join(
            "\t".join(row)
            for row in [
                [
                    "A/Test/1/20|EPI1|A_/_H1N1|Original|pdm09|6B.1A|2020-01-02",
                    "6B.1A.5a.2a",
                    "C.1",
                    "C",
                ],
                [
                    "A/Test/2/20|EPI2|A_/_H1N1|Original|pdm09|6B.1A|2020-01-03",
                    "6B.1A.5a.2a.1",
                    "",
                    "D.1",
                ],
                [
                    "A/Test/3/20|EPI3|A_/_H1N1|Original|pdm09|6B.1A|2020-01-04",
                    "6B.1",
                    "Unassigned",
                    "unknown",
                ],
            ]
        )
        + "\n"
    )
    return path


def test_read_nextclade_table_defaults_to_nextclade_priority(tmp_path):
    path = _write_branch_fixture(tmp_path)

    result = read_nextclade_table(path)

    assert result["prediction_branch"].tolist() == ["C.1", "D.1", "6B.1"]


def test_read_nextclade_table_can_use_legacy_priority(tmp_path):
    path = _write_branch_fixture(tmp_path)

    result = read_nextclade_table(path, label_mode="legacy")

    assert result["prediction_branch"].tolist() == [
        "6B.1A.5a.2a",
        "6B.1A.5a.2a.1",
        "6B.1",
    ]


def test_default_binning_starts_in_2017():
    frame = pd.DataFrame(
        {
            "collection_date": pd.to_datetime(["2016-12-31", "2017-01-01"]),
            "prediction_branch": ["A", "B"],
        }
    )

    counts, _, totals = build_binned_tables(frame, end_date="2017-01-14")

    assert DEFAULT_START_DATE == "2017-01-01"
    assert counts.index.min() == pd.Timestamp(DEFAULT_START_DATE)
    assert totals.sum() == 1


def test_unknown_label_mode_is_rejected(tmp_path):
    path = _write_branch_fixture(tmp_path)

    try:
        read_nextclade_table(path, label_mode="other")
    except ValueError as error:
        assert "label_mode" in str(error)
    else:
        raise AssertionError("Expected an invalid label_mode to raise ValueError")


@pytest.fixture
def frequency_history():
    index = pd.date_range("2023-01-01", periods=8, freq="14D", name="bin_start")
    return pd.DataFrame(
        {
            "A": [0.80, 0.75, 0.70, 0.64, 0.58, 0.52, 0.46, 0.40],
            "B": [0.20, 0.25, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60],
        },
        index=index,
    )


def test_b0_repeats_latest_frequency(frequency_history):
    future = pd.date_range("2023-04-23", periods=3, freq="14D", name="bin_start")
    result = forecast_b0_persistence(frequency_history, future)

    assert result.index.equals(future)
    assert result["A"].tolist() == [0.4, 0.4, 0.4]
    assert result.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "forecaster", [forecast_b1_frequency_trend, forecast_b2_renewal_selection]
)
def test_trend_baselines_are_normalized_and_continue_growth(
    frequency_history, forecaster
):
    future = pd.date_range("2023-04-23", periods=3, freq="14D", name="bin_start")
    result = forecaster(frequency_history, future, lookback_bins=6)

    assert result.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert result["B"].iloc[-1] > result["B"].iloc[0]


def test_frequency_metrics_are_zero_for_perfect_forecast(frequency_history):
    metrics = evaluate_frequency_forecast(frequency_history, frequency_history)

    assert metrics.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_benchmark_backtest_returns_all_models(frequency_history):
    predictions, actual, metrics = run_benchmark_backtest(
        frequency_history,
        cutoff="2023-03-12",
        horizon_bins=2,
        lookback_bins=4,
    )

    assert set(predictions) == {"B0", "B1", "B2"}
    assert len(actual) == 2
    assert set(metrics.columns) == {"MAE", "RMSE", "mean_TVD", "mean_JSD"}
