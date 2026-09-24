"""1. Data integrity of the production dataset (data/daily_adjusted_prices_common.csv)."""

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as engine

TICKERS = ["SPY", "GLD", "AGG", "DBC"]
ALL_DATES_CSV = engine.PROJECT_DIR / "data" / "daily_adjusted_prices_all.csv"

# Largest plausible one-day move for these diversified ETFs. The worst real moves
# since 2006 are around -11% (SPY, 16 Mar 2020); anything beyond 25% is a data error.
IMPOSSIBLE_DAILY_MOVE = 0.25
# Moves above this are listed in the report so a human can eyeball them.
NOTABLE_DAILY_MOVE = 0.08


def test_app_reads_the_common_dataset():
    assert engine.DATA_PATH.name == "daily_adjusted_prices_common.csv"
    assert engine.DATA_PATH.exists()


def test_expected_tickers_present(raw_common_csv):
    assert list(raw_common_csv.columns) == TICKERS


def test_index_is_sorted_unique_datetime(raw_common_csv):
    idx = raw_common_csv.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.is_monotonic_increasing
    assert idx.is_unique, f"duplicate dates: {idx[idx.duplicated()].tolist()}"
    assert idx.tz is None


def test_no_weekend_dates(raw_common_csv):
    weekend = raw_common_csv.index[raw_common_csv.index.dayofweek >= 5]
    assert weekend.empty, f"weekend dates in data: {weekend.tolist()}"


def test_no_missing_values(raw_common_csv):
    missing = raw_common_csv.isna().sum()
    assert missing.sum() == 0, f"NaN counts per ticker: {missing.to_dict()}"
    assert np.isfinite(raw_common_csv.to_numpy()).all()


def test_prices_are_positive(raw_common_csv):
    assert (raw_common_csv > 0).all().all()


def test_loader_does_not_drop_rows(raw_common_csv, prices):
    # load_prices() applies dropna(); on a clean file it must be a no-op.
    assert prices.equals(raw_common_csv.sort_index())


def test_date_coverage(prices):
    first, last = prices.index.min(), prices.index.max()
    print(f"\nProduction dataset: {first.date()} -> {last.date()}, {len(prices)} trading days")
    assert first == pd.Timestamp("2006-02-06"), "first date on which DBC trades alongside the others"
    assert last >= pd.Timestamp("2026-01-01")
    # ~252 trading days a year: guard against large missing blocks.
    years = (last - first).days / 365.25
    assert 245 <= len(prices) / years <= 255
    largest_gap = prices.index.to_series().diff().max()
    assert largest_gap <= pd.Timedelta(days=5), f"largest calendar gap {largest_gap}"


def test_no_impossible_daily_returns(prices):
    returns = prices.pct_change().dropna()
    extreme = returns[returns.abs() > IMPOSSIBLE_DAILY_MOVE].stack().dropna()
    assert extreme.empty, f"impossible daily moves:\n{extreme}"

    notable = returns[returns.abs() > NOTABLE_DAILY_MOVE].stack().dropna().sort_values()
    print("\nDaily moves larger than 8% (checked against market history):")
    for (date, ticker), value in notable.items():
        print(f"  {date.date()} {ticker} {value:+.2%}")
    # All notable moves must fall in known stress episodes.
    # Each episode was checked against raw Yahoo OHLC and volume (the intraday range and 2-3x
    # normal volume confirm a real move rather than a bad tick or adjustment artifact).
    stress_windows = [
        ("2008-09-01", "2009-06-30"),  # global financial crisis
        ("2013-04-01", "2013-04-30"),  # gold crash (GLD -8.8% on 15 Apr 2013)
        ("2020-03-01", "2020-04-30"),  # COVID-19 crash
        ("2025-04-01", "2025-04-30"),  # tariff shock (SPY +10.5% on 9 Apr 2025)
        ("2026-01-26", "2026-02-06"),  # gold sell-off (GLD -10.3% on 30 Jan 2026, intraday low 430.8)
    ]
    in_window = pd.Series(False, index=notable.index)
    for start, end in stress_windows:
        dates = notable.index.get_level_values(0)
        in_window |= (dates >= start) & (dates <= end)
    assert in_window.all(), f"large moves outside known crises:\n{notable[~in_window]}"


def test_no_stale_price_runs(prices):
    """A price repeating for many days in a row usually means a broken feed."""
    for ticker in prices:
        unchanged = prices[ticker].diff().eq(0)
        longest = unchanged.groupby((~unchanged).cumsum()).sum().max()
        assert longest < 5, f"{ticker} unchanged for {longest} consecutive days"


def test_common_dataset_is_intersection_of_all_dataset(raw_common_csv):
    """The app's file must be exactly the all-dates file restricted to dates where every ETF trades."""
    all_dates = pd.read_csv(ALL_DATES_CSV, index_col=0, parse_dates=True)
    expected = all_dates.dropna(how="any")
    assert raw_common_csv.index.equals(expected.index)
    pd.testing.assert_frame_equal(raw_common_csv, expected)
    # DBC launched last; before its first price the other ETFs trade alone.
    first_common = raw_common_csv.index.min()
    assert all_dates.loc[:first_common, "DBC"].iloc[:-1].isna().all()


@pytest.mark.parametrize("ticker", TICKERS)
def test_each_asset_uses_the_common_calendar(prices, ticker):
    assert prices[ticker].notna().all()
    assert prices[ticker].index.equals(prices.index)
