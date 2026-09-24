"""8. Validation of the fixed crisis windows and the stress-test calculation."""

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as engine
from dashboard import analytics
from dashboard.config import BENCHMARKS, CRISIS_PERIODS
from tests.conftest import DEFAULT_WEIGHTS, FREQUENCIES

EXPECTED_WINDOWS = {
    "Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "COVID-19 Shock": ("2020-02-19", "2020-03-23"),
    "2022 Inflation / Energy Shock": ("2022-01-01", "2022-12-31"),
}
EXPECTED_BOUNDS = {  # first and last NYSE trading day inside each window
    "Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "COVID-19 Shock": ("2020-02-19", "2020-03-23"),
    "2022 Inflation / Energy Shock": ("2022-01-03", "2022-12-30"),
}
# NYSE full-day closures inside the windows (independent of the data file).
NYSE_HOLIDAYS = pd.DatetimeIndex([
    "2007-11-22", "2007-12-25", "2008-01-01", "2008-01-21", "2008-02-18", "2008-03-21",
    "2008-05-26", "2008-07-04", "2008-09-01", "2008-11-27", "2008-12-25", "2009-01-01",
    "2009-01-19", "2009-02-16",
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04",
    "2022-09-05", "2022-11-24", "2022-12-26",
])


def nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end).difference(NYSE_HOLIDAYS)


def test_crisis_windows_are_unchanged_from_v1():
    assert CRISIS_PERIODS == EXPECTED_WINDOWS


@pytest.mark.parametrize("name", list(EXPECTED_WINDOWS))
def test_trading_day_subset(prices, name):
    start, end = CRISIS_PERIODS[name]
    first, last = EXPECTED_BOUNDS[name]
    subset = engine.filter_prices_by_date(prices, start, end)
    expected_days = nyse_trading_days(start, end)
    assert subset.index[0] == pd.Timestamp(first)
    assert subset.index[-1] == pd.Timestamp(last)
    assert subset.index.equals(expected_days), "data is missing (or has extra) trading days"
    rows = len(expected_days)
    print(f"\n{name}: {first} -> {last}, {rows} trading days")
    pd.testing.assert_frame_equal(subset, prices.loc[start:end])

    summary = engine.calculate_crisis_summary(prices, start, end, DEFAULT_WEIGHTS, {"SPY": 100})
    assert summary["Start"] == pd.Timestamp(first) and summary["End"] == pd.Timestamp(last)
    assert summary["Trading Days"] == rows - 1
    assert summary["Portfolio Path"].index.equals(subset.index)


@pytest.mark.parametrize("name, peak, trough", [
    ("Global Financial Crisis", "2007-10-09", "2009-03-09"),
    ("COVID-19 Shock", "2020-02-19", "2020-03-23"),
])
def test_windows_bracket_the_actual_spy_peak_and_trough(prices, name, peak, trough):
    """Market-history check: the windows capture the S&P 500's real peak-to-trough moves."""
    spy = engine.filter_prices_by_date(prices, *CRISIS_PERIODS[name])["SPY"]
    assert spy.idxmax() == pd.Timestamp(peak)
    assert spy.idxmin() == pd.Timestamp(trough)


def test_2022_window_is_the_calendar_year(prices):
    spy = engine.filter_prices_by_date(prices, *CRISIS_PERIODS["2022 Inflation / Energy Shock"])["SPY"]
    assert spy.idxmax() == pd.Timestamp("2022-01-03")          # the 2022 market peak
    assert spy.idxmin() == pd.Timestamp("2022-10-12")          # the 2022 bear-market low
    assert spy.iloc[-1] / spy.iloc[0] - 1 < -0.15              # SPY was down roughly 18-19% in 2022


@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("name", list(EXPECTED_WINDOWS))
def test_crisis_summary_uses_selected_rebalancing_and_same_method_for_benchmark(prices, name, frequency):
    start, end = CRISIS_PERIODS[name]
    subset = engine.filter_prices_by_date(prices, start, end)
    benchmark_weights = BENCHMARKS["60/40 SPY + AGG"]
    summary = engine.calculate_crisis_summary(prices, start, end, DEFAULT_WEIGHTS, benchmark_weights, frequency)

    direct_portfolio = engine.calculate_rebalanced_portfolio(subset, DEFAULT_WEIGHTS, 1000, frequency)
    direct_benchmark = engine.calculate_rebalanced_portfolio(subset, benchmark_weights, 1000, frequency)
    np.testing.assert_allclose(summary["Portfolio Path"], direct_portfolio["Portfolio Value"] / 1000, rtol=1e-12)
    np.testing.assert_allclose(summary["Benchmark Path"], direct_benchmark["Portfolio Value"] / 1000, rtol=1e-12)

    mp, mb = engine.calculate_metrics(direct_portfolio), engine.calculate_metrics(direct_benchmark)
    assert summary["Portfolio Return"] == pytest.approx(mp["Total Return"], rel=1e-12)
    assert summary["Portfolio Max DD"] == pytest.approx(mp["Max Drawdown"], rel=1e-12)
    assert summary["Portfolio Volatility"] == pytest.approx(mp["Annual Volatility"], rel=1e-12)
    assert summary["Benchmark Return"] == pytest.approx(mb["Total Return"], rel=1e-12)
    assert summary["Benchmark Max DD"] == pytest.approx(mb["Max Drawdown"], rel=1e-12)

    asset_returns = subset.iloc[-1] / subset.iloc[0] - 1
    assert summary["Best ETF"] == asset_returns.idxmax()
    assert summary["Worst ETF"] == asset_returns.idxmin()


def test_rebalancing_choice_changes_multi_period_crisis_results(prices):
    start, end = CRISIS_PERIODS["Global Financial Crisis"]
    returns = {
        f: engine.calculate_crisis_summary(prices, start, end, DEFAULT_WEIGHTS, {"SPY": 100}, f)["Portfolio Return"]
        for f in FREQUENCIES
    }
    print("\nGFC portfolio return by rebalancing rule:", {k: f"{v:.4%}" for k, v in returns.items()})
    assert len({round(v, 12) for v in returns.values()}) == 4


def test_covid_window_has_no_boundary_except_month_end(prices):
    """19 Feb - 23 Mar 2020 crosses only the 1 March month boundary."""
    start, end = CRISIS_PERIODS["COVID-19 Shock"]
    r = {f: engine.calculate_crisis_summary(prices, start, end, DEFAULT_WEIGHTS, {"SPY": 100}, f)["Portfolio Return"]
         for f in FREQUENCIES}
    assert r["never"] == r["quarterly"] == r["annually"]
    assert r["monthly"] != r["never"]


@pytest.mark.parametrize("rebalance", ["Never", "Monthly", "Quarterly", "Annually"])
def test_app_layer_passes_rebalancing_to_stress_test(rebalance):
    """The cached function the Streamlit page calls must honour the sidebar's rebalancing choice."""
    version = analytics.get_data_version()
    periods = tuple((n, s, e) for n, (s, e) in CRISIS_PERIODS.items())
    summaries = analytics.crisis_summaries(
        version, analytics.weights_key(DEFAULT_WEIGHTS), analytics.weights_key({"SPY": 100}), rebalance, periods
    )
    prices = engine.load_prices()
    for name, (start, end) in CRISIS_PERIODS.items():
        direct = engine.calculate_crisis_summary(prices, start, end, DEFAULT_WEIGHTS, {"SPY": 100}, rebalance.lower())
        assert summaries[name]["Portfolio Return"] == pytest.approx(direct["Portfolio Return"], rel=1e-12)
