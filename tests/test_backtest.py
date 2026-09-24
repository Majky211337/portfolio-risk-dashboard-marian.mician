"""3, 4, 6, 7: single-asset backtests, benchmark consistency, rebalancing and initial-investment scaling."""

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as engine
from dashboard.config import BENCHMARKS
from tests.conftest import DEFAULT_WEIGHTS, FREQUENCIES
from tests.reference import share_based_backtest

TICKERS = ["SPY", "GLD", "AGG", "DBC"]
PERCENT_METRICS = [
    "Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
    "Max Drawdown", "VaR 95", "CVaR 95", "Best Day", "Worst Day", "Positive Days Ratio",
]


# ---------------------------------------------------------------- 3. single asset

@pytest.mark.parametrize("ticker", TICKERS)
def test_single_asset_equals_normalized_price_path(prices, ticker):
    portfolio = engine.calculate_rebalanced_portfolio(prices, {ticker: 100}, 10_000, "annually")
    expected = 10_000 * prices[ticker] / prices[ticker].iloc[0]
    np.testing.assert_allclose(portfolio["Portfolio Value"], expected, rtol=1e-12)
    np.testing.assert_allclose(
        portfolio["Daily Return"].iloc[1:], prices[ticker].pct_change().iloc[1:], rtol=1e-9, atol=1e-15
    )


@pytest.mark.parametrize("ticker", TICKERS)
def test_single_asset_identical_across_rebalancing_modes(prices, ticker):
    results = {f: engine.calculate_rebalanced_portfolio(prices, {ticker: 100}, 10_000, f) for f in FREQUENCIES}
    for frequency, frame in results.items():
        np.testing.assert_allclose(
            frame["Portfolio Value"], results["never"]["Portfolio Value"], rtol=1e-12,
            err_msg=f"{ticker}: {frequency} differs from never",
        )


@pytest.mark.parametrize("ticker", TICKERS)
def test_zero_weight_assets_do_not_change_a_single_asset_result(prices, ticker):
    padded = {t: (100 if t == ticker else 0) for t in TICKERS}
    a = engine.calculate_rebalanced_portfolio(prices, padded, 1_000, "monthly")
    b = engine.calculate_rebalanced_portfolio(prices, {ticker: 100}, 1_000, "monthly")
    np.testing.assert_allclose(a["Portfolio Value"], b["Portfolio Value"], rtol=1e-12)


# ---------------------------------------------------------------- 4. benchmark consistency

@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_100pct_spy_portfolio_matches_spy_only_benchmark(prices, frequency):
    assert BENCHMARKS["SPY only"] == {"SPY": 100}
    portfolio = engine.calculate_rebalanced_portfolio(
        prices, {"SPY": 100, "GLD": 0, "AGG": 0, "DBC": 0}, 10_000, frequency
    )
    benchmark = engine.calculate_rebalanced_portfolio(prices, BENCHMARKS["SPY only"], 10_000, frequency)
    np.testing.assert_allclose(portfolio["Portfolio Value"], benchmark["Portfolio Value"], rtol=1e-12)

    for rf in (0.0, 0.03):
        mp, mb = engine.calculate_metrics(portfolio, rf), engine.calculate_metrics(benchmark, rf)
        for key in ["Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio",
                    "Max Drawdown", "VaR 95", "CVaR 95"]:
            assert mp[key] == pytest.approx(mb[key], rel=1e-12, abs=1e-15), key


# ---------------------------------------------------------------- 6. rebalancing

@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("period", [(None, None), ("2007-10-01", "2009-03-09"), ("2019-11-15", "2021-02-10")])
def test_engine_matches_share_based_reference(prices, frequency, period):
    """Engine (return compounding) vs reference (share counts, re-bought at each rebalance)."""
    window = engine.filter_prices_by_date(prices, *period)
    result = engine.calculate_rebalanced_portfolio(window, DEFAULT_WEIGHTS, 10_000, frequency)
    reference = share_based_backtest(window, DEFAULT_WEIGHTS, 10_000, frequency)
    np.testing.assert_allclose(result["Portfolio Value"], reference.values, rtol=1e-11)


def _implied_spy_weight(prices: pd.DataFrame, portfolio: pd.DataFrame) -> pd.Series:
    """
    For a SPY/AGG portfolio the day-t return is w*r_SPY + (1-w)*r_AGG, where w is the SPY
    weight carried into day t. Solving for w reads the engine's weights from its output alone.
    """
    r = prices[["SPY", "AGG"]].pct_change()
    rp = portfolio["Daily Return"]
    spread = r["SPY"] - r["AGG"]
    implied = (rp - r["AGG"]) / spread
    return implied[spread.abs() > 2e-3].dropna()  # skip days where the two returns nearly coincide


PERIOD_CODES = {"monthly": "M", "quarterly": "Q", "annually": "Y"}


@pytest.mark.parametrize("frequency", ["monthly", "quarterly", "annually"])
def test_weights_reset_at_period_boundaries_and_drift_between(prices, frequency):
    window = engine.filter_prices_by_date(prices, "2010-01-01", "2019-12-31")
    weights = {"SPY": 60, "AGG": 40}
    portfolio = engine.calculate_rebalanced_portfolio(window, weights, 10_000, frequency)
    implied = _implied_spy_weight(window, portfolio)

    periods = window.index.to_period(PERIOD_CODES[frequency])
    first_days = window.index[1:][periods[1:] != periods[:-1]]

    reset_days = implied.index.intersection(first_days)
    assert len(reset_days) >= 0.7 * len(first_days)  # most boundary days are measurable
    np.testing.assert_allclose(implied.loc[reset_days], 0.60, atol=1e-7,
                               err_msg="weights are not reset to target on the first day of a period")

    # Between rebalances the weight must follow buy-and-hold drift exactly.
    reference = share_based_backtest(window, weights, 10_000, frequency)
    carried_in = reference.weights["SPY"].shift(1)
    carried_in.loc[reference.rebalance_dates] = 0.60
    np.testing.assert_allclose(implied, carried_in.loc[implied.index], atol=1e-7)
    drift = (implied.drop(reset_days) - 0.60).abs()
    assert drift.max() > 0.01, "weights never drift away from target between rebalances"
    assert set(reference.rebalance_dates) == set(first_days)


def test_never_rebalancing_is_pure_buy_and_hold(prices):
    window = engine.filter_prices_by_date(prices, "2016-01-01", "2019-12-31")
    portfolio = engine.calculate_rebalanced_portfolio(window, {"SPY": 60, "AGG": 40}, 10_000, "never")
    growth = window[["SPY", "AGG"]] / window[["SPY", "AGG"]].iloc[0]
    expected = 10_000 * (0.6 * growth["SPY"] + 0.4 * growth["AGG"])
    np.testing.assert_allclose(portfolio["Portfolio Value"], expected, rtol=1e-12)
    implied = _implied_spy_weight(window, portfolio)
    assert implied.iloc[-1] > 0.65  # equities outgrew bonds, so the SPY weight drifted upward


def test_small_window_by_hand(prices):
    """
    SPY/AGG 60/40, EUR 10,000, 24 Mar - 9 Apr 2021: one month boundary (1 Apr), which is also
    a quarter boundary, but no year boundary. Printed for manual inspection.
    """
    window = engine.filter_prices_by_date(prices, "2021-03-24", "2021-04-09")
    weights = {"SPY": 60, "AGG": 40}
    runs = {f: engine.calculate_rebalanced_portfolio(window, weights, 10_000, f)["Portfolio Value"] for f in FREQUENCIES}

    table = window[["SPY", "AGG"]].copy()
    for f, values in runs.items():
        table[f] = values
    print("\n" + table.round(4).to_string())

    np.testing.assert_array_equal(runs["never"], runs["annually"])      # no year boundary crossed
    np.testing.assert_array_equal(runs["monthly"], runs["quarterly"])   # 1 Apr is both
    before = window.index < "2021-04-01"
    np.testing.assert_array_equal(runs["never"][before], runs["monthly"][before])
    assert not np.isclose(runs["never"].iloc[-1], runs["monthly"].iloc[-1], rtol=1e-9)

    # Hand calculation for 1 Apr 2021 (the rebalance day) and 31 Mar (no rebalance).
    px = window[["SPY", "AGG"]]
    v0 = 10_000
    shares = {"SPY": 0.6 * v0 / px["SPY"].iloc[0], "AGG": 0.4 * v0 / px["AGG"].iloc[0]}
    v_mar31 = shares["SPY"] * px.at["2021-03-31", "SPY"] + shares["AGG"] * px.at["2021-03-31", "AGG"]
    r_apr1 = px.loc["2021-04-01"] / px.loc["2021-03-31"] - 1
    v_apr1_rebalanced = v_mar31 * (1 + 0.6 * r_apr1["SPY"] + 0.4 * r_apr1["AGG"])
    assert runs["monthly"].loc["2021-03-31"] == pytest.approx(v_mar31, rel=1e-12)
    assert runs["monthly"].loc["2021-04-01"] == pytest.approx(v_apr1_rebalanced, rel=1e-12)


def test_year_end_boundary_triggers_all_calendar_rebalances(prices):
    window = engine.filter_prices_by_date(prices, "2021-12-15", "2022-01-14")
    runs = {f: engine.calculate_rebalanced_portfolio(window, DEFAULT_WEIGHTS, 1_000, f)["Portfolio Value"] for f in FREQUENCIES}
    np.testing.assert_array_equal(runs["monthly"], runs["quarterly"])
    np.testing.assert_array_equal(runs["monthly"], runs["annually"])
    assert not np.isclose(runs["never"].iloc[-1], runs["annually"].iloc[-1], rtol=1e-9)


# ---------------------------------------------------------------- 7. initial investment scaling

@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("weights", [DEFAULT_WEIGHTS, {"SPY": 30, "GLD": 15, "AGG": 55, "DBC": 0}])
def test_initial_investment_scales_values_not_metrics(prices, frequency, weights):
    small = engine.calculate_rebalanced_portfolio(prices, weights, 1_000, frequency)
    large = engine.calculate_rebalanced_portfolio(prices, weights, 10_000, frequency)
    np.testing.assert_allclose(large["Portfolio Value"], 10 * small["Portfolio Value"], rtol=1e-12)
    np.testing.assert_allclose(large["Daily Return"].iloc[1:], small["Daily Return"].iloc[1:], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(large["Drawdown"], small["Drawdown"], atol=1e-14)

    ms, ml = engine.calculate_metrics(small, 0.02), engine.calculate_metrics(large, 0.02)
    assert ml["End Value"] == pytest.approx(10 * ms["End Value"], rel=1e-12)
    for key in PERCENT_METRICS:
        assert ml[key] == pytest.approx(ms[key], rel=1e-9, abs=1e-13), key
    assert ml["Longest Drawdown Days"] == ms["Longest Drawdown Days"]
