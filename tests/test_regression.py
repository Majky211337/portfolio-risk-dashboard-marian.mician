"""
10. Regression: the v2 engine must reproduce the v1 (Dash) engine exactly.

Scenario grid: 7 allocations x 4 rebalancing rules x 5 date ranges = 140 backtests,
each compared on the full daily value/return/drawdown series and on every metric at
two risk-free rates (280 metric sets), plus rolling metrics, correlation, Monte Carlo
and the crisis table.

Intentional differences from v1 (not regressions):
- The v1 crisis table always rebalanced annually; v2 follows the user's selection, so
  the crisis comparison is made with frequency="annually".
- Sortino ratio: v2 uses the conventional downside deviation (mean of squared
  shortfalls over ALL days); v1 averaged over the downside days only. Sortino is
  therefore excluded from exact equality and instead checked against the exact
  conversion  Sortino_v1 = Sortino_v2 * sqrt(N_downside / N_all).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as new
from dashboard.config import BENCHMARKS, CRISIS_PERIODS, PORTFOLIO_PRESETS
from tests.legacy import legacy_portfolio_engine as old

WEIGHTS = [
    {"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10},
    {"SPY": 30, "GLD": 15, "AGG": 55, "DBC": 0},
    {"SPY": 25, "GLD": 25, "AGG": 25, "DBC": 25},
    {"SPY": 80, "GLD": 5, "AGG": 5, "DBC": 10},
    {"SPY": 12.34, "GLD": 40.66, "AGG": 7, "DBC": 40},
    {"SPY": 100},
    {"SPY": 60, "AGG": 40},
]
FREQUENCIES = ["never", "monthly", "quarterly", "annually"]
RANGES = [
    (None, None),
    ("2007-10-01", "2009-03-09"),
    ("2020-01-01", "2020-12-31"),
    ("2016-05-22", "2026-05-22"),
    ("2011-07-15", "2013-02-28"),
]
METRIC_KEYS = [
    "Start Value", "End Value", "Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio",
    "Calmar Ratio", "Max Drawdown", "Longest Drawdown Days", "VaR 95", "CVaR 95",
    "Best Day", "Worst Day", "Positive Days Ratio", "Positive Days", "Negative Days", "Total Trading Days",
]
SCENARIOS = list(itertools.product(range(len(WEIGHTS)), FREQUENCIES, range(len(RANGES))))


@pytest.mark.parametrize(
    "w_idx, frequency, r_idx", SCENARIOS,
    ids=[f"w{w}-{f}-r{r}" for w, f, r in SCENARIOS],
)
def test_backtest_and_metrics_identical_to_v1(prices, w_idx, frequency, r_idx):
    weights, window = WEIGHTS[w_idx], new.filter_prices_by_date(prices, *RANGES[r_idx])
    legacy = old._compute_rebalanced_portfolio(window, weights, 10_000, frequency)
    current = new.calculate_rebalanced_portfolio(window, weights, 10_000, frequency)

    assert list(legacy.columns) == list(current.columns)
    assert legacy.index.equals(current.index)
    pd.testing.assert_frame_equal(legacy, current, check_exact=True, check_names=False)

    for rf in (0.0, 0.035):
        m_old, m_new = old.calculate_metrics(legacy, rf), new.calculate_metrics(current, rf)
        for key in METRIC_KEYS:
            a, b = m_old[key], m_new[key]
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b), key
            else:
                assert a == b, f"{key}: v1 {a!r} vs v2 {b!r}"
        _assert_sortino_methodology_change(current, rf, m_old["Sortino Ratio"], m_new["Sortino Ratio"])


def _assert_sortino_methodology_change(portfolio, rf, sortino_v1, sortino_v2):
    """Documents the deliberate v2 Sortino correction as an exact relationship to v1."""
    r = portfolio["Daily Return"].dropna()
    daily_rf = (1 + rf) ** (1 / 252) - 1
    n_downside, n_all = int((r < daily_rf).sum()), len(r)
    if n_downside == 0:
        assert np.isnan(sortino_v1) and np.isnan(sortino_v2)
        return
    assert sortino_v1 == pytest.approx(sortino_v2 * np.sqrt(n_downside / n_all), rel=1e-12)
    if sortino_v2 > 0:
        # sqrt(N_downside / N_all) <= 1, so v1 never exceeded a positive v2 (equal only if every day is a loss).
        assert sortino_v2 >= sortino_v1


@pytest.mark.parametrize("window_days", [126, 252, 504, 756])
def test_rolling_metrics_identical_to_v1(prices, window_days):
    portfolio = new.calculate_rebalanced_portfolio(prices, WEIGHTS[0], 10_000, "annually")
    a = old.calculate_rolling_metrics(portfolio, window_days, 0.02).astype(float)
    b = new.calculate_rolling_metrics(portfolio, window_days, 0.02).astype(float)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_correlation_identical_to_v1(prices):
    pd.testing.assert_frame_equal(old.calculate_asset_correlation(prices), new.calculate_asset_correlation(prices))


@pytest.mark.parametrize("horizon, sims", [(5, 500), (10, 1000), (30, 2000)])
def test_monte_carlo_identical_to_v1(prices, horizon, sims):
    portfolio = new.calculate_rebalanced_portfolio(prices, WEIGHTS[0], 10_000, "annually")
    pd.testing.assert_frame_equal(
        old.monte_carlo_forecast(portfolio, horizon, sims), new.monte_carlo_forecast(portfolio, horizon, sims),
        check_exact=True,
    )


@pytest.mark.parametrize("preset", list(PORTFOLIO_PRESETS))
def test_crisis_table_matches_v1_with_annual_rebalancing(prices, preset):
    """Reproduces v1's calculate_crisis_period_summary (annual rebalancing, EUR 1,000)."""
    weights = PORTFOLIO_PRESETS[preset]
    for benchmark in BENCHMARKS.values():
        for start, end in CRISIS_PERIODS.values():
            window = new.filter_prices_by_date(prices, start, end)
            v1_portfolio = old.calculate_metrics(old._compute_rebalanced_portfolio(window, weights, 1000, "annually"))
            v1_benchmark = old.calculate_metrics(old._compute_rebalanced_portfolio(window, benchmark, 1000, "annually"))
            v2 = new.calculate_crisis_summary(prices, start, end, weights, benchmark, "annually")
            assert v2["Portfolio Return"] == pytest.approx(v1_portfolio["Total Return"], abs=1e-15)
            assert v2["Portfolio Max DD"] == pytest.approx(v1_portfolio["Max Drawdown"], abs=1e-15)
            assert v2["Benchmark Return"] == pytest.approx(v1_benchmark["Total Return"], abs=1e-15)
            assert v2["Benchmark Max DD"] == pytest.approx(v1_benchmark["Max Drawdown"], abs=1e-15)
            asset_returns = window.iloc[-1] / window.iloc[0] - 1
            assert v2["Best ETF"] == asset_returns.idxmax() and v2["Worst ETF"] == asset_returns.idxmin()
