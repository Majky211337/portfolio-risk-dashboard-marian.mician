"""5. Independent validation of every headline metric, with a dedicated Sortino analysis."""

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as engine
from tests.conftest import DEFAULT_WEIGHTS
from tests.reference import reference_metrics, share_based_backtest

FIXED_PERIOD = ("2010-01-04", "2019-12-31")
CHECKED = [
    "Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
    "Calmar Ratio", "VaR 95", "CVaR 95", "Best Day", "Worst Day",
]
SCENARIOS = [
    (DEFAULT_WEIGHTS, "annually", FIXED_PERIOD, 0.0),
    (DEFAULT_WEIGHTS, "annually", FIXED_PERIOD, 0.02),
    ({"SPY": 30, "GLD": 15, "AGG": 55, "DBC": 0}, "quarterly", ("2007-10-01", "2009-03-09"), 0.0),
    ({"SPY": 25, "GLD": 25, "AGG": 25, "DBC": 25}, "monthly", (None, None), 0.035),
    ({"SPY": 100}, "never", ("2020-01-01", "2020-12-31"), 0.01),
]


@pytest.mark.parametrize("weights, frequency, period, rf", SCENARIOS)
def test_engine_metrics_match_independent_formulas(prices, weights, frequency, period, rf):
    window = engine.filter_prices_by_date(prices, *period)
    portfolio = engine.calculate_rebalanced_portfolio(window, weights, 10_000, frequency)

    # The value series itself is re-derived independently first.
    reference_values = share_based_backtest(window, weights, 10_000, frequency).values
    np.testing.assert_allclose(portfolio["Portfolio Value"], reference_values, rtol=1e-11)

    expected = reference_metrics(reference_values, rf)
    actual = engine.calculate_metrics(portfolio, rf)
    for key in CHECKED:
        assert actual[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-12), key


def test_fixed_portfolio_reference_values(prices):
    """Human-readable record of the reference portfolio (60/15/15/10, annual, 2010-2019, rf 0)."""
    window = engine.filter_prices_by_date(prices, *FIXED_PERIOD)
    reference_values = share_based_backtest(window, DEFAULT_WEIGHTS, 10_000, "annually").values
    ref = reference_metrics(reference_values, 0.0)
    eng = engine.calculate_metrics(engine.calculate_rebalanced_portfolio(window, DEFAULT_WEIGHTS, 10_000, "annually"))
    print(f"\n{'metric':<22}{'independent':>16}{'engine':>16}{'abs diff':>12}")
    for key in CHECKED:
        print(f"{key:<22}{ref[key]:>16.10f}{eng[key]:>16.10f}{abs(ref[key] - eng[key]):>12.1e}")
    assert ref["CAGR"] == pytest.approx(eng["CAGR"], rel=1e-12)


def test_volatility_uses_sample_standard_deviation(prices):
    portfolio = engine.calculate_rebalanced_portfolio(prices, DEFAULT_WEIGHTS, 1, "annually")
    r = portfolio["Daily Return"].dropna().to_numpy()
    vol = engine.calculate_metrics(portfolio)["Annual Volatility"]
    assert vol == pytest.approx(np.std(r, ddof=1) * np.sqrt(252), rel=1e-12)
    assert vol != pytest.approx(np.std(r, ddof=0) * np.sqrt(252), rel=1e-6)


def test_var_uses_linear_interpolated_5th_percentile(prices):
    portfolio = engine.calculate_rebalanced_portfolio(prices, DEFAULT_WEIGHTS, 1, "annually")
    r = portfolio["Daily Return"].dropna().to_numpy()
    m = engine.calculate_metrics(portfolio)
    assert m["VaR 95"] == pytest.approx(np.percentile(r, 5, method="linear"), rel=1e-12)
    tail = r[r <= m["VaR 95"]]
    assert m["CVaR 95"] == pytest.approx(tail.mean(), rel=1e-12)
    assert len(tail) == pytest.approx(0.05 * len(r), abs=1)
    assert m["CVaR 95"] < m["VaR 95"] < 0


def test_cagr_uses_calendar_days(prices):
    portfolio = engine.calculate_rebalanced_portfolio(prices, DEFAULT_WEIGHTS, 1, "annually")
    v = portfolio["Portfolio Value"]
    years = (v.index[-1] - v.index[0]).days / 365.25
    m = engine.calculate_metrics(portfolio)
    assert m["CAGR"] == pytest.approx((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1, rel=1e-12)
    # Trading-day annualisation would give a visibly different number (the sample has ~252.2 days/yr).
    trading_day_cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / (len(v) - 1)) - 1
    print(f"\nCAGR calendar-day {m['CAGR']:.6%} vs trading-day {trading_day_cagr:.6%}")


def test_longest_drawdown_counts_consecutive_trading_days_below_peak():
    values = pd.Series([100, 110, 105, 104, 111, 111, 90, 95, 100, 112],
                       index=pd.bdate_range("2024-01-01", periods=10), dtype=float)
    frame = pd.DataFrame({"Portfolio Value": values, "Daily Return": values.pct_change(),
                          "Drawdown": values / values.cummax() - 1})
    m = engine.calculate_metrics(frame)
    assert m["Longest Drawdown Days"] == 3            # 90, 95, 100 below the 111 peak
    assert m["Max Drawdown"] == pytest.approx(90 / 111 - 1)
    assert m["Best Day"] == pytest.approx(112 / 100 - 1)
    assert m["Worst Day"] == pytest.approx(90 / 111 - 1)


# ---------------------------------------------------------------- Sortino deep-dive
#
# v2 uses the conventional target downside deviation (TDD):
#     daily_rf = (1 + rf) ** (1/252) - 1
#     TDD      = sqrt( mean over ALL days of min(r - daily_rf, 0)^2 ) * sqrt(252)
#     Sortino  = (CAGR - rf) / TDD
# v1 averaged the squared shortfalls over the downside days only. This was a deliberate
# methodology correction; the exact v1 -> v2 relationship is pinned below.


@pytest.mark.parametrize("rf", [0.0, 0.02, 0.05])
def test_sortino_matches_conventional_definition(prices, rf):
    window = engine.filter_prices_by_date(prices, *FIXED_PERIOD)
    portfolio = engine.calculate_rebalanced_portfolio(window, DEFAULT_WEIGHTS, 10_000, "annually")
    ref = reference_metrics(portfolio["Portfolio Value"], rf)
    assert engine.calculate_metrics(portfolio, rf)["Sortino Ratio"] == pytest.approx(ref["Sortino Ratio"], rel=1e-12)


def test_sortino_by_hand():
    """Five returns, rf = 0: shortfalls are 0, -0.02, 0, -0.01, 0 -> mean square over 5 days."""
    returns = np.array([0.03, -0.02, 0.01, -0.01, 0.02])
    values = pd.Series(100 * np.concatenate(([1.0], np.cumprod(1 + returns))),
                       index=pd.bdate_range("2024-01-01", periods=6))
    frame = pd.DataFrame({"Portfolio Value": values, "Daily Return": values.pct_change(),
                          "Drawdown": values / values.cummax() - 1})
    m = engine.calculate_metrics(frame, 0.0)
    tdd = np.sqrt((0.02 ** 2 + 0.01 ** 2) / 5) * np.sqrt(252)   # divide by ALL 5 days, not by 2
    assert tdd == pytest.approx(0.1587450787, rel=1e-9)
    assert m["Sortino Ratio"] == pytest.approx(m["CAGR"] / tdd, rel=1e-12)
    legacy_tdd = np.sqrt((0.02 ** 2 + 0.01 ** 2) / 2) * np.sqrt(252)
    assert m["Sortino Ratio"] != pytest.approx(m["CAGR"] / legacy_tdd, rel=1e-3)


def test_sortino_uses_daily_risk_free_rate_as_target(prices):
    """Days between 0 and the daily rf count as shortfalls; days above it contribute zero."""
    portfolio = engine.calculate_rebalanced_portfolio(prices, DEFAULT_WEIGHTS, 1, "annually")
    r = portfolio["Daily Return"].dropna().to_numpy()
    rf = 0.05
    daily_rf = (1 + rf) ** (1 / 252) - 1
    tdd = np.sqrt(np.mean(np.minimum(r - daily_rf, 0) ** 2)) * np.sqrt(252)
    m = engine.calculate_metrics(portfolio, rf)
    assert m["Sortino Ratio"] == pytest.approx((m["CAGR"] - rf) / tdd, rel=1e-12)
    assert ((r > 0) & (r < daily_rf)).any(), "sample should contain small positive shortfall days"


def test_sortino_v1_to_v2_relationship(prices):
    """
    v1:  DD = sqrt( sum(shortfall^2) / N_downside ) * sqrt(252)
    v2:  DD = sqrt( sum(shortfall^2) / N_all      ) * sqrt(252)
    =>   Sortino_v1 = Sortino_v2 * sqrt(N_downside / N_all)   (same CAGR numerator)
    """
    rows = []
    for label, weights, period in [
        ("60/15/15/10 2010-2019", DEFAULT_WEIGHTS, FIXED_PERIOD),
        ("60/15/15/10 full", DEFAULT_WEIGHTS, (None, None)),
        ("100% SPY full", {"SPY": 100}, (None, None)),
        ("100% AGG full", {"AGG": 100}, (None, None)),
    ]:
        window = engine.filter_prices_by_date(prices, *period)
        portfolio = engine.calculate_rebalanced_portfolio(window, weights, 10_000, "annually")
        ref = reference_metrics(portfolio["Portfolio Value"], 0.0)
        m = engine.calculate_metrics(portfolio, 0.0)
        share_down = ref["Downside Days"] / ref["Return Days"]
        assert m["Sortino Ratio"] == pytest.approx(ref["Sortino Ratio"], rel=1e-12)
        assert ref["Sortino Legacy (v1)"] == pytest.approx(m["Sortino Ratio"] * np.sqrt(share_down), rel=1e-12)
        rows.append((label, ref["Sortino Legacy (v1)"], m["Sortino Ratio"], share_down, m["Sharpe Ratio"]))

    print(f"\n{'portfolio':<24}{'v1 Sortino':>12}{'v2 Sortino':>12}{'down share':>12}{'Sharpe':>9}")
    for label, old, new, share, sharpe in rows:
        print(f"{label:<24}{old:>12.4f}{new:>12.4f}{share:>12.1%}{sharpe:>9.4f}")


def test_sortino_vs_sharpe_in_these_historical_samples(prices):
    """
    NOT a financial invariant: Sortino can be below Sharpe (e.g. negatively skewed returns
    or a negative excess return). This pins an observed property of these specific samples:
    with the v1 formula Sortino fell BELOW Sharpe here, with the corrected formula it is above.
    """
    for weights, period in [
        (DEFAULT_WEIGHTS, FIXED_PERIOD),
        (DEFAULT_WEIGHTS, (None, None)),
        ({"SPY": 100}, (None, None)),
        ({"AGG": 100}, (None, None)),
    ]:
        window = engine.filter_prices_by_date(prices, *period)
        portfolio = engine.calculate_rebalanced_portfolio(window, weights, 10_000, "annually")
        m = engine.calculate_metrics(portfolio, 0.0)
        legacy = reference_metrics(portfolio["Portfolio Value"], 0.0)["Sortino Legacy (v1)"]
        assert legacy < m["Sharpe Ratio"] < m["Sortino Ratio"], (weights, period)


def test_sortino_edge_cases():
    index = pd.bdate_range("2024-01-01", periods=6)
    rising = pd.Series([100, 101, 102, 103, 104, 105], index=index, dtype=float)
    frame = pd.DataFrame({"Portfolio Value": rising, "Daily Return": rising.pct_change(),
                          "Drawdown": rising / rising.cummax() - 1})
    m = engine.calculate_metrics(frame)
    assert np.isnan(m["Sortino Ratio"])   # no downside days -> undefined, not infinite
    assert np.isnan(m["Calmar Ratio"])    # no drawdown -> undefined
    assert m["Max Drawdown"] == 0
