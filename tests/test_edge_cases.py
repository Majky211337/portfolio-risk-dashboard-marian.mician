"""11. Edge cases: invalid inputs must be rejected clearly (engine) or handled safely (app)."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

import portfolio_engine as engine
from dashboard import charts
from dashboard.sidebar import Settings, normalized_weights

APP = str(Path(__file__).resolve().parent.parent / "app.py")


# ---------------------------------------------------------------- engine

def test_weights_not_summing_to_100_are_normalized_by_the_engine(prices):
    """The engine rescales (legacy behaviour); the app refuses such input before it gets here."""
    a = engine.calculate_rebalanced_portfolio(prices, {"SPY": 120, "AGG": 80}, 1000, "annually")
    b = engine.calculate_rebalanced_portfolio(prices, {"SPY": 60, "AGG": 40}, 1000, "annually")
    np.testing.assert_allclose(a["Portfolio Value"], b["Portfolio Value"], rtol=1e-12)


def test_zero_weights_are_allowed(prices):
    result = engine.calculate_rebalanced_portfolio(prices, {"SPY": 100, "GLD": 0, "AGG": 0, "DBC": 0}, 1000)
    assert result["Portfolio Value"].notna().all()


def test_all_zero_weights_rejected(prices):
    with pytest.raises(ValueError, match="sum to 0"):
        engine.calculate_rebalanced_portfolio(prices, {"SPY": 0, "GLD": 0, "AGG": 0, "DBC": 0}, 1000)


def test_negative_weights_rejected(prices):
    with pytest.raises(ValueError, match="negative"):
        engine.calculate_rebalanced_portfolio(prices, {"SPY": 120, "AGG": -20}, 1000)


@pytest.mark.parametrize("rows", [0, 1])
def test_insufficient_history_rejected(prices, rows):
    with pytest.raises(ValueError, match="two price observations"):
        engine.calculate_rebalanced_portfolio(prices.iloc[:rows], {"SPY": 100}, 1000)


def test_two_rows_is_the_minimum_backtest(prices):
    result = engine.calculate_rebalanced_portfolio(prices.iloc[:2], {"SPY": 100}, 1000)
    metrics = engine.calculate_metrics(result)
    assert metrics["Total Trading Days"] == 1
    assert np.isnan(metrics["Annual Volatility"])      # sample std of one return is undefined
    assert np.isnan(metrics["Sharpe Ratio"])


def test_metrics_reject_a_series_without_returns(prices):
    one_row = pd.DataFrame({"Portfolio Value": [1.0], "Daily Return": [np.nan], "Drawdown": [0.0]},
                           index=pd.DatetimeIndex(["2024-01-02"]))
    with pytest.raises(ValueError, match="daily return"):
        engine.calculate_metrics(one_row)


def test_invalid_date_string_rejected(prices):
    with pytest.raises(ValueError):
        engine.filter_prices_by_date(prices, "not-a-date", "2020-01-01")


def test_start_after_end_gives_empty_window_and_clear_error(prices):
    window = engine.filter_prices_by_date(prices, "2021-01-01", "2020-01-01")
    assert window.empty
    with pytest.raises(ValueError, match="two price observations"):
        engine.calculate_rebalanced_portfolio(window, {"SPY": 100}, 1000)


def test_window_outside_data_is_empty(prices):
    assert engine.filter_prices_by_date(prices, "1990-01-01", "1995-01-01").empty
    assert engine.calculate_crisis_summary(prices, "1990-01-01", "1995-01-01", {"SPY": 100}, {"SPY": 100}) is None


def test_empty_dataframe_rejected():
    empty = pd.DataFrame(columns=["SPY", "GLD", "AGG", "DBC"], index=pd.DatetimeIndex([]), dtype=float)
    with pytest.raises(ValueError):
        engine.calculate_rebalanced_portfolio(empty, {"SPY": 100}, 1000)


def test_missing_ticker_column_rejected(prices):
    with pytest.raises(ValueError, match="DBC"):
        engine.calculate_rebalanced_portfolio(prices.drop(columns="DBC"), {"SPY": 90, "DBC": 10}, 1000)


def test_nan_gap_in_a_held_asset_rejected(prices):
    broken = prices.copy()
    broken.iloc[100, broken.columns.get_loc("GLD")] = np.nan
    with pytest.raises(ValueError, match="gaps"):
        engine.calculate_rebalanced_portfolio(broken, {"SPY": 50, "GLD": 50}, 1000)


def test_unknown_rebalance_frequency_rejected(prices):
    with pytest.raises(ValueError, match="rebalance frequency"):
        engine.calculate_rebalanced_portfolio(prices, {"SPY": 100}, 1000, "weekly")


def test_rolling_metrics_with_insufficient_history_is_empty(prices):
    portfolio = engine.calculate_rebalanced_portfolio(prices.iloc[:100], {"SPY": 100}, 1000)
    rolling = engine.calculate_rolling_metrics(portfolio, 252)
    assert rolling.dropna(how="all").empty


def test_constant_prices_do_not_produce_inf():
    flat = pd.DataFrame({"SPY": 100.0}, index=pd.bdate_range("2024-01-01", periods=60))
    portfolio = engine.calculate_rebalanced_portfolio(flat, {"SPY": 100}, 1000)
    m = engine.calculate_metrics(portfolio)
    assert m["Total Return"] == 0 and m["Max Drawdown"] == 0
    for key in ("Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"):
        assert np.isnan(m[key]), key   # undefined -> NaN (shown as "–"), never +/-inf
    rolling = engine.calculate_rolling_metrics(portfolio, 20)
    assert not np.isinf(rolling.to_numpy(dtype=float)).any()


# ---------------------------------------------------------------- app input layer

def _settings(**weights) -> Settings:
    return Settings(weights=weights, start=None, end=None, initial_investment=1000, risk_free_rate=0,
                    rebalance="Annually", benchmark_name="SPY only", benchmark_weights={"SPY": 100})


@pytest.mark.parametrize("weights, valid", [
    ({"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10}, True),
    ({"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 9.99}, False),    # 99.99%
    ({"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10.004}, True),   # within 0.005 pp rounding tolerance
    ({"SPY": 70, "GLD": 15, "AGG": 15, "DBC": 10}, False),      # 110%
    ({"SPY": 0, "GLD": 0, "AGG": 0, "DBC": 0}, False),
    ({"SPY": 100, "GLD": 0, "AGG": 0, "DBC": 0}, True),
])
def test_app_weight_validation(weights, valid):
    assert _settings(**weights).weights_valid is valid


def test_normalize_button_logic():
    # 70/15/15/10 (110%) -> proportional 63.64/13.64/13.64/9.09 = 100.01; the -0.01 residual
    # goes to the largest weight.
    result = normalized_weights({"SPY": 70, "GLD": 15, "AGG": 15, "DBC": 10})
    assert result == {"SPY": 63.63, "GLD": 13.64, "AGG": 13.64, "DBC": 9.09}
    assert sum(result.values()) == pytest.approx(100, abs=1e-9)
    assert normalized_weights({"SPY": 0, "GLD": 0, "AGG": 0, "DBC": 0}) == {t: 25.0 for t in ("SPY", "GLD", "AGG", "DBC")}


# ---------------------------------------------------------------- charts never crash on empty input

@pytest.mark.parametrize("build", [
    lambda: charts.growth_chart(pd.DataFrame(), None, "x"),
    lambda: charts.drawdown_chart(pd.DataFrame(), None, "x"),
    lambda: charts.return_distribution_chart(pd.Series(dtype=float), np.nan, np.nan),
    lambda: charts.return_distribution_chart(pd.Series([np.inf, np.nan]), np.nan, np.nan),
    lambda: charts.rolling_chart(pd.DataFrame(columns=["Rolling Return"]), None, "Rolling Return", True, "12M"),
    lambda: charts.correlation_heatmap(pd.DataFrame()),
    lambda: charts.allocation_donut({"SPY": 0}),
    lambda: charts.monthly_returns_heatmap(pd.DataFrame()),
    lambda: charts.annual_returns_chart(pd.Series(dtype=float), None),
    lambda: charts.crisis_path_chart(pd.Series(dtype=float), pd.Series(dtype=float)),
    lambda: charts.crisis_comparison_chart({"A": None}),
    lambda: charts.monte_carlo_chart(None, pd.Series(dtype=float)),
])
def test_chart_builders_handle_empty_input(build):
    assert isinstance(build(), go.Figure)


# ---------------------------------------------------------------- full app (AppTest)

app_test = pytest.importorskip("streamlit.testing.v1")


def _run(setup=None):
    at = app_test.AppTest.from_file(APP, default_timeout=180)
    if setup:
        setup(at)
    at.run()
    if setup:  # widget values need a rerun after being set on a fresh app
        at.run()
    return at


def _assert_blocked(at, message_fragment):
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) == 0, "results were shown for invalid input"
    messages = " ".join(m.value for m in list(at.error) + list(at.warning))
    assert message_fragment in messages, messages


@pytest.mark.app
def test_app_default_renders_without_exceptions():
    at = _run()
    assert not at.exception
    assert len(at.metric) > 30 and len(at.get("plotly_chart")) >= 12


@pytest.mark.app
def test_app_rejects_weights_not_summing_to_100():
    at = app_test.AppTest.from_file(APP, default_timeout=180).run()
    at.number_input(key="spy").set_value(70.0).run()
    _assert_blocked(at, "not 100%")
    [b for b in at.button if b.label.startswith("Normalize")][0].click().run()
    assert not at.exception and len(at.metric) > 30


@pytest.mark.app
def test_app_rejects_all_zero_weights():
    at = app_test.AppTest.from_file(APP, default_timeout=180).run()
    for key in ("spy", "gld", "agg", "dbc"):
        at.number_input(key=key).set_value(0.0)
    at.run()
    _assert_blocked(at, "not 100%")
    assert all(b.disabled for b in at.button if b.label.startswith("Normalize"))


@pytest.mark.app
def test_app_weight_inputs_cannot_go_negative():
    at = app_test.AppTest.from_file(APP, default_timeout=180).run()
    assert all(at.number_input(key=k).min == 0.0 for k in ("spy", "gld", "agg", "dbc"))


@pytest.mark.app
@pytest.mark.parametrize("start, end, fragment", [
    ("2020-05-01", "2019-05-01", "start date must be before"),
    ("2020-03-02", "2020-03-13", "only 10 trading days"),
])
def test_app_rejects_bad_periods(start, end, fragment):
    from datetime import date
    at = app_test.AppTest.from_file(APP, default_timeout=180).run()
    at.date_input(key="start").set_value(date.fromisoformat(start))
    at.date_input(key="end").set_value(date.fromisoformat(end))
    at.run()
    _assert_blocked(at, fragment)


@pytest.mark.app
def test_app_ignores_malformed_url_parameters():
    at = app_test.AppTest.from_file(APP, default_timeout=180)
    at.query_params.update({"spy": "abc", "gld": "-50", "start": "2020-13-45", "reb": "Weekly", "bm": "nope"})
    at.run()
    assert not at.exception
    assert at.number_input(key="gld").value >= 0


@pytest.mark.app
def test_app_reports_missing_ticker_column(monkeypatch, prices):
    st.cache_data.clear()
    monkeypatch.setattr(engine, "load_prices", lambda *a, **k: prices.drop(columns="DBC"))
    at = app_test.AppTest.from_file(APP, default_timeout=180).run()
    st.cache_data.clear()
    _assert_blocked(at, "missing required tickers")


@pytest.mark.app
def test_app_reports_gaps_in_price_data(monkeypatch, prices):
    broken = prices.copy()
    broken.iloc[200:210, broken.columns.get_loc("AGG")] = np.nan
    st.cache_data.clear()
    monkeypatch.setattr(engine, "load_prices", lambda *a, **k: broken)
    at = app_test.AppTest.from_file(APP, default_timeout=180).run()
    st.cache_data.clear()
    _assert_blocked(at, "gaps")
