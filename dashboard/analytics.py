"""
Cached wrappers around portfolio_engine.

Every function takes only small, hashable arguments (dates, weight tuples,
numbers) so Streamlit's cache keys stay cheap, and a `data_version` (the CSV's
modification time) so a data refresh invalidates everything automatically.
"""

import pandas as pd
import streamlit as st

import portfolio_engine as engine

WeightsKey = tuple[tuple[str, float], ...]


def weights_key(weights: dict) -> WeightsKey:
    """Hashable, order-independent representation of a weights dict (zero weights dropped)."""
    return tuple(sorted((ticker, float(w)) for ticker, w in weights.items() if w > 0))


def get_data_version() -> float:
    return engine.DATA_PATH.stat().st_mtime if engine.DATA_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_prices(data_version: float) -> pd.DataFrame:
    return engine.load_prices()


def _prices_between(data_version: float, start, end) -> pd.DataFrame:
    return engine.filter_prices_by_date(load_prices(data_version), start, end)


@st.cache_data(show_spinner=False, max_entries=64)
def backtest(
    data_version: float,
    start,
    end,
    weights: WeightsKey,
    initial_value: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    prices = _prices_between(data_version, start, end)
    return engine.calculate_rebalanced_portfolio(
        prices, dict(weights), initial_value, rebalance_frequency.lower()
    )


@st.cache_data(show_spinner=False, max_entries=64)
def metrics(portfolio: pd.DataFrame, risk_free_rate: float) -> dict:
    return engine.calculate_metrics(portfolio, risk_free_rate=risk_free_rate)


@st.cache_data(show_spinner=False, max_entries=32)
def rolling_metrics(portfolio: pd.DataFrame, window_days: int, risk_free_rate: float) -> pd.DataFrame:
    return engine.calculate_rolling_metrics(portfolio, window_days, risk_free_rate)


@st.cache_data(show_spinner=False)
def asset_correlation(data_version: float, start, end) -> pd.DataFrame:
    return engine.calculate_asset_correlation(_prices_between(data_version, start, end))


@st.cache_data(show_spinner=False)
def asset_statistics(data_version: float, start, end, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Stand-alone buy-and-hold statistics for each ETF, using the same metric definitions."""
    prices = _prices_between(data_version, start, end)
    rows = {}
    for ticker in tickers:
        single = engine.calculate_rebalanced_portfolio(prices, {ticker: 1}, 1.0, "never")
        m = engine.calculate_metrics(single)
        rows[ticker] = {
            "Total Return": m["Total Return"],
            "CAGR": m["CAGR"],
            "Volatility": m["Annual Volatility"],
            "Sharpe": m["Sharpe Ratio"],
            "Max Drawdown": m["Max Drawdown"],
        }
    return pd.DataFrame(rows).T


@st.cache_data(show_spinner=False, max_entries=32)
def crisis_summaries(
    data_version: float,
    portfolio_weights: WeightsKey,
    benchmark_weights: WeightsKey,
    rebalance_frequency: str,
    periods: tuple[tuple[str, str, str], ...],
) -> dict:
    prices = load_prices(data_version)
    return {
        name: engine.calculate_crisis_summary(
            prices,
            start,
            end,
            dict(portfolio_weights),
            dict(benchmark_weights),
            rebalance_frequency=rebalance_frequency.lower(),
        )
        for name, start, end in periods
    }


@st.cache_data(show_spinner=False, max_entries=16)
def monte_carlo(
    portfolio: pd.DataFrame,
    horizon_years: int,
    n_simulations: int,
    starting_value: float,
    n_sample_paths: int = 40,
):
    return engine.run_monte_carlo(
        portfolio,
        horizon_years=horizon_years,
        n_simulations=n_simulations,
        starting_value=starting_value,
        n_sample_paths=n_sample_paths,
    )
