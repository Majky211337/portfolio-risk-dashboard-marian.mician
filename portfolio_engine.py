"""
Portfolio calculation engine.

Pure pandas / NumPy logic with no UI dependencies: price loading, rebalanced
backtests, performance & risk metrics, rolling analytics, crisis stress tests,
calendar returns and bootstrap Monte Carlo simulation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "daily_adjusted_prices_common.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TRADING_DAYS = 252
REBALANCE_FREQUENCIES = ("never", "monthly", "quarterly", "annually")


# --------------------------------------------------
# 1. Data loading
# --------------------------------------------------

def load_prices(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load daily adjusted close prices, keeping only dates where every ETF has a price."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices.dropna(how="any")
    return prices


def filter_prices_by_date(
    prices: pd.DataFrame,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Return the rows between start_date and end_date (both inclusive)."""
    filtered = prices.copy()

    if start_date is not None:
        filtered = filtered[filtered.index >= pd.to_datetime(start_date)]

    if end_date is not None:
        filtered = filtered[filtered.index <= pd.to_datetime(end_date)]

    return filtered


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from adjusted close prices."""
    return prices.pct_change().dropna()


# --------------------------------------------------
# 2. Portfolio weights
# --------------------------------------------------

def normalize_weights(weights: dict) -> pd.Series:
    """
    Scale weights so they sum to 1.

    Example: {SPY: 60, GLD: 15, AGG: 15, DBC: 10} -> 0.60, 0.15, 0.15, 0.10
    """
    weights_series = pd.Series(weights, dtype=float)

    if (weights_series < 0).any():
        raise ValueError("Weights cannot be negative.")

    total_weight = weights_series.sum()

    if total_weight == 0:
        raise ValueError("Weights must not sum to 0.")

    return weights_series / total_weight


def validate_assets(prices: pd.DataFrame, weights: pd.Series) -> None:
    """Ensure every weighted ETF is present in the price data."""
    missing_assets = set(weights.index) - set(prices.columns)

    if missing_assets:
        raise ValueError(f"Price data is missing these ETFs: {sorted(missing_assets)}")


# --------------------------------------------------
# 3. Rebalanced portfolio backtest
# --------------------------------------------------

def _rebalance_period_keys(index: pd.DatetimeIndex, rebalance_frequency: str) -> np.ndarray:
    """Label each date with its rebalancing period; a new label starts a new period."""
    if rebalance_frequency == "annually":
        return index.year.to_numpy()
    if rebalance_frequency == "quarterly":
        return index.year.to_numpy() * 4 + (index.month.to_numpy() - 1) // 3
    if rebalance_frequency == "monthly":
        return index.year.to_numpy() * 12 + index.month.to_numpy()
    return np.zeros(len(index), dtype=int)


def calculate_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: dict,
    initial_value: float = 1000.0,
    rebalance_frequency: str = "annually",
) -> pd.DataFrame:
    """
    Backtest a buy-and-hold portfolio that is reset to its target weights on the
    first trading day of each new month / quarter / year (or never).

    Rebalancing happens at the previous close, i.e. before that day's return is
    applied. Between rebalancing dates each holding drifts with its own returns.

    Returns a DataFrame indexed by date with "Portfolio Value", "Daily Return"
    and "Drawdown" columns.
    """
    rebalance_frequency = str(rebalance_frequency).lower()
    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(
            f"Unknown rebalance frequency '{rebalance_frequency}'. "
            f"Expected one of {REBALANCE_FREQUENCIES}."
        )

    weights_series = normalize_weights(weights)
    validate_assets(prices, weights_series)

    if len(prices) < 2:
        raise ValueError("At least two price observations are required.")

    prices = prices[list(weights_series.index)]
    returns = calculate_daily_returns(prices)
    target_weights = weights_series.to_numpy()

    # A period boundary exists wherever a return date belongs to a different
    # period than the trading day before it.
    if len(returns) != len(prices) - 1:
        raise ValueError("Price data contains gaps (NaN values) for the selected assets.")
    period_keys = _rebalance_period_keys(prices.index, rebalance_frequency)
    is_new_period = period_keys[1:] != period_keys[:-1]
    segment_starts = np.flatnonzero(is_new_period)
    segment_bounds = np.concatenate(([0], segment_starts[segment_starts > 0], [len(returns)]))

    growth = (1.0 + returns).to_numpy()
    asset_values = np.empty_like(growth)
    segment_start_value = float(initial_value)

    for start, end in zip(segment_bounds[:-1], segment_bounds[1:]):
        # Same multiplication order as a day-by-day loop: ((w * V) * g1) * g2 ...
        segment = growth[start:end].copy()
        segment[0] *= target_weights * segment_start_value
        np.cumprod(segment, axis=0, out=segment)
        asset_values[start:end] = segment
        segment_start_value = segment[-1].sum()

    portfolio_values = np.concatenate(([float(initial_value)], asset_values.sum(axis=1)))
    pv = pd.Series(portfolio_values, index=prices.index, name="Portfolio Value")
    pv.index.name = "Date"

    return pd.DataFrame(
        {
            "Portfolio Value": pv,
            "Daily Return": pv.pct_change(),
            "Drawdown": pv / pv.cummax() - 1,
        }
    )


# --------------------------------------------------
# 4. Performance & risk metrics
# --------------------------------------------------

def calculate_longest_drawdown_days(drawdown: pd.Series) -> int:
    """Longest run of consecutive trading days spent below a previous peak."""
    longest = 0
    current = 0
    for is_below_peak in (drawdown < 0):
        if is_below_peak:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def calculate_metrics(
    portfolio: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """
    Performance and risk statistics for a backtested portfolio.

    Methodology:
    - CAGR uses calendar time (days / 365.25).
    - Volatility = std of daily returns * sqrt(252).
    - Sharpe = (CAGR - rf) / volatility.
    - Sortino = (CAGR - rf) / downside deviation, using the conventional target
      downside deviation: sqrt(mean(min(r - daily_rf, 0)^2)) * sqrt(252), where the
      mean runs over ALL trading days (days above the target contribute zero).
      Changed in v2: v1 averaged over the downside days only, which understated
      Sortino by a factor of sqrt(share of downside days).
    - Calmar = CAGR / |max drawdown|.
    - VaR / CVaR are historical 95 % one-day figures (5th percentile of daily
      returns and the mean of returns at or below it).
    """
    portfolio_values = portfolio["Portfolio Value"]
    daily_returns = portfolio["Daily Return"].dropna()

    if len(daily_returns) == 0:
        raise ValueError("At least one daily return is required to compute metrics.")

    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]

    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]

    total_return = end_value / start_value - 1

    number_of_years = (end_date - start_date).days / 365.25
    cagr = (end_value / start_value) ** (1 / number_of_years) - 1 if number_of_years > 0 else np.nan

    annual_volatility = daily_returns.std() * np.sqrt(trading_days)

    if annual_volatility == 0 or pd.isna(annual_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (cagr - risk_free_rate) / annual_volatility

    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    shortfalls = np.minimum(daily_returns - daily_rf, 0.0)
    downside_dev = np.sqrt((shortfalls ** 2).mean()) * np.sqrt(trading_days)
    sortino_ratio = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else np.nan

    max_drawdown = portfolio["Drawdown"].min()
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    var_95 = daily_returns.quantile(0.05)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()

    positive_days = int((daily_returns > 0).sum())
    negative_days = int((daily_returns < 0).sum())
    total_days = len(daily_returns)

    return {
        "Start Date": start_date,
        "End Date": end_date,
        "Start Value": start_value,
        "End Value": end_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown": max_drawdown,
        "Longest Drawdown Days": calculate_longest_drawdown_days(portfolio["Drawdown"]),
        "VaR 95": var_95,
        "CVaR 95": cvar_95,
        "Best Day": daily_returns.max(),
        "Worst Day": daily_returns.min(),
        "Positive Days Ratio": positive_days / total_days,
        "Positive Days": positive_days,
        "Negative Days": negative_days,
        "Total Trading Days": total_days,
    }


def calculate_rolling_metrics(
    portfolio: pd.DataFrame,
    window_days: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Rolling return, annualized volatility and Sharpe over a trailing window.

    window_days is in trading days (252 ≈ 12 months). The rolling Sharpe uses the
    window's annualized compounded return, mirroring the full-period Sharpe.
    """
    daily_returns = portfolio["Daily Return"].dropna()
    columns = ["Rolling Return", "Rolling Volatility", "Rolling Sharpe"]

    if len(daily_returns) < window_days:
        return pd.DataFrame(columns=columns, index=daily_returns.index, dtype=float)

    rolling_growth = (1 + daily_returns).rolling(window_days).apply(np.prod, raw=True)
    rolling_return = rolling_growth - 1
    rolling_vol = daily_returns.rolling(window_days).std() * np.sqrt(trading_days)
    rolling_annualized_return = rolling_growth ** (trading_days / window_days) - 1
    rolling_sharpe = (rolling_annualized_return - risk_free_rate) / rolling_vol

    return pd.DataFrame(
        {
            "Rolling Return": rolling_return,
            "Rolling Volatility": rolling_vol,
            "Rolling Sharpe": rolling_sharpe.replace([np.inf, -np.inf], np.nan),
        }
    )


def calculate_asset_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of the ETFs' daily returns."""
    return calculate_daily_returns(prices).corr()


def calculate_period_returns(portfolio: pd.DataFrame, frequency: str = "ME") -> pd.Series:
    """
    Compounded returns per calendar period ("ME" = month, "YE" = year).

    The first period is measured from the backtest's starting value, so a
    partial first month / year is included rather than dropped.
    """
    values = portfolio["Portfolio Value"]
    period_end_values = values.resample(frequency).last().dropna()
    previous_values = period_end_values.shift(1)
    previous_values.iloc[0] = values.iloc[0]
    return period_end_values / previous_values - 1


def calculate_monthly_returns_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Year x month table of monthly returns plus a compounded "Year" column.

    Month-over-month returns are measured between month-end values, so the
    first (partial) month of the backtest is not shown, as in v1.
    """
    monthly = portfolio["Portfolio Value"].resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()

    if monthly_returns.empty:
        return pd.DataFrame()

    frame = monthly_returns.to_frame(name="Return")
    frame["Year"] = frame.index.year
    frame["Month"] = frame.index.month

    table = frame.pivot(index="Year", columns="Month", values="Return")
    table = table.reindex(columns=range(1, 13))
    table["Year"] = (1 + table).prod(axis=1, min_count=1) - 1
    return table


# --------------------------------------------------
# 5. Crisis stress testing
# --------------------------------------------------

def calculate_crisis_summary(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    portfolio_weights: dict,
    benchmark_weights: dict,
    rebalance_frequency: str = "annually",
    initial_value: float = 1000.0,
    min_observations: int = 5,
) -> dict | None:
    """
    Portfolio vs benchmark behaviour inside a historical stress window.

    Returns None when the window has fewer than `min_observations` trading days
    in the data.
    """
    period_prices = filter_prices_by_date(prices, start_date, end_date)

    if len(period_prices) < min_observations:
        return None

    portfolio = calculate_rebalanced_portfolio(
        period_prices, portfolio_weights, initial_value, rebalance_frequency
    )
    benchmark = calculate_rebalanced_portfolio(
        period_prices, benchmark_weights, initial_value, rebalance_frequency
    )

    portfolio_metrics = calculate_metrics(portfolio)
    benchmark_metrics = calculate_metrics(benchmark)
    asset_returns = period_prices.iloc[-1] / period_prices.iloc[0] - 1

    return {
        "Start": period_prices.index[0],
        "End": period_prices.index[-1],
        "Trading Days": len(period_prices) - 1,
        "Portfolio Return": portfolio_metrics["Total Return"],
        "Portfolio Max DD": portfolio_metrics["Max Drawdown"],
        "Portfolio Volatility": portfolio_metrics["Annual Volatility"],
        "Benchmark Return": benchmark_metrics["Total Return"],
        "Benchmark Max DD": benchmark_metrics["Max Drawdown"],
        "Benchmark Volatility": benchmark_metrics["Annual Volatility"],
        "Best ETF": asset_returns.idxmax(),
        "Worst ETF": asset_returns.idxmin(),
        "Asset Returns": asset_returns,
        "Portfolio Path": portfolio["Portfolio Value"] / initial_value,
        "Benchmark Path": benchmark["Portfolio Value"] / initial_value,
    }


# --------------------------------------------------
# 6. Monte Carlo simulation
# --------------------------------------------------

MC_MAX_SIMS = 2000  # upper bound that keeps a 30Y run at roughly 120 MB peak memory
MC_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class MonteCarloResult:
    percentiles: pd.DataFrame     # p5 ... p95 per future business day
    terminal_values: np.ndarray   # one ending value per simulated path
    sample_paths: pd.DataFrame    # a handful of individual paths for display
    starting_value: float
    horizon_years: int
    n_simulations: int


def run_monte_carlo(
    portfolio: pd.DataFrame,
    horizon_years: int = 10,
    n_simulations: int = 1000,
    starting_value: float | None = None,
    n_sample_paths: int = 0,
    trading_days: int = TRADING_DAYS,
    seed: int = 42,
) -> MonteCarloResult | None:
    """
    Bootstrap simulation of future portfolio value.

    Each path resamples (with replacement) the portfolio's realised daily
    returns, so fat tails present in history are preserved without assuming
    normality. float32 and in-place cumprod keep memory low.

    starting_value defaults to the backtest's final value. Returns None when
    fewer than 30 daily returns are available.
    """
    daily_returns = portfolio["Daily Return"].dropna().to_numpy(dtype=np.float32)
    if len(daily_returns) < 30:
        return None

    n_simulations = min(int(n_simulations), MC_MAX_SIMS)
    if starting_value is None:
        starting_value = float(portfolio["Portfolio Value"].iloc[-1])
    horizon_days = int(horizon_years * trading_days)

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(
        0, len(daily_returns), size=(n_simulations, horizon_days), dtype=np.int32
    )
    paths = daily_returns[sample_idx]
    del sample_idx

    paths += 1.0
    np.cumprod(paths, axis=1, out=paths)
    paths *= starting_value

    percentiles = np.percentile(paths, MC_PERCENTILES, axis=0).astype(np.float32)
    dates = pd.bdate_range(start=portfolio.index[-1], periods=horizon_days + 1)

    percentile_frame = pd.DataFrame(
        {
            f"p{p}": np.concatenate(([starting_value], percentiles[i]))
            for i, p in enumerate(MC_PERCENTILES)
        },
        index=dates,
    )

    n_sample_paths = min(n_sample_paths, n_simulations)
    sample_paths = pd.DataFrame(
        np.column_stack(
            [np.full(n_sample_paths, starting_value, dtype=np.float32), paths[:n_sample_paths]]
        ).T,
        index=dates,
    ) if n_sample_paths else pd.DataFrame(index=dates)

    return MonteCarloResult(
        percentiles=percentile_frame,
        terminal_values=paths[:, -1].astype(np.float64),
        sample_paths=sample_paths,
        starting_value=float(starting_value),
        horizon_years=int(horizon_years),
        n_simulations=n_simulations,
    )


def monte_carlo_forecast(
    portfolio: pd.DataFrame,
    horizon_years: int = 10,
    n_simulations: int = 1000,
    trading_days: int = TRADING_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """Percentile fan (p5, p25, p50, p75, p95) of simulated portfolio value over time."""
    result = run_monte_carlo(
        portfolio,
        horizon_years=horizon_years,
        n_simulations=n_simulations,
        trading_days=trading_days,
        seed=seed,
    )
    if result is None:
        return pd.DataFrame(columns=[f"p{p}" for p in MC_PERCENTILES])
    return result.percentiles


def summarize_monte_carlo(result: MonteCarloResult) -> dict:
    """Headline statistics of the simulated ending-value distribution."""
    terminal = result.terminal_values
    years = result.horizon_years
    median = float(np.median(terminal))
    return {
        "Median Ending Value": median,
        "P5 Ending Value": float(np.percentile(terminal, 5)),
        "P95 Ending Value": float(np.percentile(terminal, 95)),
        "Median CAGR": (median / result.starting_value) ** (1 / years) - 1,
        "Probability of Loss": float((terminal < result.starting_value).mean()),
        "Probability of Doubling": float((terminal >= 2 * result.starting_value).mean()),
    }


# --------------------------------------------------
# 7. Command-line smoke test
# --------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dictionary to the terminal."""
    percent_keys = {
        "Total Return", "CAGR", "Annual Volatility", "Max Drawdown",
        "Best Day", "Worst Day", "Positive Days Ratio", "VaR 95", "CVaR 95",
    }
    print("\nPORTFOLIO METRICS")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, pd.Timestamp):
            print(f"{key}: {value.date()}")
        elif isinstance(value, float) and key in percent_keys:
            print(f"{key}: {value:.2%}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    prices = load_prices()
    weights = {"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10}

    portfolio = calculate_rebalanced_portfolio(prices, weights, 1000, "annually")
    correlation = calculate_asset_correlation(prices)

    OUTPUT_DIR.mkdir(exist_ok=True)
    portfolio.to_csv(OUTPUT_DIR / "portfolio_backtest_example.csv")
    correlation.to_csv(OUTPUT_DIR / "asset_correlation_example.csv")

    print(portfolio.tail())
    print_metrics(calculate_metrics(portfolio))
    print("\nAsset correlation:")
    print(correlation.round(3))
