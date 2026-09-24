"""
Independent reference implementations used to validate portfolio_engine.

Nothing here imports or calls the production engine. The backtest is written as
a holdings (share-count) simulation, a different formulation from the engine's
return compounding, so agreement between the two is meaningful evidence.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def period_key(ts: pd.Timestamp, frequency: str):
    frequency = frequency.lower()
    if frequency == "monthly":
        return (ts.year, ts.month)
    if frequency == "quarterly":
        return (ts.year, (ts.month - 1) // 3)
    if frequency == "annually":
        return ts.year
    return None  # never


@dataclass
class ReferenceBacktest:
    values: pd.Series               # portfolio value at each close
    weights: pd.DataFrame           # asset weights at each close (after drift, before any rebalance)
    rebalance_dates: list           # dates whose return is earned from freshly reset target weights


def share_based_backtest(prices: pd.DataFrame, weights: dict, initial_value: float, frequency: str) -> ReferenceBacktest:
    """
    Buy shares at target weights on day 0. On the first trading day of each new
    period, re-buy target weights using the previous close (so that day's return
    is earned by the target weights). Otherwise hold the share counts unchanged.
    """
    total = sum(weights.values())
    target = {t: w / total for t, w in weights.items()}
    tickers = list(target)
    px = prices[tickers].to_numpy(dtype=float)
    dates = prices.index
    target_vec = np.array([target[t] for t in tickers])

    shares = target_vec * initial_value / px[0]
    values = np.empty(len(dates))
    weights_at_close = np.empty((len(dates), len(tickers)))
    values[0] = initial_value
    weights_at_close[0] = target_vec
    rebalance_dates = []

    for i in range(1, len(dates)):
        new_period = period_key(dates[i], frequency) != period_key(dates[i - 1], frequency)
        if frequency.lower() != "never" and new_period:
            value_prev_close = float(shares @ px[i - 1])
            shares = target_vec * value_prev_close / px[i - 1]
            rebalance_dates.append(dates[i])
        holdings = shares * px[i]
        values[i] = holdings.sum()
        weights_at_close[i] = holdings / values[i]

    return ReferenceBacktest(
        values=pd.Series(values, index=dates),
        weights=pd.DataFrame(weights_at_close, index=dates, columns=tickers),
        rebalance_dates=rebalance_dates,
    )


def reference_metrics(values: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """Metric formulas written from scratch with NumPy (no pandas statistics, no engine calls)."""
    v = values.to_numpy(dtype=float)
    dates = values.index
    r = v[1:] / v[:-1] - 1

    total_return = v[-1] / v[0] - 1
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (v[-1] / v[0]) ** (1 / years) - 1
    volatility = np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)

    # Sortino, conventional target downside deviation (the v2 definition), step by step:
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1       # 1. annual rf -> daily rf
    downside = [min(x - daily_rf, 0.0) for x in r]                  # 2. min(r - daily_rf, 0) for EVERY day
    squared = [d * d for d in downside]                             # 3. square
    mean_all_days = sum(squared) / len(squared)                     # 4. mean over ALL days (zeros included)
    dd_conventional = mean_all_days ** 0.5 * TRADING_DAYS ** 0.5    # 5-6. sqrt, annualize by sqrt(252)

    # v1 legacy definition, kept for documentation: mean over the downside days only.
    shortfalls = r[r < daily_rf] - daily_rf
    dd_legacy = np.sqrt(np.sum(shortfalls ** 2) / len(shortfalls)) * np.sqrt(TRADING_DAYS)
    arithmetic_excess = (np.mean(r) - daily_rf) * TRADING_DAYS

    running_max = np.maximum.accumulate(v)
    drawdown = v / running_max - 1
    max_drawdown = drawdown.min()

    var_95 = np.percentile(r, 5)  # linear interpolation between order statistics
    cvar_95 = r[r <= var_95].mean()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": volatility,
        "Sharpe Ratio": (cagr - risk_free_rate) / volatility,
        "Sortino Ratio": (cagr - risk_free_rate) / dd_conventional,
        "Sortino Legacy (v1)": (cagr - risk_free_rate) / dd_legacy,
        "Sortino (arith. mean numerator)": arithmetic_excess / dd_conventional,
        "Downside Deviation (legacy)": dd_legacy,
        "Downside Deviation (conventional)": dd_conventional,
        "Downside Days": len(shortfalls),
        "Return Days": len(r),
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": cagr / abs(max_drawdown),
        "VaR 95": var_95,
        "CVaR 95": cvar_95,
        "Best Day": r.max(),
        "Worst Day": r.min(),
    }
