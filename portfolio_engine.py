from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Cesty k dátam
# --------------------------------------------------

DATA_PATH = Path("data/daily_adjusted_prices_common.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Načítanie a príprava dát
# --------------------------------------------------

@lru_cache(maxsize=2)
def _load_prices_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """Inner cached load — keyed on path + mtime so file changes invalidate cache."""
    prices = pd.read_csv(path_str, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices.dropna(how="any")
    return prices


def load_prices(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Načíta denné adjusted close ceny ETF zo súboru CSV.
    Cache-ovaný cez path + file mtime — neoplatí sa znovu parsovať pri každej zmene slidera.
    """
    p = Path(path)
    mtime = p.stat().st_mtime if p.exists() else 0.0
    return _load_prices_cached(str(p), mtime)


def filter_prices_by_date(
    prices: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None
) -> pd.DataFrame:
    """
    Vyfiltruje ceny podľa dátumu.
    Toto neskôr použijeme v Dash aplikácii pri date picker inpute.
    """
    filtered = prices.copy()

    if start_date is not None:
        filtered = filtered[filtered.index >= pd.to_datetime(start_date)]

    if end_date is not None:
        filtered = filtered[filtered.index <= pd.to_datetime(end_date)]

    return filtered


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočíta denné výnosy z adjusted close cien.
    """
    returns = prices.pct_change().dropna()
    return returns


# --------------------------------------------------
# 3. Váhy portfólia
# --------------------------------------------------

def normalize_weights(weights: dict) -> pd.Series:
    """
    Normalizuje váhy tak, aby ich súčet bol 100 %.
    Napríklad:
    60, 15, 15, 10 -> 0.60, 0.15, 0.15, 0.10
    """
    weights_series = pd.Series(weights, dtype=float)

    if (weights_series < 0).any():
        raise ValueError("Váhy nemôžu byť záporné.")

    total_weight = weights_series.sum()

    if total_weight == 0:
        raise ValueError("Súčet váh nesmie byť 0.")

    # Ak zadáš váhy ako 60, 15, 15, 10, premení ich na 0.60, 0.15, ...
    weights_series = weights_series / total_weight

    return weights_series


def validate_assets(prices: pd.DataFrame, weights: pd.Series) -> None:
    """
    Skontroluje, či sa všetky ETF z váh nachádzajú aj v dátach.
    """
    missing_assets = set(weights.index) - set(prices.columns)

    if missing_assets:
        raise ValueError(f"V dátach chýbajú tieto ETF: {missing_assets}")


# --------------------------------------------------
# 4. Výpočet portfólia s ročným rebalancingom
# --------------------------------------------------

@lru_cache(maxsize=16)
def _portfolio_cached(
    weights_key: tuple,
    initial_value: float,
    rebalance_frequency: str,
    start_idx: pd.Timestamp,
    end_idx: pd.Timestamp,
    data_fingerprint: int,
) -> pd.DataFrame:
    """Inner cached compute. Reloads prices fresh and filters by [start_idx, end_idx]."""
    prices = load_prices()
    prices = prices.loc[start_idx:end_idx]
    weights = dict(weights_key)
    return _compute_rebalanced_portfolio(
        prices=prices,
        weights=weights,
        initial_value=initial_value,
        rebalance_frequency=rebalance_frequency,
    )


def calculate_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: dict,
    initial_value: float = 1000.0,
    annual_rebalancing: bool = True,
    rebalance_frequency: str | None = None,
) -> pd.DataFrame:
    """
    Wrapper s cache. Pokúsi sa odhadnúť či je `prices` celý dataset (alebo časť) na základe
    min/max indexu a deleguje na `_portfolio_cached`. Ak vstupné prices nezodpovedajú raw súboru,
    fallback je priamy non-cached výpočet.
    """
    if rebalance_frequency is None:
        rebalance_frequency = "annually" if annual_rebalancing else "never"

    rebalance_frequency = str(rebalance_frequency).lower()
    if rebalance_frequency not in {"never", "monthly", "quarterly", "annually"}:
        rebalance_frequency = "annually"

    weights_series = normalize_weights(weights)
    validate_assets(prices, weights_series)

    # Try cache when caller passed a slice of load_prices() (the common path from Dash callback)
    try:
        full = load_prices()
        if (
            prices.shape[1] == full.shape[1]
            and prices.index.min() in full.index
            and prices.index.max() in full.index
            and (prices.columns == full.columns).all()
        ):
            weights_key = tuple(
                sorted((str(k), float(weights[k])) for k in weights)
            )
            fingerprint = int(DATA_PATH.stat().st_mtime) if DATA_PATH.exists() else 0
            return _portfolio_cached(
                weights_key=weights_key,
                initial_value=float(initial_value),
                rebalance_frequency=rebalance_frequency,
                start_idx=prices.index.min(),
                end_idx=prices.index.max(),
                data_fingerprint=fingerprint,
            )
    except Exception:
        pass

    return _compute_rebalanced_portfolio(
        prices=prices,
        weights=weights,
        initial_value=initial_value,
        rebalance_frequency=rebalance_frequency,
    )


def _compute_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: dict,
    initial_value: float = 1000.0,
    rebalance_frequency: str = "annually",
) -> pd.DataFrame:
    """Skutočný výpočet — bez cache."""
    weights_series = normalize_weights(weights)
    validate_assets(prices, weights_series)

    selected_assets = list(weights_series.index)
    prices = prices[selected_assets].copy()

    returns = calculate_daily_returns(prices)

    current_values = weights_series * initial_value

    records = []

    first_date = prices.index[0]

    first_record = {
        "Date": first_date,
        "Portfolio Value": initial_value,
        "Daily Return": np.nan,
    }

    for asset in selected_assets:
        first_record[f"{asset} Value"] = current_values[asset]
        first_record[f"{asset} Weight"] = weights_series[asset]

    records.append(first_record)

    previous_year = first_date.year
    previous_month = first_date.month
    previous_quarter = (first_date.month - 1) // 3

    for date, daily_asset_returns in returns.iterrows():
        current_year = date.year
        current_month = date.month
        current_quarter = (current_month - 1) // 3

        should_rebalance = False
        if rebalance_frequency == "annually" and current_year != previous_year:
            should_rebalance = True
        elif rebalance_frequency == "quarterly" and (
            current_year != previous_year or current_quarter != previous_quarter
        ):
            should_rebalance = True
        elif rebalance_frequency == "monthly" and (
            current_year != previous_year or current_month != previous_month
        ):
            should_rebalance = True

        if should_rebalance:
            total_value_before_rebalance = current_values.sum()
            current_values = weights_series * total_value_before_rebalance

        current_values = current_values * (1 + daily_asset_returns)

        portfolio_value = current_values.sum()
        actual_weights = current_values / portfolio_value

        record = {
            "Date": date,
            "Portfolio Value": portfolio_value,
        }

        for asset in selected_assets:
            record[f"{asset} Value"] = current_values[asset]
            record[f"{asset} Weight"] = actual_weights[asset]

        records.append(record)

        previous_year = current_year
        previous_month = current_month
        previous_quarter = current_quarter

    portfolio = pd.DataFrame(records).set_index("Date")

    # Iba 3 stĺpce sú v Dash aplikácii reálne použité — zvyšok by len rástol cache pamäť.
    pv = portfolio["Portfolio Value"]
    running_max = pv.cummax()

    result = pd.DataFrame(
        {
            "Portfolio Value": pv,
            "Daily Return": pv.pct_change(),
            "Drawdown": pv / running_max - 1,
        }
    )

    return result


# --------------------------------------------------
# 5. Metriky portfólia
# --------------------------------------------------

def calculate_metrics(
    portfolio: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> dict:
    """
    Vypočíta základné výkonnostné a rizikové metriky.
    """

    portfolio_values = portfolio["Portfolio Value"]
    daily_returns = portfolio["Daily Return"].dropna()

    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]

    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]

    total_return = end_value / start_value - 1

    number_of_days = (end_date - start_date).days
    number_of_years = number_of_days / 365.25

    cagr = (end_value / start_value) ** (1 / number_of_years) - 1

    annual_volatility = daily_returns.std() * np.sqrt(trading_days)

    if annual_volatility == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (cagr - risk_free_rate) / annual_volatility

    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    downside_returns = daily_returns[daily_returns < daily_rf] - daily_rf
    if len(downside_returns) > 0:
        downside_dev = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(trading_days)
        sortino_ratio = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else np.nan
    else:
        sortino_ratio = np.nan

    max_drawdown = portfolio["Drawdown"].min()

    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    var_95 = daily_returns.quantile(0.05)
    cvar_95 = daily_returns[daily_returns <= var_95].mean()

    drawdown_series = portfolio["Drawdown"]
    in_drawdown = drawdown_series < 0
    longest_dd_days = 0
    current_dd_days = 0
    for value in in_drawdown:
        if value:
            current_dd_days += 1
            longest_dd_days = max(longest_dd_days, current_dd_days)
        else:
            current_dd_days = 0

    best_day = daily_returns.max()
    worst_day = daily_returns.min()

    positive_days = (daily_returns > 0).sum()
    negative_days = (daily_returns < 0).sum()
    total_days = len(daily_returns)

    positive_days_ratio = positive_days / total_days

    metrics = {
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
        "Longest Drawdown Days": longest_dd_days,
        "VaR 95": var_95,
        "CVaR 95": cvar_95,
        "Best Day": best_day,
        "Worst Day": worst_day,
        "Positive Days Ratio": positive_days_ratio,
        "Positive Days": positive_days,
        "Negative Days": negative_days,
        "Total Trading Days": total_days,
    }

    return metrics


def calculate_rolling_metrics(
    portfolio: pd.DataFrame,
    window_days: int = 252,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Vypočíta rolling metriky (return, volatility, Sharpe) cez kĺzavé okno.

    window_days = počet obchodných dní v okne (252 ≈ 12 mesiacov)
    """
    daily_returns = portfolio["Daily Return"].dropna()

    if len(daily_returns) < window_days:
        return pd.DataFrame(
            columns=["Rolling Return", "Rolling Volatility", "Rolling Sharpe"],
            index=daily_returns.index,
        )

    rolling_return = (1 + daily_returns).rolling(window_days).apply(
        np.prod, raw=True
    ) - 1

    rolling_vol = daily_returns.rolling(window_days).std() * np.sqrt(trading_days)

    rolling_annualized_return = (1 + daily_returns).rolling(window_days).apply(
        np.prod, raw=True
    ) ** (trading_days / window_days) - 1

    rolling_sharpe = (rolling_annualized_return - risk_free_rate) / rolling_vol

    return pd.DataFrame({
        "Rolling Return": rolling_return,
        "Rolling Volatility": rolling_vol,
        "Rolling Sharpe": rolling_sharpe,
    })


MC_MAX_SIMS = 2000  # cap to keep peak memory under ~80 MB on free tier


def monte_carlo_forecast(
    portfolio: pd.DataFrame,
    horizon_years: int = 10,
    n_simulations: int = 1000,
    trading_days: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap simulácia budúcich ciest hodnoty portfolia.

    Pamäťovo šetrné:
    - float32 (50 % menej než float64)
    - n_simulations capnuté na MC_MAX_SIMS (2000) — 5000 by žralo ~300 MB pri 30Y
    - cumprod in-place

    Vracia DataFrame s percentilmi (p5, p25, p50, p75, p95) cez čas.
    """
    daily_returns = portfolio["Daily Return"].dropna().to_numpy(dtype=np.float32)
    if len(daily_returns) < 30:
        return pd.DataFrame(columns=["p5", "p25", "p50", "p75", "p95"])

    n_simulations = min(int(n_simulations), MC_MAX_SIMS)
    starting_value = float(portfolio["Portfolio Value"].iloc[-1])
    horizon_days = int(horizon_years * trading_days)

    rng = np.random.default_rng(seed)
    # int32 stačí — max hodnota je len(daily_returns) ~ 5000. Šetrí 50 % oproti int64.
    sample_idx = rng.integers(
        0, len(daily_returns), size=(n_simulations, horizon_days), dtype=np.int32
    )
    paths = daily_returns[sample_idx]  # float32
    del sample_idx  # 60+ MB voľná pamäť pred cumprod

    # In-place: (1 + r) → cumprod → scale. Žiadne intermediate kópie.
    paths += 1.0
    np.cumprod(paths, axis=1, out=paths)
    paths *= starting_value

    percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0).astype(np.float32)

    last_date = portfolio.index[-1]
    business_days = pd.bdate_range(start=last_date, periods=horizon_days + 1)

    # Prepend starting value as first day
    return pd.DataFrame(
        {
            "p5":  np.concatenate(([starting_value], percentiles[0])),
            "p25": np.concatenate(([starting_value], percentiles[1])),
            "p50": np.concatenate(([starting_value], percentiles[2])),
            "p75": np.concatenate(([starting_value], percentiles[3])),
            "p95": np.concatenate(([starting_value], percentiles[4])),
        },
        index=business_days,
    )


def calculate_asset_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočíta korelačnú maticu denných výnosov jednotlivých ETF.
    """
    returns = calculate_daily_returns(prices)
    correlation = returns.corr()
    return correlation


# --------------------------------------------------
# 6. Pomocné formátovanie výstupu
# --------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """
    Pekne vypíše metriky do terminálu.
    """
    print()
    print("METRIKY PORTFÓLIA")
    print("-" * 40)

    for key, value in metrics.items():
        if isinstance(value, pd.Timestamp):
            print(f"{key}: {value.date()}")
        elif isinstance(value, float):
            if key in [
                "Total Return",
                "CAGR",
                "Annual Volatility",
                "Max Drawdown",
                "Best Day",
                "Worst Day",
                "Positive Days Ratio",
            ]:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


# --------------------------------------------------
# 7. Testovací beh súboru
# --------------------------------------------------

if __name__ == "__main__":
    prices = load_prices()

    # Testovacie váhy - neskôr ich napojíme na slidery v Dash aplikácii
    weights = {
        "SPY": 60,
        "GLD": 15,
        "AGG": 15,
        "DBC": 10,
    }

    portfolio = calculate_rebalanced_portfolio(
        prices=prices,
        weights=weights,
        initial_value=1000,
        annual_rebalancing=True
    )

    metrics = calculate_metrics(portfolio)

    correlation = calculate_asset_correlation(prices)

    # Uloženie testovacích výstupov
    portfolio.to_csv(OUTPUT_DIR / "portfolio_backtest_example.csv")
    correlation.to_csv(OUTPUT_DIR / "asset_correlation_example.csv")

    print("Načítané ceny:")
    print(prices.head())
    print()
    print(prices.tail())

    print()
    print("Vývoj portfólia:")
    print(portfolio.head())
    print()
    print(portfolio.tail())

    print_metrics(metrics)

    print()
    print("Korelačná matica ETF:")
    print(correlation)

    print()
    print("Výstupy boli uložené do priečinka outputs.")