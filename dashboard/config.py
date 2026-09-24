"""Static configuration: asset universe, presets, benchmarks, stress periods and UI constants."""

ASSETS = {
    "SPY": {"name": "US Equities", "description": "SPDR S&P 500 ETF: large-cap US stocks"},
    "GLD": {"name": "Gold", "description": "SPDR Gold Shares: physical gold bullion"},
    "AGG": {"name": "US Bonds", "description": "iShares Core US Aggregate Bond ETF: investment-grade bonds"},
    "DBC": {"name": "Commodities", "description": "Invesco DB Commodity Index: energy, metals and agriculture futures"},
}
TICKERS = list(ASSETS)

# Portfolio presets carried over unchanged from v1 (weights in %).
DEFAULT_PRESET = "Default (60/15/15/10)"
PORTFOLIO_PRESETS = {
    DEFAULT_PRESET: {"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10},
    "All Weather (Ray Dalio)": {"SPY": 30, "GLD": 15, "AGG": 55, "DBC": 0},
    "Permanent Portfolio": {"SPY": 25, "GLD": 25, "AGG": 50, "DBC": 0},
    "Golden Butterfly": {"SPY": 40, "GLD": 20, "AGG": 40, "DBC": 0},
    "Equal Weight": {"SPY": 25, "GLD": 25, "AGG": 25, "DBC": 25},
    "Conservative": {"SPY": 30, "GLD": 10, "AGG": 55, "DBC": 5},
    "Aggressive Growth": {"SPY": 80, "GLD": 5, "AGG": 5, "DBC": 10},
}
CUSTOM_PRESET = "Custom"

BENCHMARKS = {
    "SPY only": {"SPY": 100},
    "60/40 SPY + AGG": {"SPY": 60, "AGG": 40},
    "No commodities: SPY + AGG": {"SPY": 70, "AGG": 30},
    "Equity + Gold: SPY + GLD": {"SPY": 80, "GLD": 20},
}
DEFAULT_BENCHMARK = "SPY only"

CRISIS_PERIODS = {
    "Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "COVID-19 Shock": ("2020-02-19", "2020-03-23"),
    "2022 Inflation / Energy Shock": ("2022-01-01", "2022-12-31"),
}
CRISIS_DESCRIPTIONS = {
    "Global Financial Crisis": "S&P 500 peak to the March 2009 low: credit crisis and deep recession.",
    "COVID-19 Shock": "The fastest bear market on record as the pandemic shut down economies.",
    "2022 Inflation / Energy Shock": "Rate hikes hit stocks and bonds at the same time.",
}

REBALANCE_OPTIONS = ["Never", "Monthly", "Quarterly", "Annually"]
DEFAULT_REBALANCE = "Annually"

QUICK_PERIODS = ["Full history", "YTD", "1Y", "3Y", "5Y", "10Y", *CRISIS_PERIODS, "Custom"]

ROLLING_WINDOWS = {"6M": 126, "12M": 252, "24M": 504, "36M": 756}
ROLLING_METRICS = {
    "Return": {"column": "Rolling Return", "percent": True, "title": "Rolling return"},
    "Volatility": {"column": "Rolling Volatility", "percent": True, "title": "Rolling volatility (annualized)"},
    "Sharpe": {"column": "Rolling Sharpe", "percent": False, "title": "Rolling Sharpe ratio"},
}

MC_HORIZONS = [5, 10, 20, 30]
MC_SIMULATIONS = [500, 1000, 2000]

# A backtest needs at least this many trading days to be meaningful.
MIN_TRADING_DAYS = 20

METRIC_HELP = {
    "End Value": "Final portfolio value at the end of the selected period.",
    "Total Return": "Cumulative percentage return from start to end of the period.",
    "CAGR": "Compound annual growth rate: the constant yearly return that produces the same end value.",
    "Volatility": "Annualized standard deviation of daily returns. Higher means larger swings.",
    "Sharpe": "(CAGR − risk-free rate) ÷ volatility. Return earned per unit of total risk.",
    "Sortino": "(CAGR − risk-free rate) ÷ downside deviation. Like Sharpe, but only penalizes losses.",
    "Calmar": "CAGR ÷ |maximum drawdown|. Return earned per unit of worst-case loss.",
    "Max Drawdown": "Largest peak-to-trough decline in portfolio value.",
    "Longest Drawdown": "Longest stretch of consecutive trading days spent below a previous peak.",
    "VaR": "Historical 1-day Value-at-Risk (95%): on 5% of days the loss was at least this large.",
    "CVaR": "Expected shortfall (95%): the average return on the worst 5% of days.",
    "Best Day": "Largest single-day gain.",
    "Worst Day": "Largest single-day loss.",
    "Positive Days": "Share of trading days with a positive return.",
}

# Colours: categorical asset hues validated for CVD separation on the dark surface.
ASSET_COLORS = {"SPY": "#3987e5", "GLD": "#c98500", "AGG": "#199e70", "DBC": "#d95926"}
PORTFOLIO_COLOR = "#9085e9"
BENCHMARK_COLOR = "#9aa3b2"
NEGATIVE_COLOR = "#e66767"
WARNING_COLOR = "#ec835a"
GRID_COLOR = "rgba(255,255,255,0.07)"
DIVERGING_SCALE = [
    [0.0, "#c93a3a"],
    [0.25, "#8f3b3b"],
    [0.5, "#2b2f38"],
    [0.75, "#23578f"],
    [1.0, "#3987e5"],
]
