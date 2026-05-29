import base64
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

from portfolio_engine import (
    load_prices,
    filter_prices_by_date,
    calculate_rebalanced_portfolio,
    calculate_metrics,
    calculate_asset_correlation,
    calculate_rolling_metrics,
    monte_carlo_forecast,
)


# --------------------------------------------------
# 1. Načítanie dát
# --------------------------------------------------

prices = load_prices()

MIN_DATE = prices.index.min().date()
MAX_DATE = prices.index.max().date()

ASSETS = ["SPY", "GLD", "AGG", "DBC"]

COLOR_MAP = {
    "SPY": "#6366f1",
    "GLD": "#f59e0b",
    "AGG": "#10b981",
    "DBC": "#a855f7",
}

LINE_COLOR = "#6366f1"
BENCHMARK_COLOR = "#94a3b8"
DRAWDOWN_COLOR = "#ef4444"


def chart_theme(theme):
    """Returns dict of theme-aware colors for plotly charts."""
    if theme == "dark":
        return {
            "template": "plotly_dark",
            "grid": "#334155",
            "axis_font": "#cbd5e1",
            "title_font": "#f1f5f9",
            "annotation_font": "#94a3b8",
            "heatmap_mid": "#1e293b",
        }
    return {
        "template": "plotly_white",
        "grid": "#e5e7eb",
        "axis_font": "#475569",
        "title_font": "#0f172a",
        "annotation_font": "#475569",
        "heatmap_mid": "#f8fafc",
    }

BENCHMARKS = {
    "SPY only": {
        "SPY": 100,
    },
    "60/40 SPY + AGG": {
        "SPY": 60,
        "AGG": 40,
    },
    "No commodities: SPY + AGG": {
        "SPY": 70,
        "AGG": 30,
    },
    "Equity + Gold: SPY + GLD": {
        "SPY": 80,
        "GLD": 20,
    },

}

CRISIS_PERIODS = {
    "Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "COVID-19 Shock": ("2020-02-19", "2020-03-23"),
    "2022 Inflation / Energy Shock": ("2022-01-01", "2022-12-31"),
}

PERIOD_PRESETS = {
    "Full Sample": (str(MIN_DATE), str(MAX_DATE)),
    "Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "COVID-19 Shock": ("2020-02-19", "2020-03-23"),
    "2022 Inflation / Energy Shock": ("2022-01-01", "2022-12-31"),
}

DATE_QUICK_RANGES = ["YTD", "1Y", "3Y", "5Y", "10Y", "MAX"]


def compute_quick_range(label: str, max_date) -> tuple:
    """Returns (start_date, end_date) for a quick range label."""
    end = pd.to_datetime(max_date)
    if label == "MAX":
        return None, end.strftime("%Y-%m-%d")
    if label == "YTD":
        return f"{end.year}-01-01", end.strftime("%Y-%m-%d")
    years_map = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}
    years = years_map.get(label, 1)
    start = end - pd.DateOffset(years=years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


PORTFOLIO_PRESETS = {
    "Default (60/15/15/10)": {"SPY": 60, "GLD": 15, "AGG": 15, "DBC": 10},
    "All Weather (Ray Dalio)": {"SPY": 30, "GLD": 15, "AGG": 55, "DBC": 0},
    "Permanent Portfolio": {"SPY": 25, "GLD": 25, "AGG": 50, "DBC": 0},
    "Golden Butterfly": {"SPY": 40, "GLD": 20, "AGG": 40, "DBC": 0},
    "Equal Weight": {"SPY": 25, "GLD": 25, "AGG": 25, "DBC": 25},
    "Conservative": {"SPY": 30, "GLD": 10, "AGG": 55, "DBC": 5},
    "Aggressive Growth": {"SPY": 80, "GLD": 5, "AGG": 5, "DBC": 10},
}

REBALANCE_FREQUENCIES = ["Never", "Monthly", "Quarterly", "Annually"]

ROLLING_WINDOWS = {
    "6M": 126,
    "12M": 252,
    "24M": 504,
    "36M": 756,
}

MONTE_CARLO_HORIZONS = ["5Y", "10Y", "20Y", "30Y"]
MONTE_CARLO_SIMS = ["500", "1000", "5000"]


ROLLING_METRICS = {
    "Return": {
        "column": "Rolling Return",
        "title_suffix": "Rolling Return",
        "y_tickformat": ".0%",
        "hover_format": ".2%",
        "is_percent": True,
        "zero_baseline": True,
    },
    "Volatility": {
        "column": "Rolling Volatility",
        "title_suffix": "Rolling Volatility (annualized)",
        "y_tickformat": ".0%",
        "hover_format": ".2%",
        "is_percent": True,
        "zero_baseline": False,
    },
    "Sharpe": {
        "column": "Rolling Sharpe",
        "title_suffix": "Rolling Sharpe Ratio",
        "y_tickformat": ".2f",
        "hover_format": ".2f",
        "is_percent": False,
        "zero_baseline": True,
    },
}

METRIC_TOOLTIPS = {
    "End Value": "Final portfolio value at the end of the selected period.",
    "Total Return": "Cumulative percentage return from start to end of the period.",
    "CAGR": "Compound Annual Growth Rate — the constant yearly return that would produce the same end value.",
    "Volatility": "Annualized standard deviation of daily returns. Higher = more variability.",
    "Sharpe Ratio": "Risk-adjusted return: (CAGR − risk-free rate) / volatility. Higher is better.",
    "Sortino Ratio": "Like Sharpe, but only penalizes downside volatility. Higher is better.",
    "Calmar Ratio": "CAGR divided by absolute Max Drawdown. Reward per unit of worst-case loss.",
    "Max Drawdown": "Largest peak-to-trough decline observed during the period.",
    "Longest DD": "Number of trading days the portfolio spent below a previous peak.",
    "VaR 95%": "Daily Value-at-Risk: on the worst 5% of days, you lose at least this much.",
    "CVaR 95%": "Average loss on the worst 5% of days (expected shortfall).",
    "Best Day": "Largest single-day gain in the period.",
    "Worst Day": "Largest single-day loss in the period.",
}

# --------------------------------------------------
# 2. Pomocné formátovanie
# --------------------------------------------------

def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f} €".replace(",", " ")


def format_slider_eur(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f} €".replace(",", " ")


def format_slider_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def make_sparkline(values, color="#6366f1", width=140, height=32, fill=True):
    """Generate inline SVG sparkline as html.Img with base64 data URI."""
    if values is None:
        return None

    series = [v for v in values if v is not None and not pd.isna(v)]
    if len(series) < 2:
        return None

    if len(series) > 80:
        step = max(1, len(series) // 80)
        series = series[::step] + [series[-1]]

    vmin = min(series)
    vmax = max(series)
    rng = vmax - vmin if vmax > vmin else 1

    pad = 2
    n = len(series)
    pts = []
    for i, v in enumerate(series):
        x = pad + (i / (n - 1)) * (width - 2 * pad)
        y = pad + (1 - (v - vmin) / rng) * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(pts)
    fill_attr = ""
    if fill:
        last_x = pad + (width - 2 * pad)
        fill_attr = (
            f'<polygon points="{pad},{height - pad} {polyline} {last_x},{height - pad}" '
            f'fill="{color}" fill-opacity="0.18"/>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">'
        f'{fill_attr}'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return html.Img(
        src=f"data:image/svg+xml;base64,{encoded}",
        className="metric-sparkline",
    )


SPARKLINE_COLORS = {
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral": "#6366f1",
}


def metric_card(title, value, subtitle, tone="neutral", tooltip=None, sparkline=None):
    title_children = [html.Span(title)]
    if tooltip:
        title_children.append(
            html.Span("i", className="metric-info", title=tooltip)
        )

    value_class = "metric-value"
    if tone in ("positive", "negative"):
        value_class += f" {tone}"

    children = [
        html.Div(title_children, className="metric-title"),
        html.Div(value, className=value_class),
        html.Div(subtitle, className="metric-subtitle"),
    ]
    if sparkline is not None:
        children.append(sparkline)

    return html.Div(children=children, className="metric-card")


def tone_for(value, positive_good=True):
    """Returns 'positive'/'negative'/'neutral' based on sign of value."""
    if value is None or pd.isna(value):
        return "neutral"
    if value > 0:
        return "positive" if positive_good else "negative"
    if value < 0:
        return "negative" if positive_good else "positive"
    return "neutral"


def slider_block(asset, default_value):
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Label(f"{asset} weight", className="slider-label"),
                    html.Span(
                        id=f"{asset.lower()}-weight-label",
                        className="slider-value-label",
                    ),
                ],
                className="slider-header",
            ),
            dcc.Slider(
                id=f"{asset.lower()}-weight",
                min=0,
                max=100,
                step=0.01,
                value=default_value,
                marks={
                    0: "0%",
                    25: "25%",
                    50: "50%",
                    75: "75%",
                    100: "100%",
                },
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="slider-block",
    )


def investment_slider_block(default_value=10000):
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Label("Initial investment", className="slider-label"),
                    html.Span(
                        id="initial-investment-label",
                        className="slider-value-label",
                    ),
                ],
                className="slider-header",
            ),
            dcc.Slider(
                id="initial-investment",
                min=1000,
                max=100000,
                step=500,
                value=default_value,
                marks={
                    1000: "1k",
                    25000: "25k",
                    50000: "50k",
                    75000: "75k",
                    100000: "100k",
                },
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="slider-block",
    )


def risk_free_rate_slider_block(default_value=0.0):
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Label("Risk-free rate", className="slider-label"),
                    html.Span(
                        id="risk-free-rate-label",
                        className="slider-value-label",
                    ),
                ],
                className="slider-header",
            ),
            dcc.Slider(
                id="risk-free-rate",
                min=0,
                max=6,
                step=0.10,
                value=default_value,
                marks={
                    0: "0%",
                    1: "1%",
                    2: "2%",
                    3: "3%",
                    4: "4%",
                    5: "5%",
                    6: "6%",
                },
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="slider-block",
    )


def benchmark_dropdown_block(default_value="SPY only"):
    return html.Div(
        children=[
            html.Label("Benchmark", className="slider-label"),
            dcc.Dropdown(
                id="benchmark-selector",
                options=[
                    {"label": benchmark_name, "value": benchmark_name}
                    for benchmark_name in BENCHMARKS.keys()
                ],
                value=default_value,
                clearable=False,
                className="benchmark-dropdown",
            ),
        ],
        className="slider-block",
    )

def portfolio_preset_dropdown_block(default_value="Default (60/15/15/10)"):
    return html.Div(
        children=[
            html.Label("Portfolio preset", className="slider-label"),
            dcc.Dropdown(
                id="portfolio-preset",
                options=[
                    {"label": name, "value": name}
                    for name in PORTFOLIO_PRESETS.keys()
                ],
                value=default_value,
                clearable=False,
                className="benchmark-dropdown",
            ),
        ],
        className="slider-block",
    )


def rebalance_frequency_dropdown_block(default_value="Annually"):
    return html.Div(
        children=[
            html.Label("Rebalance frequency", className="slider-label"),
            dcc.Dropdown(
                id="rebalance-frequency",
                options=[
                    {"label": freq, "value": freq}
                    for freq in REBALANCE_FREQUENCIES
                ],
                value=default_value,
                clearable=False,
                className="benchmark-dropdown",
            ),
        ],
        className="slider-block",
    )


def period_preset_dropdown_block(default_value="Full Sample"):
    return html.Div(
        children=[
            html.Label("Period preset", className="slider-label"),
            dcc.Dropdown(
                id="period-preset",
                options=[
                    {"label": preset_name, "value": preset_name}
                    for preset_name in PERIOD_PRESETS.keys()
                ],
                value=default_value,
                clearable=False,
                className="benchmark-dropdown",
            ),
        ],
        className="slider-block",
    )

def rebalance_weights_to_100(raw_weights, triggered_asset=None):
    """
    Upraví váhy tak, aby mali spolu presne 100 %.

    Keď používateľ zmení jednu váhu, táto váha zostane fixná.
    Ostatné ETF sa prepočítajú do zvyšného priestoru tak,
    aby zachovali svoj vzájomný pomer.
    """

    weights = {
        asset: max(0, min(100, float(value)))
        for asset, value in raw_weights.items()
    }

    assets = list(weights.keys())

    if triggered_asset is None:
        total = sum(weights.values())

        if total == 0:
            equal_weight = 100 / len(assets)
            return {asset: equal_weight for asset in assets}

        normalized = {
            asset: weights[asset] / total * 100
            for asset in assets
        }

        normalized = {
            asset: round(value, 2)
            for asset, value in normalized.items()
        }

        difference = round(100 - sum(normalized.values()), 2)

        if abs(difference) > 0:
            normalized[assets[-1]] = round(normalized[assets[-1]] + difference, 2)

        return normalized

    fixed_value = weights[triggered_asset]
    remaining_weight = 100 - fixed_value

    other_assets = [asset for asset in assets if asset != triggered_asset]
    other_total = sum(weights[asset] for asset in other_assets)

    adjusted = {
        triggered_asset: fixed_value
    }

    if other_total == 0:
        equal_weight = remaining_weight / len(other_assets)

        for asset in other_assets:
            adjusted[asset] = equal_weight
    else:
        for asset in other_assets:
            adjusted[asset] = weights[asset] / other_total * remaining_weight

    adjusted = {
        asset: round(value, 2)
        for asset, value in adjusted.items()
    }

    difference = round(100 - sum(adjusted.values()), 2)

    if abs(difference) > 0:
        correction_asset = other_assets[-1] if other_assets else triggered_asset
        adjusted[correction_asset] = round(adjusted[correction_asset] + difference, 2)

    return adjusted


# --------------------------------------------------
# 3. Grafy
# --------------------------------------------------

def calculate_crisis_period_summary(
    period_name,
    start_date,
    end_date,
    portfolio_weights,
    benchmark_weights,
):
    period_prices = filter_prices_by_date(prices, start_date, end_date)

    if len(period_prices) < 5:
        return None

    crisis_portfolio = calculate_rebalanced_portfolio(
        prices=period_prices,
        weights=portfolio_weights,
        initial_value=1000,
        annual_rebalancing=True,
    )

    crisis_benchmark = calculate_rebalanced_portfolio(
        prices=period_prices,
        weights=benchmark_weights,
        initial_value=1000,
        annual_rebalancing=True,
    )

    portfolio_metrics = calculate_metrics(crisis_portfolio)
    benchmark_metrics = calculate_metrics(crisis_benchmark)

    asset_returns = period_prices.iloc[-1] / period_prices.iloc[0] - 1

    best_asset = asset_returns.idxmax()
    worst_asset = asset_returns.idxmin()

    return {
        "Period": period_name,
        "Dates": f"{start_date} → {end_date}",
        "Portfolio Return": portfolio_metrics["Total Return"],
        "Portfolio Max DD": portfolio_metrics["Max Drawdown"],
        "Benchmark Return": benchmark_metrics["Total Return"],
        "Benchmark Max DD": benchmark_metrics["Max Drawdown"],
        "Best ETF": best_asset,
        "Worst ETF": worst_asset,
    }


def build_crisis_table(portfolio_weights, benchmark_weights):
    rows = []

    for period_name, dates in CRISIS_PERIODS.items():
        start_date, end_date = dates

        summary = calculate_crisis_period_summary(
            period_name=period_name,
            start_date=start_date,
            end_date=end_date,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
        )

        if summary is None:
            continue

        def _td_signed(value, positive_good=True):
            cls = tone_for(value, positive_good=positive_good)
            return html.Td(format_percent(value), className=cls)

        rows.append(
            html.Tr(
                children=[
                    html.Td(
                        children=[
                            html.Div(summary["Period"], className="crisis-period-name"),
                            html.Div(summary["Dates"], className="crisis-period-dates"),
                        ]
                    ),
                    _td_signed(summary["Portfolio Return"]),
                    _td_signed(summary["Portfolio Max DD"]),
                    _td_signed(summary["Benchmark Return"]),
                    _td_signed(summary["Benchmark Max DD"]),
                    html.Td(summary["Best ETF"]),
                    html.Td(summary["Worst ETF"]),
                ]
            )
        )

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div("Crisis Period Performance", className="crisis-title"),
                    html.Div(
                        "Selected portfolio compared with the chosen benchmark during major market stress periods.",
                        className="crisis-subtitle",
                    ),
                ],
                className="crisis-header",
            ),
            html.Div(
                children=[
                    html.Table(
                        children=[
                            html.Thead(
                                html.Tr(
                                    children=[
                                        html.Th("Crisis Period"),
                                        html.Th("Portfolio Return"),
                                        html.Th("Portfolio Max DD"),
                                        html.Th("Benchmark Return"),
                                        html.Th("Benchmark Max DD"),
                                        html.Th("Best ETF"),
                                        html.Th("Worst ETF"),
                                    ]
                                )
                            ),
                            html.Tbody(rows),
                        ],
                        className="crisis-table",
                    )
                ],
                className="crisis-table-container",
            ),
        ]
    )

def build_value_chart(portfolio, benchmark_portfolio=None, benchmark_name="Benchmark", theme="light"):
    t = chart_theme(theme)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio["Portfolio Value"],
            mode="lines",
            name="Selected Portfolio",
            line=dict(color=LINE_COLOR, width=2.7),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Portfolio: €%{y:,.2f}<extra></extra>",
        )
    )

    if benchmark_portfolio is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_portfolio.index,
                y=benchmark_portfolio["Portfolio Value"],
                mode="lines",
                name=benchmark_name,
                line=dict(color=BENCHMARK_COLOR, width=2.2, dash="dash"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Benchmark: €%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Portfolio Value vs Benchmark",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        xaxis_title=None,
        yaxis_title=None,
        template=t["template"],
        height=460,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=t["grid"], zeroline=False, tickprefix="€")

    return fig


def build_drawdown_chart(portfolio, theme="light"):
    t = chart_theme(theme)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio["Drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(color=DRAWDOWN_COLOR, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.20)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Portfolio Drawdown",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        xaxis_title=None,
        yaxis_title=None,
        yaxis_tickformat=".0%",
        template=t["template"],
        height=380,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
    )

    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=t["grid"], zeroline=False)

    return fig


def build_rolling_metrics_chart(portfolio, metric_name, window_label, risk_free_rate, theme="light"):
    t = chart_theme(theme)
    metric_config = ROLLING_METRICS.get(metric_name, ROLLING_METRICS["Return"])
    window_days = ROLLING_WINDOWS.get(window_label, 252)

    rolling_df = calculate_rolling_metrics(
        portfolio=portfolio,
        window_days=window_days,
        risk_free_rate=risk_free_rate,
    )

    series = rolling_df[metric_config["column"]].dropna()

    fig = go.Figure()

    if len(series) == 0:
        fig.update_layout(
            title=f"Not enough data for {window_label} rolling window",
            template="plotly_white",
            height=360,
        )
        return fig

    if metric_config["zero_baseline"]:
        positive = series.where(series >= 0)
        negative = series.where(series < 0)

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=positive,
                mode="lines",
                line=dict(color="#059669", width=2),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.18)",
                name="Positive",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>%{{y:{metric_config['hover_format']}}}<extra></extra>",
                connectgaps=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=negative,
                mode="lines",
                line=dict(color="#dc2626", width=2),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.18)",
                name="Negative",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>%{{y:{metric_config['hover_format']}}}<extra></extra>",
                connectgaps=False,
            )
        )

        fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"))
    else:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                line=dict(color="#f59e0b", width=2),
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.16)",
                name=metric_name,
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>%{{y:{metric_config['hover_format']}}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{window_label} {metric_config['title_suffix']}",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        template=t["template"],
        height=360,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["grid"],
        zeroline=False,
        tickformat=metric_config["y_tickformat"],
    )

    return fig


def build_monte_carlo_chart(portfolio, horizon_label, n_sims_label, theme="light"):
    t = chart_theme(theme)

    try:
        horizon_years = int(horizon_label.rstrip("Y"))
    except ValueError:
        horizon_years = 10

    try:
        n_sims = int(n_sims_label)
    except ValueError:
        n_sims = 1000

    forecast = monte_carlo_forecast(
        portfolio=portfolio,
        horizon_years=horizon_years,
        n_simulations=n_sims,
    )

    fig = go.Figure()

    if forecast.empty:
        fig.update_layout(
            title=dict(
                text="Monte Carlo: not enough data",
                font=dict(color=t["title_font"]),
            ),
            template=t["template"],
            height=380,
        )
        return fig

    # Historical line (last 2 years for context)
    hist = portfolio["Portfolio Value"].iloc[-min(len(portfolio), 504):]
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            line=dict(color="#94a3b8", width=2),
            name="Historical",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>€%{y:,.0f}<extra></extra>",
        )
    )

    # Fan: outer band 5-95
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast["p95"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast["p5"], mode="lines",
            line=dict(width=0),
            fill="tonexty", fillcolor="rgba(99, 102, 241, 0.12)",
            name="5–95% band",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>5–95%% range<extra></extra>",
        )
    )

    # Inner band 25-75
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast["p75"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast["p25"], mode="lines",
            line=dict(width=0),
            fill="tonexty", fillcolor="rgba(99, 102, 241, 0.28)",
            name="25–75% band",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>25–75%% range<extra></extra>",
        )
    )

    # Median
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast["p50"], mode="lines",
            line=dict(color="#6366f1", width=2.6),
            name="Median",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Median: €%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Monte Carlo Forecast — {horizon_years}Y, {n_sims:,} simulations",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        template=t["template"],
        height=440,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(
        showgrid=True, gridcolor=t["grid"], zeroline=False, tickprefix="€",
    )

    return fig


def build_correlation_heatmap(filtered_prices, theme="light"):
    t = chart_theme(theme)
    correlation = calculate_asset_correlation(filtered_prices)
    assets = list(correlation.columns)
    z = correlation.values
    text = [[f"{val:.2f}" for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=assets,
            y=assets,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, family="Inter, Segoe UI, Arial", color="white"),
            colorscale=[
                [0.0, "#dc2626"],
                [0.5, t["heatmap_mid"]],
                [1.0, "#059669"],
            ],
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=dict(text="ρ", font=dict(size=14, color=t["axis_font"])),
                thickness=14,
                len=0.7,
                outlinewidth=0,
                tickfont=dict(color=t["axis_font"]),
            ),
            hovertemplate="<b>%{y} ↔ %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Asset Correlation (daily returns)",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        template=t["template"],
        height=380,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    return fig


def build_monthly_returns_heatmap(portfolio, theme="light"):
    t = chart_theme(theme)
    monthly = portfolio["Portfolio Value"].resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()

    if len(monthly_returns) == 0:
        return go.Figure().update_layout(
            title="Monthly Returns",
            template="plotly_white",
            height=320,
        )

    df = monthly_returns.to_frame(name="Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    pivot = df.pivot(index="Year", columns="Month", values="Return")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    yearly_totals = (1 + pivot).prod(axis=1, min_count=1) - 1

    z_months = pivot.values * 100
    z_year = yearly_totals.values.reshape(-1, 1) * 100
    z = np.concatenate([z_months, z_year], axis=1)

    x_labels = month_labels + ["Year"]

    text = [
        [f"{v:.1f}" if not pd.isna(v) else "" for v in row]
        for row in z
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=[str(y) for y in pivot.index],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12, family="Inter, Segoe UI, Arial"),
            colorscale=[
                [0.0, "#dc2626"],
                [0.5, t["heatmap_mid"]],
                [1.0, "#059669"],
            ],
            zmid=0,
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text="%", font=dict(size=13, color=t["axis_font"])),
                thickness=14,
                len=0.7,
                outlinewidth=0,
                tickfont=dict(color=t["axis_font"]),
            ),
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        )
    )

    fig.add_shape(
        type="rect",
        x0=11.5, x1=12.5,
        y0=-0.5, y1=len(pivot.index) - 0.5,
        line=dict(color="#94a3b8", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    fig.update_layout(
        title=dict(
            text="Monthly Returns Heatmap (%)",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        template=t["template"],
        height=max(380, 30 * len(pivot.index) + 130),
        margin=dict(l=55, r=30, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color=t["axis_font"]),
        ),
        xaxis=dict(
            side="top",
            tickfont=dict(size=12, color=t["axis_font"]),
        ),
    )

    return fig


def build_allocation_chart(weights, theme="light"):
    t = chart_theme(theme)
    labels = list(weights.keys())
    values = list(weights.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[COLOR_MAP.get(asset, "#64748b") for asset in labels],
                    line=dict(color="white", width=3),
                ),
                texttemplate="<b>%{label}</b><br>%{percent}",
                textposition="inside",
                insidetextorientation="horizontal",
                textfont=dict(
                    size=13,
                    color="white",
                    family="Inter, Segoe UI, Arial",
                ),
                hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Target Allocation",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color=t["title_font"]),
        ),
        template=t["template"],
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", color=t["axis_font"]),
        showlegend=False,
        uniformtext=dict(
            minsize=11,
            mode="hide",
        ),
        annotations=[
            dict(
                text="Allocation",
                x=0.5,
                y=0.5,
                font=dict(size=15, color=t["annotation_font"]),
                showarrow=False,
            )
        ],
    )

    return fig


# --------------------------------------------------
# 4. Dash aplikácia
# --------------------------------------------------

app = Dash(__name__)
server = app.server

app.title = "Portfolio Risk Dashboard – Marián Mičian"

app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="url-loaded", data=False),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            children=[
                                                "Portfolio Analytics Project · by ",
                                                html.Span("Marián Mičian", className="eyebrow-author"),
                                            ],
                                            className="eyebrow",
                                        ),
                                        html.H1(
                                            "Interactive Portfolio Risk Dashboard",
                                            className="hero-title",
                                        ),
                                        html.P(
                                            "Commodity allocation, annual rebalancing and crisis-period risk analysis "
                                            "based on daily adjusted close ETF data.",
                                            className="hero-subtitle",
                                        ),
                                        html.Div(
                                            children=[
                                                html.Span("SPY", className="badge"),
                                                html.Span("GLD", className="badge"),
                                                html.Span("AGG", className="badge"),
                                                html.Span("DBC", className="badge"),
                                                html.Span("Daily Data", className="badge"),
                                                html.Span("Annual Rebalancing", className="badge"),
                                                html.Span("Risk-free Rate", className="badge"),
                                                html.Span("Benchmark Comparison", className="badge"),
                                                html.Span("2006–2026 Sample", className="badge"),
                                            ],
                                            className="badge-row",
                                        ),
                                    ],
                                    className="hero-text",
                                ),
                                html.Button(
                                    children=[
                                        html.Span("🌙", className="theme-toggle-icon", id="theme-toggle-icon"),
                                        html.Span("Dark mode", id="theme-toggle-label"),
                                    ],
                                    id="theme-toggle",
                                    n_clicks=0,
                                    className="theme-toggle",
                                ),
                            ],
                            className="hero-content",
                        ),
                    ],
                    className="hero",
                ),

                dcc.Store(id="theme-store", data="light"),

                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div("Portfolio Controls", className="sidebar-title"),

                                html.Div("Investment Period", className="section-label"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=MIN_DATE,
                                    max_date_allowed=MAX_DATE,
                                    start_date=MIN_DATE,
                                    end_date=MAX_DATE,
                                    display_format="YYYY-MM-DD",
                                    className="date-picker",
                                ),
                                period_preset_dropdown_block("Full Sample"),

                                dcc.RadioItems(
                                    id="date-quick-range",
                                    options=[
                                        {"label": label, "value": label}
                                        for label in DATE_QUICK_RANGES
                                    ],
                                    value=None,
                                    className="segmented quick-range-row",
                                    inputClassName="segmented-input",
                                    labelClassName="segmented-label",
                                ),

                                html.Div("Simulation Settings", className="section-label"),
                                investment_slider_block(10000),
                                risk_free_rate_slider_block(0.0),
                                rebalance_frequency_dropdown_block("Annually"),
                                benchmark_dropdown_block("SPY only"),

                                html.Div("Target Allocation", className="section-label"),
                                portfolio_preset_dropdown_block("Default (60/15/15/10)"),
                                slider_block("SPY", 60),
                                slider_block("GLD", 15),
                                slider_block("AGG", 15),
                                slider_block("DBC", 10),

                                html.Div(
                                    children=[
                                        html.Button(
                                            "Reset to defaults",
                                            id="reset-button",
                                            n_clicks=0,
                                            className="reset-button",
                                        ),
                                        html.Button(
                                            "Copy share link",
                                            id="copy-link-button",
                                            n_clicks=0,
                                            className="reset-button copy-link-button",
                                        ),
                                    ],
                                    className="sidebar-actions",
                                ),

                                html.Div(
                                    id="weights-summary",
                                    className="weights-box",
                                ),

                                html.Div(
                                    children=[
                                        html.Div("Methodology", className="methodology-title"),
                                        html.P(
                                            "The dashboard uses daily adjusted close prices. The initial investment, "
                                            "risk-free rate and benchmark can be adjusted interactively. Portfolio weights "
                                            "always sum to 100%. When one ETF weight is changed, the remaining ETF weights "
                                            "are rescaled proportionally. The portfolio is rebalanced once per year on the "
                                            "first trading day of a new year.",
                                            className="methodology-text",
                                        ),
                                    ],
                                    className="methodology-box",
                                ),
                            ],
                            className="sidebar",
                        ),

                        html.Div(
                            children=[
                                dcc.Loading(
                                    id="loading-metrics",
                                    type="circle",
                                    color="#6366f1",
                                    parent_className="loading-wrap",
                                    children=html.Div(
                                        id="metrics-cards",
                                        className="metrics-grid",
                                    ),
                                ),

                                dcc.Loading(
                                    id="loading-value",
                                    type="circle",
                                    color="#6366f1",
                                    parent_className="loading-wrap",
                                    children=html.Div(
                                        children=[
                                            dcc.Graph(
                                                id="portfolio-value-chart",
                                                config={"displayModeBar": False},
                                            ),
                                        ],
                                        className="chart-card",
                                    ),
                                ),

                                dcc.Loading(
                                    id="loading-dd-pie",
                                    type="circle",
                                    color="#6366f1",
                                    parent_className="loading-wrap",
                                    children=html.Div(
                                        children=[
                                            html.Div(
                                                children=[
                                                    dcc.Graph(
                                                        id="drawdown-chart",
                                                        config={"displayModeBar": False},
                                                    ),
                                                ],
                                                className="chart-card no-margin",
                                            ),
                                            html.Div(
                                                children=[
                                                    dcc.Graph(
                                                        id="allocation-pie-chart",
                                                        config={"displayModeBar": False},
                                                    ),
                                                ],
                                                className="chart-card no-margin",
                                            ),
                                        ],
                                        className="two-column-grid",
                                    ),
                                ),

                                html.Div(
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    "Rolling Metrics",
                                                    className="rolling-title",
                                                ),
                                                html.Div(
                                                    children=[
                                                        dcc.RadioItems(
                                                            id="rolling-metric",
                                                            options=[
                                                                {"label": name, "value": name}
                                                                for name in ROLLING_METRICS.keys()
                                                            ],
                                                            value="Return",
                                                            className="segmented",
                                                            inputClassName="segmented-input",
                                                            labelClassName="segmented-label",
                                                        ),
                                                        dcc.RadioItems(
                                                            id="rolling-window",
                                                            options=[
                                                                {"label": name, "value": name}
                                                                for name in ROLLING_WINDOWS.keys()
                                                            ],
                                                            value="12M",
                                                            className="segmented",
                                                            inputClassName="segmented-input",
                                                            labelClassName="segmented-label",
                                                        ),
                                                    ],
                                                    className="rolling-controls",
                                                ),
                                            ],
                                            className="rolling-header",
                                        ),
                                        dcc.Loading(
                                            id="loading-rolling",
                                            type="circle",
                                            color="#6366f1",
                                            children=dcc.Graph(
                                                id="rolling-metrics-chart",
                                                config={"displayModeBar": False},
                                            ),
                                        ),
                                    ],
                                    className="chart-card",
                                ),

                                html.Div(
                                    children=[
                                        dcc.Loading(
                                            id="loading-corr",
                                            type="circle",
                                            color="#6366f1",
                                            children=dcc.Graph(
                                                id="correlation-heatmap",
                                                config={"displayModeBar": False},
                                            ),
                                        ),
                                    ],
                                    className="chart-card",
                                ),

                                html.Div(
                                    children=[
                                        dcc.Loading(
                                            id="loading-monthly",
                                            type="circle",
                                            color="#6366f1",
                                            children=dcc.Graph(
                                                id="monthly-returns-heatmap",
                                                config={"displayModeBar": False},
                                            ),
                                        ),
                                    ],
                                    className="chart-card",
                                ),

                                html.Div(
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    "Monte Carlo Forecast",
                                                    className="rolling-title",
                                                ),
                                                html.Div(
                                                    children=[
                                                        dcc.RadioItems(
                                                            id="mc-horizon",
                                                            options=[
                                                                {"label": h, "value": h}
                                                                for h in MONTE_CARLO_HORIZONS
                                                            ],
                                                            value="10Y",
                                                            className="segmented",
                                                            inputClassName="segmented-input",
                                                            labelClassName="segmented-label",
                                                        ),
                                                        dcc.RadioItems(
                                                            id="mc-sims",
                                                            options=[
                                                                {"label": f"{n} sims", "value": n}
                                                                for n in MONTE_CARLO_SIMS
                                                            ],
                                                            value="1000",
                                                            className="segmented",
                                                            inputClassName="segmented-input",
                                                            labelClassName="segmented-label",
                                                        ),
                                                    ],
                                                    className="rolling-controls",
                                                ),
                                            ],
                                            className="rolling-header",
                                        ),
                                        dcc.Loading(
                                            id="loading-mc",
                                            type="circle",
                                            color="#6366f1",
                                            children=dcc.Graph(
                                                id="monte-carlo-chart",
                                                config={"displayModeBar": False},
                                            ),
                                        ),
                                    ],
                                    className="chart-card",
                                ),

                                html.Div(
                                    children=[
                                        dcc.Loading(
                                            id="loading-crisis",
                                            type="circle",
                                            color="#6366f1",
                                            children=html.Div(id="crisis-table"),
                                        ),
                                    ],
                                    className="chart-card crisis-card",
                                ),
                            ],
                            className="main-content",
                        ),
                    ],
                    className="dashboard-grid",
                ),

                html.Footer(
                    children=[
                        html.Div(
                            children=[
                                "Built by ",
                                html.Strong("Marián Mičian"),
                                html.Span(" · 2026", className="footer-dim"),
                            ],
                            className="footer-author",
                        ),
                        html.Div(
                            children=[
                                "Powered by ",
                                html.A("Dash", href="https://dash.plotly.com/", target="_blank", className="footer-link"),
                                " · ",
                                html.A("Plotly", href="https://plotly.com/python/", target="_blank", className="footer-link"),
                                " · Data via Yahoo Finance",
                            ],
                            className="footer-meta",
                        ),
                    ],
                    className="page-footer",
                ),
            ],
            className="page-container",
        )
    ],
    className="app-shell",
)


# --------------------------------------------------
# 5. Callback - prepočítanie dashboardu
# --------------------------------------------------

@app.callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("period-preset", "value"),
)
def update_date_range_from_preset(period_preset):
    if period_preset not in PERIOD_PRESETS:
        period_preset = "Full Sample"

    start_date, end_date = PERIOD_PRESETS[period_preset]

    return start_date, end_date


@app.callback(
    Output("date-range", "start_date", allow_duplicate=True),
    Output("date-range", "end_date", allow_duplicate=True),
    Input("date-quick-range", "value"),
    prevent_initial_call=True,
)
def update_date_range_from_quick(quick_range):
    if not quick_range:
        return no_update, no_update
    start, end = compute_quick_range(quick_range, MAX_DATE)
    return start if start else str(MIN_DATE), end


@app.callback(
    Output("spy-weight", "value", allow_duplicate=True),
    Output("gld-weight", "value", allow_duplicate=True),
    Output("agg-weight", "value", allow_duplicate=True),
    Output("dbc-weight", "value", allow_duplicate=True),
    Input("portfolio-preset", "value"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset_or_reset(preset_name, _reset_clicks):
    triggered = ctx.triggered_id

    if triggered == "reset-button":
        weights = PORTFOLIO_PRESETS["Default (60/15/15/10)"]
    else:
        weights = PORTFOLIO_PRESETS.get(preset_name)
        if weights is None:
            return no_update, no_update, no_update, no_update

    return weights["SPY"], weights["GLD"], weights["AGG"], weights["DBC"]

@app.callback(
    Output("spy-weight", "value"),
    Output("gld-weight", "value"),
    Output("agg-weight", "value"),
    Output("dbc-weight", "value"),
    Output("spy-weight-label", "children"),
    Output("gld-weight-label", "children"),
    Output("agg-weight-label", "children"),
    Output("dbc-weight-label", "children"),
    Output("initial-investment-label", "children"),
    Output("risk-free-rate-label", "children"),
    Output("metrics-cards", "children"),
    Output("portfolio-value-chart", "figure"),
    Output("drawdown-chart", "figure"),
    Output("allocation-pie-chart", "figure"),
    Output("weights-summary", "children"),
    Output("crisis-table", "children"),
    Output("correlation-heatmap", "figure"),
    Output("monthly-returns-heatmap", "figure"),
    Output("rolling-metrics-chart", "figure"),
    Output("monte-carlo-chart", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("initial-investment", "value"),
    Input("risk-free-rate", "value"),
    Input("benchmark-selector", "value"),
    Input("rebalance-frequency", "value"),
    Input("rolling-metric", "value"),
    Input("rolling-window", "value"),
    Input("mc-horizon", "value"),
    Input("mc-sims", "value"),
    Input("theme-store", "data"),
    Input("spy-weight", "value"),
    Input("gld-weight", "value"),
    Input("agg-weight", "value"),
    Input("dbc-weight", "value"),
)
def update_dashboard(
    start_date,
    end_date,
    initial_investment,
    risk_free_rate,
    benchmark_name,
    rebalance_frequency,
    rolling_metric,
    rolling_window,
    mc_horizon,
    mc_sims,
    theme,
    spy_weight,
    gld_weight,
    agg_weight,
    dbc_weight,
):
    if initial_investment is None:
        initial_investment = 10000

    if risk_free_rate is None:
        risk_free_rate = 0.0

    if benchmark_name not in BENCHMARKS:
        benchmark_name = "SPY only"

    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        rebalance_frequency = "Annually"

    if rolling_metric not in ROLLING_METRICS:
        rolling_metric = "Return"

    if rolling_window not in ROLLING_WINDOWS:
        rolling_window = "12M"

    if mc_horizon not in MONTE_CARLO_HORIZONS:
        mc_horizon = "10Y"

    if mc_sims not in MONTE_CARLO_SIMS:
        mc_sims = "1000"

    if theme not in ("light", "dark"):
        theme = "light"

    raw_weights = {
        "SPY": spy_weight,
        "GLD": gld_weight,
        "AGG": agg_weight,
        "DBC": dbc_weight,
    }

    triggered_id = ctx.triggered_id

    slider_to_asset = {
        "spy-weight": "SPY",
        "gld-weight": "GLD",
        "agg-weight": "AGG",
        "dbc-weight": "DBC",
    }

    triggered_asset = slider_to_asset.get(triggered_id)

    adjusted_weights = rebalance_weights_to_100(
        raw_weights=raw_weights,
        triggered_asset=triggered_asset,
    )

    filtered_prices = filter_prices_by_date(prices, start_date, end_date)

    if len(filtered_prices) < 30:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Selected period is too short",
            template="plotly_white",
            height=350,
        )

        cards = [
            metric_card("Error", "Too short", "Select a longer date range"),
        ]

        weights_text = html.Div("Select a longer date range.")

        return (
            adjusted_weights["SPY"],
            adjusted_weights["GLD"],
            adjusted_weights["AGG"],
            adjusted_weights["DBC"],
            f"{adjusted_weights['SPY']:.2f}%",
            f"{adjusted_weights['GLD']:.2f}%",
            f"{adjusted_weights['AGG']:.2f}%",
            f"{adjusted_weights['DBC']:.2f}%",
            format_slider_eur(initial_investment),
            format_slider_percent(risk_free_rate),
            cards,
            empty_fig,
            empty_fig,
            empty_fig,
            weights_text,
            html.Div("Crisis table unavailable for the selected settings."),
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
        )

    portfolio = calculate_rebalanced_portfolio(
        prices=filtered_prices,
        weights=adjusted_weights,
        initial_value=initial_investment,
        rebalance_frequency=rebalance_frequency.lower(),
    )

    benchmark_weights = BENCHMARKS.get(benchmark_name, BENCHMARKS["SPY only"])

    benchmark_portfolio = calculate_rebalanced_portfolio(
        prices=filtered_prices,
        weights=benchmark_weights,
        initial_value=initial_investment,
        rebalance_frequency=rebalance_frequency.lower(),
    )
    crisis_table = build_crisis_table(
    portfolio_weights=adjusted_weights,
    benchmark_weights=benchmark_weights,
    )
    metrics = calculate_metrics(
        portfolio,
        risk_free_rate=risk_free_rate / 100,
    )

    def _fmt_ratio(v):
        return "N/A" if v is None or pd.isna(v) else f"{v:.2f}"

    sortino = metrics.get("Sortino Ratio")
    calmar = metrics.get("Calmar Ratio")
    var_95 = metrics.get("VaR 95")
    longest_dd = metrics.get("Longest Drawdown Days", 0)

    pv_series = portfolio["Portfolio Value"].tolist()
    dd_series = portfolio["Drawdown"].tolist()
    daily_returns_series = portfolio["Daily Return"].dropna().tolist()

    rolling_vol_series = (
        portfolio["Daily Return"].rolling(60).std().dropna().tolist()
    )
    rolling_sharpe_window = 60
    rolling_mean = portfolio["Daily Return"].rolling(rolling_sharpe_window).mean()
    rolling_std = portfolio["Daily Return"].rolling(rolling_sharpe_window).std()
    rolling_sharpe_series = (
        ((rolling_mean / rolling_std) * np.sqrt(252)).dropna().tolist()
    )

    def _spk(values, tone="neutral", fill=True):
        return make_sparkline(values, color=SPARKLINE_COLORS[tone], fill=fill)

    total_return_tone = tone_for(metrics["Total Return"])
    cagr_tone = tone_for(metrics["CAGR"])
    sharpe_tone = tone_for(metrics["Sharpe Ratio"])
    sortino_tone = tone_for(sortino)
    calmar_tone = tone_for(calmar)
    max_dd_tone = tone_for(metrics["Max Drawdown"])
    var_tone = tone_for(var_95)
    best_day_tone = tone_for(metrics["Best Day"])
    worst_day_tone = tone_for(metrics["Worst Day"])

    cards = [
        metric_card(
            "End Value", format_number(metrics["End Value"]),
            "Final portfolio value", tone="neutral",
            tooltip=METRIC_TOOLTIPS["End Value"],
            sparkline=_spk(pv_series, tone="neutral"),
        ),
        metric_card(
            "Total Return", format_percent(metrics["Total Return"]),
            "Cumulative performance",
            tone=total_return_tone,
            tooltip=METRIC_TOOLTIPS["Total Return"],
            sparkline=_spk(pv_series, tone=total_return_tone),
        ),
        metric_card(
            "CAGR", format_percent(metrics["CAGR"]),
            "Annualized return",
            tone=cagr_tone,
            tooltip=METRIC_TOOLTIPS["CAGR"],
            sparkline=_spk(pv_series, tone=cagr_tone),
        ),
        metric_card(
            "Volatility", format_percent(metrics["Annual Volatility"]),
            "Annualized standard deviation", tone="neutral",
            tooltip=METRIC_TOOLTIPS["Volatility"],
            sparkline=_spk(rolling_vol_series, tone="neutral"),
        ),
        metric_card(
            "Sharpe Ratio", _fmt_ratio(metrics["Sharpe Ratio"]),
            "Adjusted for risk-free rate",
            tone=sharpe_tone,
            tooltip=METRIC_TOOLTIPS["Sharpe Ratio"],
            sparkline=_spk(rolling_sharpe_series, tone=sharpe_tone),
        ),
        metric_card(
            "Sortino Ratio", _fmt_ratio(sortino),
            "Downside-only risk adjustment",
            tone=sortino_tone,
            tooltip=METRIC_TOOLTIPS["Sortino Ratio"],
            sparkline=_spk(rolling_sharpe_series, tone=sortino_tone),
        ),
        metric_card(
            "Calmar Ratio", _fmt_ratio(calmar),
            "CAGR per unit of Max DD",
            tone=calmar_tone,
            tooltip=METRIC_TOOLTIPS["Calmar Ratio"],
            sparkline=_spk(pv_series, tone=calmar_tone),
        ),
        metric_card(
            "Max Drawdown", format_percent(metrics["Max Drawdown"]),
            "Worst peak-to-trough decline",
            tone=max_dd_tone,
            tooltip=METRIC_TOOLTIPS["Max Drawdown"],
            sparkline=_spk(dd_series, tone="negative"),
        ),
        metric_card(
            "Longest DD", f"{int(longest_dd)} days",
            "Time spent below previous peak",
            tone="neutral",
            tooltip=METRIC_TOOLTIPS["Longest DD"],
            sparkline=_spk(dd_series, tone="negative"),
        ),
        metric_card(
            "VaR 95%", format_percent(var_95),
            "Worst 5% daily loss threshold",
            tone=var_tone,
            tooltip=METRIC_TOOLTIPS["VaR 95%"],
            sparkline=_spk(daily_returns_series, tone="negative", fill=False),
        ),
        metric_card(
            "Best Day", format_percent(metrics["Best Day"]),
            "Highest daily return",
            tone=best_day_tone,
            tooltip=METRIC_TOOLTIPS["Best Day"],
            sparkline=_spk(daily_returns_series, tone="positive", fill=False),
        ),
        metric_card(
            "Worst Day", format_percent(metrics["Worst Day"]),
            "Lowest daily return",
            tone=worst_day_tone,
            tooltip=METRIC_TOOLTIPS["Worst Day"],
            sparkline=_spk(daily_returns_series, tone="negative", fill=False),
        ),
    ]

    value_fig = build_value_chart(
        portfolio=portfolio,
        benchmark_portfolio=benchmark_portfolio,
        benchmark_name=benchmark_name,
        theme=theme,
    )

    drawdown_fig = build_drawdown_chart(portfolio, theme=theme)
    pie_fig = build_allocation_chart(adjusted_weights, theme=theme)
    correlation_fig = build_correlation_heatmap(filtered_prices, theme=theme)
    monthly_returns_fig = build_monthly_returns_heatmap(portfolio, theme=theme)
    rolling_fig = build_rolling_metrics_chart(
        portfolio=portfolio,
        metric_name=rolling_metric,
        window_label=rolling_window,
        risk_free_rate=risk_free_rate / 100,
        theme=theme,
    )

    mc_fig = build_monte_carlo_chart(
        portfolio=portfolio,
        horizon_label=mc_horizon,
        n_sims_label=mc_sims,
        theme=theme,
    )

    weights_text = html.Div(
        children=[
            html.Div("Simulation inputs", className="weights-title"),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Span("Initial Investment"),
                            html.Strong(format_slider_eur(initial_investment)),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("Risk-free Rate"),
                            html.Strong(format_slider_percent(risk_free_rate)),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("Benchmark"),
                            html.Strong(benchmark_name),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("SPY"),
                            html.Strong(f"{adjusted_weights['SPY']:.2f}%"),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("GLD"),
                            html.Strong(f"{adjusted_weights['GLD']:.2f}%"),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("AGG"),
                            html.Strong(f"{adjusted_weights['AGG']:.2f}%"),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("DBC"),
                            html.Strong(f"{adjusted_weights['DBC']:.2f}%"),
                        ],
                        className="weight-row",
                    ),
                    html.Div(
                        children=[
                            html.Span("Total"),
                            html.Strong(f"{sum(adjusted_weights.values()):.2f}%"),
                        ],
                        className="weight-row total-row",
                    ),
                ],
                className="weights-list",
            ),
        ]
    )

    return (
        adjusted_weights["SPY"],
        adjusted_weights["GLD"],
        adjusted_weights["AGG"],
        adjusted_weights["DBC"],
        f"{adjusted_weights['SPY']:.2f}%",
        f"{adjusted_weights['GLD']:.2f}%",
        f"{adjusted_weights['AGG']:.2f}%",
        f"{adjusted_weights['DBC']:.2f}%",
        format_slider_eur(initial_investment),
        format_slider_percent(risk_free_rate),
        cards,
        value_fig,
        drawdown_fig,
        pie_fig,
        weights_text,
        crisis_table,
        correlation_fig,
        monthly_returns_fig,
        rolling_fig,
        mc_fig,
    )


# --------------------------------------------------
# 6. Theme toggle (light/dark) — clientside, persists in localStorage
# --------------------------------------------------

app.clientside_callback(
    """
    function(n_clicks, current) {
        let next;
        if (n_clicks === 0 || n_clicks === null) {
            // On initial load, read from localStorage
            const saved = localStorage.getItem('dashboard-theme');
            next = saved === 'dark' ? 'dark' : 'light';
        } else {
            next = current === 'dark' ? 'light' : 'dark';
        }
        document.documentElement.setAttribute('data-theme', next);
        try { localStorage.setItem('dashboard-theme', next); } catch (e) {}
        const icon = next === 'dark' ? '☀️' : '🌙';
        const label = next === 'dark' ? 'Light mode' : 'Dark mode';
        return [next, icon, label];
    }
    """,
    Output("theme-store", "data"),
    Output("theme-toggle-icon", "children"),
    Output("theme-toggle-label", "children"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
)


# --------------------------------------------------
# 7. Shareable URL — clientside copy + server-side state restore
# --------------------------------------------------

app.clientside_callback(
    """
    function(n_clicks, spy, gld, agg, dbc, inv, rf, bm, reb, pp, sd, ed, rm, rw) {
        if (!n_clicks) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        const params = new URLSearchParams();
        const setIf = (k, v) => {
            if (v !== null && v !== undefined && v !== '') params.set(k, v);
        };
        setIf('spy', spy);
        setIf('gld', gld);
        setIf('agg', agg);
        setIf('dbc', dbc);
        setIf('inv', inv);
        setIf('rf', rf);
        setIf('bm', bm);
        setIf('reb', reb);
        setIf('pp', pp);
        setIf('sd', sd);
        setIf('ed', ed);
        setIf('rm', rm);
        setIf('rw', rw);
        const search = '?' + params.toString();
        const url = window.location.origin + window.location.pathname + search;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url).catch(() => {});
        }
        setTimeout(() => {
            const btn = document.getElementById('copy-link-button');
            if (btn) btn.textContent = 'Copy share link';
        }, 2000);
        return ['Link copied!', search];
    }
    """,
    Output("copy-link-button", "children"),
    Output("url", "search"),
    Input("copy-link-button", "n_clicks"),
    State("spy-weight", "value"),
    State("gld-weight", "value"),
    State("agg-weight", "value"),
    State("dbc-weight", "value"),
    State("initial-investment", "value"),
    State("risk-free-rate", "value"),
    State("benchmark-selector", "value"),
    State("rebalance-frequency", "value"),
    State("period-preset", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("rolling-metric", "value"),
    State("rolling-window", "value"),
    prevent_initial_call=True,
)


def _parse_url_search(search):
    """Parse ?k=v&k=v into dict."""
    if not search:
        return {}
    from urllib.parse import parse_qs
    parsed = parse_qs(search.lstrip("?"))
    return {k: v[0] for k, v in parsed.items() if v}


@app.callback(
    Output("spy-weight", "value", allow_duplicate=True),
    Output("gld-weight", "value", allow_duplicate=True),
    Output("agg-weight", "value", allow_duplicate=True),
    Output("dbc-weight", "value", allow_duplicate=True),
    Output("initial-investment", "value"),
    Output("risk-free-rate", "value"),
    Output("benchmark-selector", "value"),
    Output("rebalance-frequency", "value"),
    Output("period-preset", "value"),
    Output("date-range", "start_date", allow_duplicate=True),
    Output("date-range", "end_date", allow_duplicate=True),
    Output("rolling-metric", "value"),
    Output("rolling-window", "value"),
    Output("url-loaded", "data"),
    Input("url", "search"),
    State("url-loaded", "data"),
    prevent_initial_call="initial_duplicate",
)
def apply_url_state(search, already_loaded):
    if already_loaded:
        return [no_update] * 13 + [True]

    params = _parse_url_search(search)
    if not params:
        return [no_update] * 13 + [True]

    def _f(key, cast=str):
        if key not in params:
            return no_update
        try:
            return cast(params[key])
        except (ValueError, TypeError):
            return no_update

    return (
        _f("spy", float),
        _f("gld", float),
        _f("agg", float),
        _f("dbc", float),
        _f("inv", float),
        _f("rf", float),
        _f("bm"),
        _f("reb"),
        _f("pp"),
        _f("sd"),
        _f("ed"),
        _f("rm"),
        _f("rw"),
        True,
    )


# --------------------------------------------------
# 7. Spustenie aplikácie
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "").lower() in ("1", "true")
    app.run(debug=debug, host="0.0.0.0", port=port)