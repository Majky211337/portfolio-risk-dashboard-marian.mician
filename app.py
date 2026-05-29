import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx

from portfolio_engine import (
    load_prices,
    filter_prices_by_date,
    calculate_rebalanced_portfolio,
    calculate_metrics,
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


def metric_card(title, value, subtitle):
    return html.Div(
        children=[
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ],
        className="metric-card",
    )


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

        rows.append(
            html.Tr(
                children=[
                    html.Td(
                        children=[
                            html.Div(summary["Period"], className="crisis-period-name"),
                            html.Div(summary["Dates"], className="crisis-period-dates"),
                        ]
                    ),
                    html.Td(format_percent(summary["Portfolio Return"])),
                    html.Td(format_percent(summary["Portfolio Max DD"])),
                    html.Td(format_percent(summary["Benchmark Return"])),
                    html.Td(format_percent(summary["Benchmark Max DD"])),
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

def build_value_chart(portfolio, benchmark_portfolio=None, benchmark_name="Benchmark"):
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
            font=dict(size=18),
        ),
        xaxis_title=None,
        yaxis_title=None,
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Arial"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        tickprefix="€",
    )

    return fig


def build_drawdown_chart(portfolio):
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
            font=dict(size=18),
        ),
        xaxis_title=None,
        yaxis_title=None,
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=25, t=55, b=35),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, Arial"),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
    )

    return fig


def build_allocation_chart(weights):
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
            font=dict(size=18),
        ),
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial"),
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
                font=dict(size=15, color="#475569"),
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

app.title = "Portfolio Risk Dashboard"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div("Portfolio Analytics Project", className="eyebrow"),
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
                            className="hero-content",
                        ),
                    ],
                    className="hero",
                ),

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

                                html.Div("Simulation Settings", className="section-label"),
                                investment_slider_block(10000),
                                risk_free_rate_slider_block(0.0),
                                benchmark_dropdown_block("SPY only"),

                                html.Div("Target Allocation", className="section-label"),
                                slider_block("SPY", 60),
                                slider_block("GLD", 15),
                                slider_block("AGG", 15),
                                slider_block("DBC", 10),

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
                                html.Div(
                                    id="metrics-cards",
                                    className="metrics-grid",
                                ),

                                html.Div(
                                    children=[
                                        dcc.Graph(
                                            id="portfolio-value-chart",
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    className="chart-card",
                                ),

                                html.Div(
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
                            ],
                            className="main-content",
                        ),
                    ],
                    className="dashboard-grid",
                ),
            ],
            className="page-container",
        )
    ],
    className="app-shell",
)

html.Div(
    children=[
        html.Div(id="crisis-table"),
    ],
    className="chart-card",
),

# --------------------------------------------------
# 5. Callback - prepočítanie dashboardu
# --------------------------------------------------

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
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("initial-investment", "value"),
    Input("risk-free-rate", "value"),
    Input("benchmark-selector", "value"),
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
        )

    portfolio = calculate_rebalanced_portfolio(
        prices=filtered_prices,
        weights=adjusted_weights,
        initial_value=initial_investment,
        annual_rebalancing=True,
    )

    benchmark_weights = BENCHMARKS.get(benchmark_name, BENCHMARKS["SPY only"])

    benchmark_portfolio = calculate_rebalanced_portfolio(
        prices=filtered_prices,
        weights=benchmark_weights,
        initial_value=initial_investment,
        annual_rebalancing=True,
    )
    crisis_table = build_crisis_table(
    portfolio_weights=adjusted_weights,
    benchmark_weights=benchmark_weights,
    )
    metrics = calculate_metrics(
        portfolio,
        risk_free_rate=risk_free_rate / 100,
    )

    cards = [
        metric_card("End Value", format_number(metrics["End Value"]), "Final portfolio value"),
        metric_card("Total Return", format_percent(metrics["Total Return"]), "Cumulative performance"),
        metric_card("CAGR", format_percent(metrics["CAGR"]), "Annualized return"),
        metric_card("Volatility", format_percent(metrics["Annual Volatility"]), "Annualized standard deviation"),
        metric_card("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", "Adjusted for selected risk-free rate"),
        metric_card("Max Drawdown", format_percent(metrics["Max Drawdown"]), "Worst peak-to-trough decline"),
        metric_card("Best Day", format_percent(metrics["Best Day"]), "Highest daily return"),
        metric_card("Worst Day", format_percent(metrics["Worst Day"]), "Lowest daily return"),
    ]

    value_fig = build_value_chart(
        portfolio=portfolio,
        benchmark_portfolio=benchmark_portfolio,
        benchmark_name=benchmark_name,
    )

    drawdown_fig = build_drawdown_chart(portfolio)
    pie_fig = build_allocation_chart(adjusted_weights)

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
    )


# --------------------------------------------------
# 6. Spustenie aplikácie
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)