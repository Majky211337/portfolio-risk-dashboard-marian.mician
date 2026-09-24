"""
Portfolio Risk Dashboard: Streamlit entry point.

Run locally with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import portfolio_engine as engine
from dashboard import analytics, charts
from dashboard.config import (
    ASSETS,
    CRISIS_DESCRIPTIONS,
    CRISIS_PERIODS,
    MC_HORIZONS,
    MC_SIMULATIONS,
    METRIC_HELP,
    MIN_TRADING_DAYS,
    ROLLING_METRICS,
    ROLLING_WINDOWS,
    TICKERS,
)
from dashboard.sidebar import Settings, render_sidebar
from dashboard.ui import (
    fmt_eur,
    fmt_pct,
    fmt_pp_delta,
    fmt_ratio,
    fmt_ratio_delta,
    inject_css,
    section_header,
)

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show_chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)


# --------------------------------------------------
# Header
# --------------------------------------------------

def render_header(prices: pd.DataFrame, settings: Settings | None = None) -> None:
    st.markdown("<div class='app-eyebrow'>Multi-asset analytics · v2.0</div>", unsafe_allow_html=True)
    st.title("Portfolio Risk Dashboard")
    st.markdown(
        "<div class='app-subtitle'>Interactive multi-asset portfolio analytics, risk assessment and stress testing</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Backtest a portfolio of US equities (SPY), gold (GLD), US bonds (AGG) and commodities (DBC) "
        "on daily adjusted close prices. Compare it with a benchmark, see how risky it has been, and "
        "check how it held up in past market crises."
    )
    last_date = prices.index.max()
    with st.container(horizontal=True, gap="small"):
        st.badge(f"Data {prices.index.min():%b %Y} – {last_date:%d %b %Y}", icon=":material/database:", color="gray")
        if settings is not None:
            st.badge(f"{settings.start:%d %b %Y} → {settings.end:%d %b %Y}", icon=":material/date_range:", color="violet")
            st.badge(f"Rebalancing: {settings.rebalance.lower()}", icon=":material/balance:", color="violet")
            st.badge(f"Benchmark: {settings.benchmark_name}", icon=":material/flag:", color="gray")
    stale_days = (pd.Timestamp.today().normalize() - last_date).days
    if stale_days > 14:
        st.caption(
            f":material/schedule: Price data ends on {last_date:%d %b %Y} ({stale_days} days ago). "
            "It is refreshed by a scheduled GitHub Action."
        )


# --------------------------------------------------
# KPI row
# --------------------------------------------------

def _sparkline(series: pd.Series) -> list[float] | None:
    values = charts.downsample(series.replace([np.inf, -np.inf], np.nan).dropna(), 90)
    return values.tolist() if len(values) > 1 else None


def render_kpis(portfolio: pd.DataFrame, m: dict, bm: dict | None, benchmark_name: str, risk_free_rate: float) -> None:
    b = bm or {}
    value_path = _sparkline(portfolio["Portfolio Value"])
    rolling_vol = _sparkline(portfolio["Daily Return"].rolling(63).std() * np.sqrt(engine.TRADING_DAYS))
    rolling_sharpe = _sparkline(analytics.rolling_metrics(portfolio, 252, risk_free_rate)["Rolling Sharpe"])
    drawdown = _sparkline(portfolio["Drawdown"])
    cards = [
        ("CAGR", fmt_pct(m["CAGR"]), fmt_pp_delta(m["CAGR"], b.get("CAGR")), "normal", METRIC_HELP["CAGR"], value_path),
        ("Volatility", fmt_pct(m["Annual Volatility"]), fmt_pp_delta(m["Annual Volatility"], b.get("Annual Volatility")), "inverse", METRIC_HELP["Volatility"], rolling_vol),
        ("Sharpe ratio", fmt_ratio(m["Sharpe Ratio"]), fmt_ratio_delta(m["Sharpe Ratio"], b.get("Sharpe Ratio")), "normal", METRIC_HELP["Sharpe"], rolling_sharpe),
        ("Sortino ratio", fmt_ratio(m["Sortino Ratio"]), fmt_ratio_delta(m["Sortino Ratio"], b.get("Sortino Ratio")), "normal", METRIC_HELP["Sortino"], rolling_sharpe),
        ("Max drawdown", fmt_pct(m["Max Drawdown"]), fmt_pp_delta(m["Max Drawdown"], b.get("Max Drawdown")), "normal", METRIC_HELP["Max Drawdown"], drawdown),
        ("Calmar ratio", fmt_ratio(m["Calmar Ratio"]), fmt_ratio_delta(m["Calmar Ratio"], b.get("Calmar Ratio")), "normal", METRIC_HELP["Calmar"], value_path),
    ]
    for column, (label, value, delta, delta_color, help_text, spark) in zip(st.columns(len(cards)), cards):
        with column:
            st.metric(
                label, value, delta=delta, delta_color=delta_color, help=help_text, border=True,
                chart_data=spark, chart_type="area",
            )
    note = f"Deltas compare against the benchmark ({benchmark_name})." if bm else "Benchmark unavailable."
    st.caption(
        f"{note} Sparklines: portfolio value, rolling 3M volatility, rolling 12M Sharpe, and drawdown."
    )


def metrics_comparison_table(m: dict, bm: dict | None, benchmark_name: str) -> pd.DataFrame:
    rows = [
        ("End value", "End Value", fmt_eur),
        ("Total return", "Total Return", fmt_pct),
        ("CAGR", "CAGR", fmt_pct),
        ("Annualized volatility", "Annual Volatility", fmt_pct),
        ("Sharpe ratio", "Sharpe Ratio", fmt_ratio),
        ("Sortino ratio", "Sortino Ratio", fmt_ratio),
        ("Calmar ratio", "Calmar Ratio", fmt_ratio),
        ("Max drawdown", "Max Drawdown", fmt_pct),
        ("Longest drawdown (days)", "Longest Drawdown Days", lambda v: f"{int(v):,}"),
        ("VaR 95% (1-day)", "VaR 95", fmt_pct),
        ("CVaR 95% (1-day)", "CVaR 95", fmt_pct),
        ("Best day", "Best Day", fmt_pct),
        ("Worst day", "Worst Day", fmt_pct),
        ("Positive days", "Positive Days Ratio", fmt_pct),
    ]
    table = {"Metric": [r[0] for r in rows], "Portfolio": [r[2](m[r[1]]) for r in rows]}
    if bm:
        table[benchmark_name] = [r[2](bm[r[1]]) for r in rows]
    return pd.DataFrame(table)


# --------------------------------------------------
# Tabs
# --------------------------------------------------

def render_performance_tab(settings, portfolio, benchmark, m, bm):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "End value", fmt_eur(m["End Value"]), border=True, help=METRIC_HELP["End Value"],
        delta=fmt_pct(m["Total Return"], 1, signed=True), delta_description="total",
    )
    c2.metric("Total return", fmt_pct(m["Total Return"]), border=True, help=METRIC_HELP["Total Return"],
              delta=fmt_pp_delta(m["Total Return"], (bm or {}).get("Total Return")),
              delta_description="vs benchmark" if bm else None)
    c3.metric("Benchmark end value", fmt_eur(bm["End Value"]) if bm else "–", border=True,
              help=f"Value of the same initial investment in the {settings.benchmark_name} benchmark.")
    c4.metric("Positive days", fmt_pct(m["Positive Days Ratio"], 1), border=True, help=METRIC_HELP["Positive Days"],
              delta=f"{m['Positive Days']:,} of {m['Total Trading Days']:,} days", delta_color="off", delta_arrow="off")

    header_col, toggle_col = st.columns([3, 1], vertical_alignment="bottom")
    with header_col:
        section_header(
            f"Growth of {fmt_eur(settings.initial_investment)}",
            f"Portfolio vs {settings.benchmark_name}, {settings.rebalance.lower()} rebalancing.",
        )
    with toggle_col:
        mode = st.segmented_control(
            "Display", ["Value", "Cumulative return"], default="Value", key="growth_mode",
            label_visibility="collapsed", required=True,
        )
    show_chart(charts.growth_chart(portfolio, benchmark, settings.benchmark_name,
                                   "value" if mode == "Value" else "return"))

    section_header("Portfolio vs benchmark", "Every metric for the selected period.")
    table = metrics_comparison_table(m, bm, settings.benchmark_name)
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)


def render_risk_tab(portfolio, benchmark, m, benchmark_name):
    row1 = st.columns(4)
    row1[0].metric("Max drawdown", fmt_pct(m["Max Drawdown"]), border=True, help=METRIC_HELP["Max Drawdown"],
                   delta=f"trough {portfolio['Drawdown'].idxmin():%d %b %Y}", delta_color="off", delta_arrow="off")
    row1[1].metric("Longest drawdown", f"{m['Longest Drawdown Days']:,} days", border=True,
                   help=METRIC_HELP["Longest Drawdown"])
    row1[2].metric("VaR 95% (1-day)", fmt_pct(m["VaR 95"]), border=True, help=METRIC_HELP["VaR"])
    row1[3].metric("CVaR 95% (1-day)", fmt_pct(m["CVaR 95"]), border=True, help=METRIC_HELP["CVaR"])
    row2 = st.columns(4)
    row2[0].metric("Annualized volatility", fmt_pct(m["Annual Volatility"]), border=True, help=METRIC_HELP["Volatility"])
    row2[1].metric("Best day", fmt_pct(m["Best Day"]), border=True, help=METRIC_HELP["Best Day"])
    row2[2].metric("Worst day", fmt_pct(m["Worst Day"]), border=True, help=METRIC_HELP["Worst Day"])
    row2[3].metric("Positive days", fmt_pct(m["Positive Days Ratio"], 1), border=True, help=METRIC_HELP["Positive Days"])

    left, right = st.columns([3, 2])
    with left:
        section_header("Drawdown", "How far the portfolio sat below its previous peak (underwater chart).")
        show_chart(charts.drawdown_chart(portfolio, benchmark, benchmark_name))
    with right:
        section_header("Daily return distribution", "The dashed lines mark the historical 95% VaR and CVaR.")
        show_chart(charts.return_distribution_chart(portfolio["Daily Return"], m["VaR 95"], m["CVaR 95"]))

    with st.expander("How to read these risk metrics"):
        st.markdown(
            "- **Max drawdown**: the worst peak-to-trough loss. It shows how much pain the portfolio actually went through.\n"
            "- **Longest drawdown**: the longest run of trading days spent below a previous high.\n"
            "- **VaR 95%**: on 5% of trading days the loss was at least this large (historical, one-day).\n"
            "- **CVaR 95%**: the average return on those worst 5% of days. It says how bad the bad days were.\n"
            "- **Volatility**: the annualized standard deviation of daily returns. It counts moves up and down alike."
        )


@st.fragment
def render_rolling_tab(portfolio, benchmark, risk_free_rate):
    c1, c2 = st.columns(2)
    metric_name = c1.segmented_control("Metric", list(ROLLING_METRICS), default="Return", key="rolling_metric", required=True)
    window = c2.segmented_control("Window", list(ROLLING_WINDOWS), default="12M", key="rolling_window", required=True)
    config = ROLLING_METRICS[metric_name]
    window_days = ROLLING_WINDOWS[window]

    rolling = analytics.rolling_metrics(portfolio, window_days, risk_free_rate)
    bench_rolling = analytics.rolling_metrics(benchmark, window_days, risk_free_rate) if benchmark is not None else None

    section_header(f"{window} {config['title'].lower()}", "Trailing-window statistics. Each point uses only the data before it.")
    show_chart(charts.rolling_chart(rolling, bench_rolling, config["column"], config["percent"], window))

    series = rolling[config["column"]].dropna()
    if not series.empty:
        fmt = fmt_pct if config["percent"] else fmt_ratio
        c = st.columns(4)
        c[0].metric("Latest", fmt(series.iloc[-1]), border=True)
        c[1].metric("Average", fmt(series.mean()), border=True)
        c[2].metric("Highest", fmt(series.max()), border=True, delta=f"{series.idxmax():%b %Y}", delta_color="off", delta_arrow="off")
        c[3].metric("Lowest", fmt(series.min()), border=True, delta=f"{series.idxmin():%b %Y}", delta_color="off", delta_arrow="off")
        if metric_name == "Return":
            share_positive = (series > 0).mean()
            st.caption(f"The portfolio had a positive {window} return in {share_positive:.0%} of rolling windows.")


def render_diversification_tab(settings, data_version):
    correlation = analytics.asset_correlation(data_version, settings.start, settings.end)

    left, right = st.columns(2)
    with left:
        section_header("Target allocation", "The weights the portfolio is reset to at each rebalance.")
        show_chart(charts.allocation_donut(settings.weights))
    with right:
        section_header("Correlation of daily returns", f"{settings.start:%b %Y} – {settings.end:%b %Y}")
        show_chart(charts.correlation_heatmap(correlation))

    values = correlation.where(~np.eye(len(correlation), dtype=bool)).stack()
    if not values.empty:
        low_pair, high_pair = values.idxmin(), values.idxmax()
        st.info(
            f"Average pairwise correlation: **{values.mean():.2f}**. The least correlated pair is "
            f"**{low_pair[0]} / {low_pair[1]}** ({values.min():.2f}), the most correlated is "
            f"**{high_pair[0]} / {high_pair[1]}** ({values.max():.2f}). Assets with low or negative "
            "correlation tend not to fall at the same time, which can smooth the portfolio's path. "
            "Correlations do change over time and often rise in a crisis.",
            icon=":material/hub:",
        )

    section_header("Individual assets", "Each ETF held on its own, buy-and-hold, over the selected period.")
    stats = analytics.asset_statistics(data_version, settings.start, settings.end, tuple(TICKERS))
    stats.insert(0, "Asset class", [ASSETS[t]["name"] for t in stats.index])
    stats.insert(1, "Weight", [settings.weights[t] / 100 for t in stats.index])
    percent_columns = ["Weight", "Total Return", "CAGR", "Volatility", "Max Drawdown"]
    stats[percent_columns] = stats[percent_columns] * 100
    st.dataframe(
        stats,
        width="stretch",
        column_config={
            **{col: st.column_config.NumberColumn(format="%.2f%%") for col in percent_columns},
            "Sharpe": st.column_config.NumberColumn(format="%.2f", help="Risk-free rate 0%."),
        },
    )


def render_calendar_tab(portfolio, benchmark):
    section_header("Monthly returns (%)", "Compounded month-end to month-end. The last column is the calendar-year total.")
    show_chart(charts.monthly_returns_heatmap(engine.calculate_monthly_returns_table(portfolio)))

    section_header("Calendar-year returns", "Portfolio vs benchmark. Partial first and last years run from the start date and up to the end date.")
    annual = engine.calculate_period_returns(portfolio, "YE")
    bench_annual = engine.calculate_period_returns(benchmark, "YE") if benchmark is not None else None
    show_chart(charts.annual_returns_chart(annual, bench_annual))

    if len(annual) > 1:
        c = st.columns(4)
        c[0].metric("Best year", fmt_pct(annual.max(), 1), border=True, delta=str(annual.idxmax().year), delta_color="off", delta_arrow="off")
        c[1].metric("Worst year", fmt_pct(annual.min(), 1), border=True, delta=str(annual.idxmin().year), delta_color="off", delta_arrow="off")
        c[2].metric("Positive years", f"{(annual > 0).sum()} of {len(annual)}", border=True)
        if bench_annual is not None:
            aligned = pd.concat([annual, bench_annual], axis=1, keys=["p", "b"]).dropna()
            c[3].metric("Years beating benchmark", f"{(aligned['p'] > aligned['b']).sum()} of {len(aligned)}", border=True)


def render_stress_tab(settings, data_version):
    periods = tuple((name, start, end) for name, (start, end) in CRISIS_PERIODS.items())
    summaries = analytics.crisis_summaries(
        data_version,
        analytics.weights_key(settings.weights),
        analytics.weights_key(settings.benchmark_weights),
        settings.rebalance,
        periods,
    )

    section_header(
        "Historical stress test",
        f"How the current allocation would have done in major market crises, compared with {settings.benchmark_name}. "
        f"The crisis windows are fixed and do not depend on the analysis period. The portfolio uses "
        f"{settings.rebalance.lower()} rebalancing.",
    )
    show_chart(charts.crisis_comparison_chart(summaries))

    columns = st.columns(len(periods))
    for column, (name, start, end) in zip(columns, periods):
        summary = summaries.get(name)
        with column, st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"{pd.Timestamp(start):%d %b %Y} → {pd.Timestamp(end):%d %b %Y} · {CRISIS_DESCRIPTIONS[name]}")
            if summary is None:
                st.warning("Not enough data for this window.", icon=":material/warning:")
                continue
            a, b = st.columns(2)
            a.metric("Portfolio return", fmt_pct(summary["Portfolio Return"], 1),
                     delta=fmt_pp_delta(summary["Portfolio Return"], summary["Benchmark Return"]),
                     help="Delta: percentage points vs the benchmark.")
            b.metric("Benchmark return", fmt_pct(summary["Benchmark Return"], 1))
            a.metric("Portfolio max DD", fmt_pct(summary["Portfolio Max DD"], 1),
                     delta=fmt_pp_delta(summary["Portfolio Max DD"], summary["Benchmark Max DD"]),
                     help="Delta: percentage points vs the benchmark.")
            b.metric("Benchmark max DD", fmt_pct(summary["Benchmark Max DD"], 1))
            show_chart(charts.crisis_path_chart(summary["Portfolio Path"], summary["Benchmark Path"]), key=f"crisis_{name}")
            st.caption(
                f"Volatility {fmt_pct(summary['Portfolio Volatility'], 1)} (benchmark "
                f"{fmt_pct(summary['Benchmark Volatility'], 1)}) · Best ETF **{summary['Best ETF']}** "
                f"({fmt_pct(summary['Asset Returns'][summary['Best ETF']], 1, signed=True)}) · Worst ETF "
                f"**{summary['Worst ETF']}** ({fmt_pct(summary['Asset Returns'][summary['Worst ETF']], 1, signed=True)})"
            )

    rows = [
        {
            "Crisis": name,
            "Trading days": s["Trading Days"],
            "Portfolio return": s["Portfolio Return"],
            "Benchmark return": s["Benchmark Return"],
            "Portfolio max DD": s["Portfolio Max DD"],
            "Benchmark max DD": s["Benchmark Max DD"],
            "Portfolio volatility": s["Portfolio Volatility"],
            "Benchmark volatility": s["Benchmark Volatility"],
            "Best ETF": s["Best ETF"],
            "Worst ETF": s["Worst ETF"],
        }
        for name, s in summaries.items() if s is not None
    ]
    if rows:
        with st.expander("Stress test table"):
            table = pd.DataFrame(rows)
            percent_columns = [k for k in table if k not in ("Crisis", "Trading days", "Best ETF", "Worst ETF")]
            table[percent_columns] = table[percent_columns] * 100
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={k: st.column_config.NumberColumn(format="%.2f%%") for k in percent_columns},
            )


@st.fragment
def render_monte_carlo_tab(portfolio, end_value):
    c1, c2, c3 = st.columns([1.2, 1, 1])
    horizon = c1.segmented_control("Horizon (years)", MC_HORIZONS, default=10, key="mc_horizon", required=True,
                                   format_func=lambda y: f"{y}Y")
    n_sims = c2.segmented_control("Simulations", MC_SIMULATIONS, default=1000, key="mc_sims", required=True,
                                  format_func=lambda n: f"{n:,}")
    starting_value = c3.number_input(
        "Starting value (€)", min_value=100.0, max_value=100_000_000.0, value=None, step=1_000.0, format="%.0f",
        placeholder=f"Backtest end value ({fmt_eur(end_value)})", key="mc_start",
        help="Leave empty to start from the portfolio's value at the end of the backtest.",
    )
    start_value = float(starting_value) if starting_value else float(end_value)

    with st.spinner("Running simulation…"):
        result = analytics.monte_carlo(portfolio, horizon, n_sims, start_value)

    if result is None:
        st.warning(
            "At least 30 daily returns are needed for a Monte Carlo simulation. Select a longer analysis period.",
            icon=":material/warning:",
        )
        return

    summary = engine.summarize_monte_carlo(result)
    c = st.columns(5)
    c[0].metric("Median ending value", fmt_eur(summary["Median Ending Value"]), border=True,
                delta=f"{fmt_pct(summary['Median CAGR'], 1)} p.a.", delta_color="off", delta_arrow="off")
    c[1].metric("Pessimistic (5th pct.)", fmt_eur(summary["P5 Ending Value"]), border=True)
    c[2].metric("Optimistic (95th pct.)", fmt_eur(summary["P95 Ending Value"]), border=True)
    c[3].metric("Probability of loss", fmt_pct(summary["Probability of Loss"], 1), border=True,
                help="Share of simulated paths that end below the starting value.")
    c[4].metric("Probability of doubling", fmt_pct(summary["Probability of Doubling"], 1), border=True,
                help="Share of simulated paths that end at twice the starting value or more.")

    section_header(
        f"Simulated value over {horizon} years",
        f"{n_sims:,} bootstrap paths, starting from {fmt_eur(start_value)}. Shaded bands show the "
        "25–75th and 5–95th percentiles, and the faint lines are individual sample paths.",
    )
    show_chart(charts.monte_carlo_chart(result, portfolio["Portfolio Value"]))

    checkpoints = [y for y in (1, 3, 5, 10, 15, 20, 25, 30) if y <= horizon]
    rows = []
    for years in checkpoints:
        row = result.percentiles.iloc[min(years * engine.TRADING_DAYS, len(result.percentiles) - 1)]
        rows.append({"Year": years, **{f"P{col[1:]}": row[col] for col in result.percentiles.columns}})
    with st.expander("Percentile table"):
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={col: st.column_config.NumberColumn(format="€%,.0f") for col in rows[0] if col != "Year"},
        )
    st.caption(
        "Method: each simulated day draws a random historical daily return from the backtest, with replacement. "
        "This keeps the fat tails of real returns but assumes the future looks like the past, with no "
        "autocorrelation or regime changes. The results show a range of outcomes and are not a forecast."
    )


def render_footer():
    st.divider()
    with st.expander("Methodology"):
        st.markdown(
            "- **Data**: daily adjusted close prices (dividends included) from Yahoo Finance, limited to "
            "dates where all four ETFs trade. A scheduled GitHub Action refreshes the data every week.\n"
            "- **Backtest**: the portfolio starts at its target weights. Each holding then drifts with its own "
            "returns, and the weights are reset on the first trading day of each new month, quarter or year "
            "(or never).\n"
            "- **CAGR** uses calendar time (days ÷ 365.25). **Volatility** = std(daily returns) × √252.\n"
            "- **Sharpe** = (CAGR − rf) ÷ volatility. **Sortino** = (CAGR − rf) ÷ annualized downside "
            "deviation of returns below the daily risk-free rate. **Calmar** = CAGR ÷ |max drawdown|.\n"
            "- **VaR / CVaR** are historical one-day 95% figures.\n"
            "- **Monte Carlo** bootstraps the portfolio's own daily returns (fixed seed, so results are reproducible)."
        )
    st.caption(
        "For education and analysis only. This is not investment advice. Past performance does not "
        "guarantee future results. Data: Yahoo Finance via yfinance."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    data_version = analytics.get_data_version()
    try:
        prices = analytics.load_prices(data_version)
    except Exception as exc:  # missing / corrupt CSV
        st.error(f"Could not load price data from `{engine.DATA_PATH.name}`: {exc}", icon=":material/error:")
        st.stop()
    if prices.empty or not set(TICKERS).issubset(prices.columns):
        st.error("The price dataset is empty or is missing required tickers.", icon=":material/error:")
        st.stop()

    min_date, max_date = prices.index.min().date(), prices.index.max().date()
    settings = render_sidebar(min_date, max_date)
    render_header(prices, settings)

    # ---- input validation: never show numbers for an invalid configuration ----
    if not settings.weights_valid:
        st.warning(
            f"Portfolio weights add up to **{settings.weight_total:.2f}%**, not 100%. No results are shown "
            "until the allocation is fixed. Adjust the weights or use **Normalize weights** in the sidebar.",
            icon=":material/balance:",
        )
        st.stop()
    if settings.start >= settings.end:
        st.error("The start date must be before the end date.", icon=":material/event_busy:")
        st.stop()
    n_days = len(engine.filter_prices_by_date(prices, settings.start, settings.end))
    if n_days < MIN_TRADING_DAYS:
        st.error(
            f"The selected period has only {n_days} trading days. Select at least {MIN_TRADING_DAYS}.",
            icon=":material/event_busy:",
        )
        st.stop()

    weights = analytics.weights_key(settings.weights)
    try:
        portfolio = analytics.backtest(
            data_version, settings.start, settings.end, weights, settings.initial_investment, settings.rebalance
        )
        m = analytics.metrics(portfolio, settings.risk_free_rate)
    except ValueError as exc:
        st.error(f"The backtest could not be calculated: {exc}", icon=":material/error:")
        st.stop()

    benchmark, bm = None, None
    try:
        benchmark = analytics.backtest(
            data_version, settings.start, settings.end, analytics.weights_key(settings.benchmark_weights),
            settings.initial_investment, settings.rebalance,
        )
        bm = analytics.metrics(benchmark, settings.risk_free_rate)
    except ValueError as exc:
        st.warning(f"Benchmark unavailable ({exc}). Showing the portfolio only.", icon=":material/warning:")

    render_kpis(portfolio, m, bm, settings.benchmark_name, settings.risk_free_rate)
    if (settings.end - settings.start).days < 365:
        st.caption(
            ":material/info: The period is shorter than one year, so CAGR and the ratios based on it are "
            "annualized extrapolations."
        )

    tabs = st.tabs([
        ":material/show_chart: Performance",
        ":material/trending_down: Risk & drawdown",
        ":material/timeline: Rolling",
        ":material/hub: Diversification",
        ":material/calendar_month: Calendar returns",
        ":material/thunderstorm: Stress test",
        ":material/casino: Monte Carlo",
    ])
    with tabs[0]:
        render_performance_tab(settings, portfolio, benchmark, m, bm)
    with tabs[1]:
        render_risk_tab(portfolio, benchmark, m, settings.benchmark_name)
    with tabs[2]:
        render_rolling_tab(portfolio, benchmark, settings.risk_free_rate)
    with tabs[3]:
        render_diversification_tab(settings, data_version)
    with tabs[4]:
        render_calendar_tab(portfolio, benchmark)
    with tabs[5]:
        render_stress_tab(settings, data_version)
    with tabs[6]:
        render_monte_carlo_tab(portfolio, m["End Value"])

    render_footer()


main()
