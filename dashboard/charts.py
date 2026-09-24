"""Plotly figure builders. Each returns a go.Figure and never raises on empty input."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.config import (
    ASSET_COLORS,
    BENCHMARK_COLOR,
    DIVERGING_SCALE,
    GRID_COLOR,
    NEGATIVE_COLOR,
    PORTFOLIO_COLOR,
    WARNING_COLOR,
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MAX_CHART_POINTS = 1500


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def downsample(data, max_points: int = MAX_CHART_POINTS):
    """Stride-downsample a long daily series for the browser, always keeping the last point."""
    n = len(data)
    if n <= max_points:
        return data
    step = int(np.ceil(n / max_points))
    positions = np.unique(np.append(np.arange(0, n, step), n - 1))
    return data.iloc[positions]


def _base_layout(fig: go.Figure, height: int = 400, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, showline=False)
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=BENCHMARK_COLOR))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height=height, legend=False)


def growth_chart(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    benchmark_name: str,
    mode: str = "value",
) -> go.Figure:
    """Portfolio vs benchmark as currency value or cumulative return."""
    if portfolio is None or portfolio.empty:
        return empty_figure("No portfolio data for the selected settings")

    def transform(frame):
        values = downsample(frame["Portfolio Value"])
        return values if mode == "value" else values / frame["Portfolio Value"].iloc[0] - 1

    value_format = "€%{y:,.0f}" if mode == "value" else "%{y:+.1%}"
    fig = go.Figure()
    if benchmark is not None and not benchmark.empty:
        series = transform(benchmark)
        fig.add_trace(go.Scatter(
            x=series.index, y=series, name=f"Benchmark · {benchmark_name}", mode="lines",
            line=dict(color=BENCHMARK_COLOR, width=1.6, dash="dot"),
            hovertemplate=f"{value_format}<extra>Benchmark</extra>",
        ))
    series = transform(portfolio)
    fig.add_trace(go.Scatter(
        x=series.index, y=series, name="Portfolio", mode="lines",
        line=dict(color=PORTFOLIO_COLOR, width=2.2),
        fill="tozeroy" if mode == "value" else None,
        fillcolor=_hex_to_rgba(PORTFOLIO_COLOR, 0.08),
        hovertemplate=f"{value_format}<extra>Portfolio</extra>",
    ))
    _base_layout(fig, height=430)
    fig.update_xaxes(hoverformat="%d %b %Y")
    if mode == "value":
        fig.update_yaxes(tickprefix="€", tickformat=",.0f", rangemode="tozero")
    else:
        fig.update_yaxes(tickformat=".0%")
        fig.add_hline(y=0, line=dict(color=BENCHMARK_COLOR, width=1, dash="dot"), opacity=0.5)
    return fig


def drawdown_chart(portfolio: pd.DataFrame, benchmark: pd.DataFrame | None, benchmark_name: str) -> go.Figure:
    if portfolio is None or portfolio.empty:
        return empty_figure("No drawdown data")

    fig = go.Figure()
    if benchmark is not None and not benchmark.empty:
        dd = downsample(benchmark["Drawdown"])
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=f"Benchmark · {benchmark_name}", mode="lines",
            line=dict(color=BENCHMARK_COLOR, width=1.4, dash="dot"),
            hovertemplate="%{y:.2%}<extra>Benchmark</extra>",
        ))
    dd = downsample(portfolio["Drawdown"])
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Portfolio", mode="lines",
        line=dict(color=NEGATIVE_COLOR, width=1.8),
        fill="tozeroy", fillcolor=_hex_to_rgba(NEGATIVE_COLOR, 0.18),
        hovertemplate="%{y:.2%}<extra>Portfolio</extra>",
    ))

    trough_date = portfolio["Drawdown"].idxmin()
    trough = portfolio["Drawdown"].min()
    if trough < 0:
        fig.add_annotation(
            x=trough_date, y=trough, text=f"Max drawdown {trough:.1%}", showarrow=True,
            arrowhead=0, ax=60, ay=10, font=dict(size=12),
        )
    _base_layout(fig, height=360)
    fig.update_xaxes(hoverformat="%d %b %Y")
    fig.update_yaxes(tickformat=".0%")
    return fig


def return_distribution_chart(daily_returns: pd.Series, var_95: float, cvar_95: float) -> go.Figure:
    returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return empty_figure("No daily returns to plot")

    fig = go.Figure(go.Histogram(
        x=returns, nbinsx=120, name="Daily returns",
        marker=dict(color=_hex_to_rgba(PORTFOLIO_COLOR, 0.75), line=dict(width=0)),
        hovertemplate="Return %{x}<br>%{y} days<extra></extra>",
        xbins=dict(size=max((returns.max() - returns.min()) / 120, 1e-4)),
    ))
    for value, label, color, ypos in (
        (var_95, "VaR 95%", WARNING_COLOR, 1.0),
        (cvar_95, "CVaR 95%", NEGATIVE_COLOR, 0.88),
    ):
        if pd.notna(value):
            fig.add_vline(x=value, line=dict(color=color, width=2, dash="dash"))
            fig.add_annotation(
                x=value, y=ypos, yref="paper", text=f"{label} {value:.2%}", showarrow=False,
                xanchor="right", font=dict(color=color, size=12), bgcolor="rgba(11,15,23,0.7)",
            )
    _base_layout(fig, height=360, legend=False)
    fig.update_layout(hovermode="closest", bargap=0.05)
    fig.update_xaxes(tickformat=".1%", range=[returns.quantile(0.0005) * 1.3, returns.quantile(0.9995) * 1.3])
    fig.update_yaxes(title_text="Trading days")
    return fig


def rolling_chart(
    portfolio_rolling: pd.DataFrame,
    benchmark_rolling: pd.DataFrame | None,
    column: str,
    percent: bool,
    window_label: str,
) -> go.Figure:
    series = portfolio_rolling[column].dropna() if column in portfolio_rolling else pd.Series(dtype=float)
    if series.empty:
        return empty_figure(f"Not enough data for a {window_label} rolling window. Select a longer period.")

    value_format = "%{y:.2%}" if percent else "%{y:.2f}"
    fig = go.Figure()
    if benchmark_rolling is not None and column in benchmark_rolling:
        bench = downsample(benchmark_rolling[column].dropna())
        if not bench.empty:
            fig.add_trace(go.Scatter(
                x=bench.index, y=bench, name="Benchmark", mode="lines",
                line=dict(color=BENCHMARK_COLOR, width=1.4, dash="dot"),
                hovertemplate=f"{value_format}<extra>Benchmark</extra>",
            ))
    series = downsample(series)
    fig.add_trace(go.Scatter(
        x=series.index, y=series, name="Portfolio", mode="lines",
        line=dict(color=PORTFOLIO_COLOR, width=2),
        hovertemplate=f"{value_format}<extra>Portfolio</extra>",
    ))
    fig.add_hline(y=0, line=dict(color=BENCHMARK_COLOR, width=1), opacity=0.4)
    _base_layout(fig, height=400)
    fig.update_xaxes(hoverformat="%d %b %Y")
    fig.update_yaxes(tickformat=".0%" if percent else ".2f")
    return fig


def correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    if correlation is None or correlation.empty or correlation.isna().all().all():
        return empty_figure("Correlation unavailable for this period")

    assets = list(correlation.columns)
    fig = go.Figure(go.Heatmap(
        z=correlation.values, x=assets, y=assets,
        text=correlation.map(lambda v: f"{v:.2f}").values, texttemplate="%{text}",
        textfont=dict(size=15), colorscale=DIVERGING_SCALE, zmin=-1, zmax=1, xgap=3, ygap=3,
        colorbar=dict(title="ρ", thickness=12, len=0.8, outlinewidth=0),
        hovertemplate="%{y} ↔ %{x}<br>Correlation %{z:.3f}<extra></extra>",
    ))
    _base_layout(fig, height=380, legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def allocation_donut(weights: dict) -> go.Figure:
    held = {t: w for t, w in weights.items() if w > 0}
    if not held:
        return empty_figure("No allocation")

    fig = go.Figure(go.Pie(
        labels=list(held), values=list(held.values()), hole=0.62, sort=False, direction="clockwise",
        marker=dict(colors=[ASSET_COLORS.get(t, BENCHMARK_COLOR) for t in held], line=dict(color="#0b0f17", width=3)),
        texttemplate="<b>%{label}</b><br>%{percent:.1%}", textposition="outside",
        hovertemplate="<b>%{label}</b><br>Weight %{percent:.2%}<extra></extra>",
    ))
    fig.add_annotation(text="Target<br>allocation", showarrow=False, font=dict(size=14, color=BENCHMARK_COLOR))
    _base_layout(fig, height=380, legend=False)
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40), hovermode="closest")
    return fig


def monthly_returns_heatmap(table: pd.DataFrame) -> go.Figure:
    if table is None or table.empty:
        return empty_figure("At least two month-ends are needed for monthly returns")

    z = table.to_numpy(dtype=float) * 100
    text = [["" if np.isnan(v) else f"{v:.1f}" for v in row] for row in z]
    limit = max(np.nanpercentile(np.abs(z[:, :12]), 98) if np.isfinite(z[:, :12]).any() else 1, 1)
    fig = go.Figure(go.Heatmap(
        z=z, x=MONTH_LABELS + ["Year"], y=[str(y) for y in table.index],
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale=DIVERGING_SCALE, zmid=0, zmin=-limit, zmax=limit, xgap=2, ygap=2,
        colorbar=dict(title="%", thickness=12, len=0.6, outlinewidth=0),
        hovertemplate="%{x} %{y}<br>Return %{z:.2f}%<extra></extra>",
    ))
    fig.add_vrect(x0=11.5, x1=12.5, line=dict(color=BENCHMARK_COLOR, width=1.5), layer="above")
    _base_layout(fig, height=max(320, 26 * len(table) + 90), legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed", showgrid=False, dtick=1)
    return fig


def annual_returns_chart(portfolio_annual: pd.Series, benchmark_annual: pd.Series | None) -> go.Figure:
    if portfolio_annual is None or portfolio_annual.empty:
        return empty_figure("No annual returns for this period")

    fig = go.Figure()
    years = [str(d.year) for d in portfolio_annual.index]
    fig.add_trace(go.Bar(
        x=years, y=portfolio_annual.values, name="Portfolio", marker=dict(color=PORTFOLIO_COLOR, cornerradius=3),
        hovertemplate="%{y:+.2%}<extra>Portfolio</extra>",
    ))
    if benchmark_annual is not None and not benchmark_annual.empty:
        fig.add_trace(go.Bar(
            x=[str(d.year) for d in benchmark_annual.index], y=benchmark_annual.values, name="Benchmark",
            marker=dict(color=BENCHMARK_COLOR, cornerradius=3),
            hovertemplate="%{y:+.2%}<extra>Benchmark</extra>",
        ))
    _base_layout(fig, height=360)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0, line=dict(color=BENCHMARK_COLOR, width=1), opacity=0.5)
    return fig


def crisis_path_chart(portfolio_path: pd.Series, benchmark_path: pd.Series, height: int = 220) -> go.Figure:
    if portfolio_path is None or portfolio_path.empty:
        return empty_figure("No data", height=height)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=benchmark_path.index, y=benchmark_path - 1, name="Benchmark", mode="lines",
        line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dot"),
        hovertemplate="%{y:+.1%}<extra>Benchmark</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=portfolio_path.index, y=portfolio_path - 1, name="Portfolio", mode="lines",
        line=dict(color=PORTFOLIO_COLOR, width=2.2),
        hovertemplate="%{y:+.1%}<extra>Portfolio</extra>",
    ))
    fig.add_hline(y=0, line=dict(color=BENCHMARK_COLOR, width=1), opacity=0.4)
    _base_layout(fig, height=height, legend=False)
    fig.update_layout(margin=dict(l=5, r=5, t=10, b=5))
    fig.update_xaxes(hoverformat="%d %b %Y", nticks=5)
    fig.update_yaxes(tickformat=".0%", nticks=5)
    return fig


def crisis_comparison_chart(summaries: dict) -> go.Figure:
    rows = {name: s for name, s in summaries.items() if s is not None}
    if not rows:
        return empty_figure("No stress periods fall inside the available data")

    names = list(rows)
    fig = go.Figure()
    for key, label, color in (
        ("Portfolio Return", "Portfolio", PORTFOLIO_COLOR),
        ("Benchmark Return", "Benchmark", BENCHMARK_COLOR),
    ):
        values = [rows[n][key] for n in names]
        fig.add_trace(go.Bar(
            y=names, x=values, name=label, orientation="h",
            marker=dict(color=color, cornerradius=3),
            text=[f"{v:+.1%}" for v in values], textposition="inside", insidetextanchor="start",
            hovertemplate="%{x:+.2%}<extra>" + label + "</extra>",
        ))
    _base_layout(fig, height=110 + 70 * len(names))
    fig.update_layout(barmode="group", bargap=0.35, hovermode="y unified")
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.add_vline(x=0, line=dict(color=BENCHMARK_COLOR, width=1), opacity=0.6)
    return fig


def monte_carlo_chart(result, history: pd.Series) -> go.Figure:
    if result is None:
        return empty_figure("At least 30 daily returns are needed for a Monte Carlo simulation")

    pct = downsample(result.percentiles, max_points=600)
    fig = go.Figure()

    hist = downsample(history.iloc[-504:], max_points=300)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist, name="Historical (last 2Y)", mode="lines",
        line=dict(color=BENCHMARK_COLOR, width=1.8),
        hovertemplate="€%{y:,.0f}<extra>Historical</extra>",
    ))

    paths = downsample(result.sample_paths, max_points=300)
    for i, column in enumerate(paths.columns):
        fig.add_trace(go.Scatter(
            x=paths.index, y=paths[column], mode="lines", hoverinfo="skip",
            line=dict(color="rgba(200,205,220,0.10)", width=1),
            name="Sample paths", legendgroup="paths", showlegend=i == 0,
        ))

    band_fill = {("p95", "p5"): 0.14, ("p75", "p25"): 0.30}
    for (upper, lower), alpha in band_fill.items():
        label = f"{lower[1:]}–{upper[1:]}th percentile"
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct[upper], mode="lines", line=dict(width=0),
            showlegend=False, hovertemplate=f"€%{{y:,.0f}}<extra>P{upper[1:]}</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct[lower], mode="lines", line=dict(width=0), name=label,
            fill="tonexty", fillcolor=_hex_to_rgba(PORTFOLIO_COLOR, alpha),
            hovertemplate=f"€%{{y:,.0f}}<extra>P{lower[1:]}</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=pct.index, y=pct["p50"], name="Median", mode="lines",
        line=dict(color=PORTFOLIO_COLOR, width=2.6),
        hovertemplate="€%{y:,.0f}<extra>Median</extra>",
    ))
    _base_layout(fig, height=460)
    fig.update_xaxes(hoverformat="%b %Y")
    # Scale to the percentile fan so a few extreme sample paths do not flatten it.
    y_max = max(float(pct["p95"].max()), float(hist.max())) * 1.1
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", range=[0, y_max])
    return fig
