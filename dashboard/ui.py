"""Formatting helpers, small layout components and the (minimal) custom CSS."""

import math

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] p {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.75;}
.app-eyebrow {font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: #9085e9; font-weight: 600; margin-bottom: 0.2rem;}
.app-subtitle {font-size: 1.05rem; opacity: 0.8; margin-top: -0.6rem;}
.section-note {font-size: 0.85rem; opacity: 0.7;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def fmt_pct(value, decimals: int = 2, signed: bool = False) -> str:
    if is_missing(value):
        return "–"
    return f"{value:+.{decimals}%}" if signed else f"{value:.{decimals}%}"


def fmt_ratio(value) -> str:
    return "–" if is_missing(value) else f"{value:.2f}"


def fmt_eur(value, decimals: int = 0) -> str:
    return "–" if is_missing(value) else f"€{value:,.{decimals}f}"


def fmt_pp_delta(portfolio_value, benchmark_value) -> str | None:
    """Difference in percentage points, e.g. '+1.25 pp'."""
    if is_missing(portfolio_value) or is_missing(benchmark_value):
        return None
    return f"{(portfolio_value - benchmark_value) * 100:+.2f} pp"


def fmt_ratio_delta(portfolio_value, benchmark_value) -> str | None:
    if is_missing(portfolio_value) or is_missing(benchmark_value):
        return None
    return f"{portfolio_value - benchmark_value:+.2f}"


def section_header(title: str, note: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)
