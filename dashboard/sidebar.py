"""Sidebar controls and the settings object they produce."""

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from dashboard.config import (
    ASSETS,
    BENCHMARKS,
    CRISIS_PERIODS,
    CUSTOM_PRESET,
    DEFAULT_BENCHMARK,
    DEFAULT_PRESET,
    DEFAULT_REBALANCE,
    PORTFOLIO_PRESETS,
    QUICK_PERIODS,
    REBALANCE_OPTIONS,
    TICKERS,
)

WEIGHT_TOLERANCE = 0.005  # percentage points


@dataclass(frozen=True)
class Settings:
    weights: dict[str, float]          # in percent, as entered
    start: date
    end: date
    initial_investment: float
    risk_free_rate: float              # decimal, e.g. 0.02
    rebalance: str                     # "Never" | "Monthly" | "Quarterly" | "Annually"
    benchmark_name: str
    benchmark_weights: dict[str, float]

    @property
    def weight_total(self) -> float:
        return sum(self.weights.values())

    @property
    def weights_valid(self) -> bool:
        return abs(self.weight_total - 100) <= WEIGHT_TOLERANCE and any(w > 0 for w in self.weights.values())


def _weight_key(ticker: str) -> str:
    return ticker.lower()


def _current_weights() -> dict[str, float]:
    return {t: float(st.session_state.get(_weight_key(t)) or 0.0) for t in TICKERS}


def _matching_preset(weights: dict[str, float]) -> str:
    for name, preset in PORTFOLIO_PRESETS.items():
        if all(abs(weights[t] - preset.get(t, 0)) < 1e-9 for t in TICKERS):
            return name
    return CUSTOM_PRESET


def normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    """Scale weights proportionally to 100 %, rounded to 2 decimals; the rounding residual goes to the largest weight."""
    total = sum(weights.values())
    if total <= 0:
        return {t: round(100 / len(weights), 2) for t in weights}
    scaled = {t: round(w / total * 100, 2) for t, w in weights.items()}
    residual = round(100 - sum(scaled.values()), 2)
    largest = max(scaled, key=scaled.get)
    scaled[largest] = round(scaled[largest] + residual, 2)
    return scaled


def _period_dates(label: str, min_date: date, max_date: date) -> tuple[date, date]:
    end = pd.Timestamp(max_date)
    if label in CRISIS_PERIODS:
        start_str, end_str = CRISIS_PERIODS[label]
        return max(pd.Timestamp(start_str).date(), min_date), min(pd.Timestamp(end_str).date(), max_date)
    if label == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif label.endswith("Y") and label[:-1].isdigit():
        start = end - pd.DateOffset(years=int(label[:-1]))
    else:
        start = pd.Timestamp(min_date)
    return max(start.date(), min_date), max_date


# ---- callbacks (run before the next script execution) ----

def _apply_preset():
    preset = PORTFOLIO_PRESETS.get(st.session_state["preset"])
    if preset:
        for t in TICKERS:
            st.session_state[_weight_key(t)] = float(preset.get(t, 0))


def _on_weight_change():
    st.session_state["preset"] = _matching_preset(_current_weights())


def _normalize():
    for t, w in normalized_weights(_current_weights()).items():
        st.session_state[_weight_key(t)] = w
    st.session_state["preset"] = _matching_preset(_current_weights())


def _apply_period(min_date: date, max_date: date):
    label = st.session_state["period"]
    if label != "Custom":
        st.session_state["start"], st.session_state["end"] = _period_dates(label, min_date, max_date)


def _on_date_change(min_date: date, max_date: date):
    current = (st.session_state.get("start"), st.session_state.get("end"))
    for label in QUICK_PERIODS[:-1]:
        if _period_dates(label, min_date, max_date) == current:
            st.session_state["period"] = label
            return
    st.session_state["period"] = "Custom"


def _default_values(min_date: date, max_date: date) -> dict:
    return {
        **{_weight_key(t): float(PORTFOLIO_PRESETS[DEFAULT_PRESET][t]) for t in TICKERS},
        "start": min_date,
        "end": max_date,
        "inv": 10_000.0,
        "rf": 0.0,
        "reb": DEFAULT_REBALANCE,
        "bm": DEFAULT_BENCHMARK,
    }


def _reset(min_date: date, max_date: date):
    # Assign explicit defaults: bound widgets would otherwise re-read stale URL parameters.
    st.session_state.update(_default_values(min_date, max_date))
    st.session_state["preset"] = DEFAULT_PRESET
    st.session_state["period"] = "Full history"


def _values_from_url(min_date: date, max_date: date) -> dict:
    """Parse settings from a shared link so derived selectors (preset, period) match on first load."""
    parsers = {
        **{_weight_key(t): lambda v: min(max(float(v), 0.0), 100.0) for t in TICKERS},
        "start": lambda v: min(max(date.fromisoformat(v), min_date), max_date),
        "end": lambda v: min(max(date.fromisoformat(v), min_date), max_date),
        "inv": float,
        "rf": lambda v: min(max(float(v), 0.0), 6.0),
        "reb": lambda v: v if v in REBALANCE_OPTIONS else DEFAULT_REBALANCE,
        "bm": lambda v: v if v in BENCHMARKS else DEFAULT_BENCHMARK,
    }
    values = {}
    for key, parse in parsers.items():
        if key in st.query_params:
            try:
                values[key] = parse(st.query_params[key])
            except (TypeError, ValueError):
                pass
    return values


def _init_state(min_date: date, max_date: date):
    """Seed session state once, preferring values from a shared URL over the defaults."""
    if "_initialized" not in st.session_state:
        st.session_state.update(_values_from_url(min_date, max_date))
        st.session_state["_initialized"] = True
    for key, value in _default_values(min_date, max_date).items():
        st.session_state.setdefault(key, value)
    # Derived selectors always reflect the current weights / dates (e.g. after a shared link).
    st.session_state["preset"] = _matching_preset(_current_weights())
    if st.session_state.get("period") is None:
        _on_date_change(min_date, max_date)


def render_sidebar(min_date: date, max_date: date) -> Settings:
    _init_state(min_date, max_date)

    with st.sidebar:
        st.markdown("### Portfolio configuration")

        # ---- Allocation ----
        st.markdown("**Allocation**")
        st.selectbox(
            "Preset", [*PORTFOLIO_PRESETS, CUSTOM_PRESET], key="preset", on_change=_apply_preset,
            help="Classic multi-asset allocations. Editing any weight switches to Custom.",
        )
        columns = st.columns(2)
        for i, ticker in enumerate(TICKERS):
            with columns[i % 2]:
                st.number_input(
                    f"{ticker} (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f",
                    key=_weight_key(ticker), on_change=_on_weight_change,
                    help=ASSETS[ticker]["description"], bind="query-params",
                )

        weights = _current_weights()
        total = sum(weights.values())
        if abs(total - 100) <= WEIGHT_TOLERANCE:
            st.success(f"Total allocation: **{total:.2f}%**", icon=":material/check_circle:")
        else:
            st.error(
                f"Total allocation is **{total:.2f}%**. Weights must add up to 100%.",
                icon=":material/error:",
            )
            st.button(
                "Normalize weights to 100%", on_click=_normalize, type="primary", width="stretch",
                disabled=total <= 0,
                help="Scales every weight proportionally so the total is exactly 100%.",
            )

        # ---- Period ----
        st.divider()
        st.markdown("**Analysis period**")
        st.selectbox(
            "Quick select", QUICK_PERIODS, key="period",
            on_change=_apply_period, args=(min_date, max_date),
        )
        c1, c2 = st.columns(2)
        c1.date_input(
            "Start", min_value=min_date, max_value=max_date, key="start", format="YYYY-MM-DD",
            on_change=_on_date_change, args=(min_date, max_date), bind="query-params",
        )
        c2.date_input(
            "End", min_value=min_date, max_value=max_date, key="end", format="YYYY-MM-DD",
            on_change=_on_date_change, args=(min_date, max_date), bind="query-params",
        )

        # ---- Simulation ----
        st.divider()
        st.markdown("**Simulation settings**")
        st.number_input(
            "Initial investment (€)", min_value=100.0, max_value=100_000_000.0, step=1_000.0,
            format="%.0f", key="inv", bind="query-params",
        )
        st.selectbox(
            "Rebalancing", REBALANCE_OPTIONS, key="reb",
            help="Weights are reset to target on the first trading day of each new period.",
            bind="query-params",
        )
        st.selectbox("Benchmark", list(BENCHMARKS), key="bm", bind="query-params")
        st.slider(
            "Risk-free rate (%)", min_value=0.0, max_value=6.0, step=0.1, format="%.1f%%", key="rf",
            help="Annual rate subtracted from CAGR in the Sharpe and Sortino ratios.",
            bind="query-params",
        )

        st.divider()
        st.button("Reset all settings", on_click=_reset, args=(min_date, max_date), width="stretch", icon=":material/restart_alt:")
        st.caption("Every setting is stored in the page URL, so you can copy it to share a configuration.")

    start, end = st.session_state["start"], st.session_state["end"]
    benchmark_name = st.session_state["bm"] if st.session_state["bm"] in BENCHMARKS else DEFAULT_BENCHMARK
    return Settings(
        weights=weights,
        start=start,
        end=end,
        initial_investment=float(st.session_state["inv"]),
        risk_free_rate=float(st.session_state["rf"]) / 100,
        rebalance=st.session_state["reb"] or DEFAULT_REBALANCE,
        benchmark_name=benchmark_name,
        benchmark_weights=BENCHMARKS[benchmark_name],
    )
