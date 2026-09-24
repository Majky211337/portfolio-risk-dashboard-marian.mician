"""
2. Source data check: stored production prices vs a fresh Yahoo Finance download.

Price definitions:
- The dataset stores Yahoo's "Adj Close" downloaded with auto_adjust=False
  (see download_daily_data.py), i.e. close prices back-adjusted for splits AND
  dividends/distributions.
- yfinance auto_adjust=True rewrites OHLC with the same adjustment factor, so its
  "Close" is the same series as "Adj Close". Both are compared below.
- The raw "Close" (unadjusted) is a different definition and must NOT match on
  older dates for distribution-paying ETFs (SPY, AGG). This proves we are not
  accidentally comparing like with unlike.

Yahoo re-applies dividend adjustments to the whole history each time a fund goes
ex-dividend. Adjusted LEVELS may therefore drift by a constant factor between
refreshes, while RETURNS (price ratios) stay identical. The strict test is on
ratios; levels are reported and held to a loose tolerance.

Nothing in this module writes to the production dataset.
"""

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as engine

TICKERS = ["SPY", "GLD", "AGG", "DBC"]
N_SAMPLE_DATES = 25
SEED = 20260924

yf = pytest.importorskip("yfinance")
pytestmark = pytest.mark.network


def _download(auto_adjust: bool) -> pd.DataFrame:
    try:
        data = yf.download(
            TICKERS, start="2006-01-01", auto_adjust=auto_adjust, progress=False, threads=False
        )
    except Exception as exc:  # network / Yahoo outage
        pytest.skip(f"Yahoo Finance unavailable: {exc}")
    if data is None or data.empty:
        pytest.skip("Yahoo Finance returned no data")
    return data


@pytest.fixture(scope="module")
def fresh_unadjusted_download():
    return _download(auto_adjust=False)


@pytest.fixture(scope="module")
def fresh_auto_adjusted_close():
    return _download(auto_adjust=True)["Close"][TICKERS]


@pytest.fixture(scope="module")
def stored():
    return engine.load_prices()


@pytest.fixture(scope="module")
def sample_dates(stored, fresh_unadjusted_download):
    fresh = fresh_unadjusted_download["Adj Close"][TICKERS].dropna()
    # Exclude the final stored date: Yahoo can revise the latest bar for a day or so.
    overlap = stored.index.intersection(fresh.index)[:-1]
    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(overlap), size=min(N_SAMPLE_DATES, len(overlap)), replace=False)
    # Always include the first common date and a date inside each crisis window.
    fixed = pd.DatetimeIndex(["2006-02-06", "2008-10-10", "2020-03-16", "2022-06-16"])
    return overlap[np.sort(picked)].union(fixed.intersection(overlap))


def _comparison_table(stored, fresh, dates) -> pd.DataFrame:
    rows = []
    for date in dates:
        for ticker in TICKERS:
            s, f = stored.at[date, ticker], fresh.at[date, ticker]
            rows.append({"date": date.date(), "ticker": ticker, "stored": s, "yahoo": f,
                         "abs_diff": s - f, "pct_diff": s / f - 1})
    return pd.DataFrame(rows)


def test_adjusted_close_levels_match(stored, fresh_unadjusted_download, sample_dates):
    fresh = fresh_unadjusted_download["Adj Close"][TICKERS]
    table = _comparison_table(stored, fresh, sample_dates)
    print(f"\nStored vs fresh Yahoo 'Adj Close' on {len(sample_dates)} dates (seed {SEED}):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    worst = table.loc[table["pct_diff"].abs().idxmax()]
    print(f"Largest level difference: {worst['ticker']} {worst['date']} {worst['pct_diff']:+.4%}")
    # Levels may shift by one quarter's dividend adjustment between refreshes (< 1%).
    assert table["pct_diff"].abs().max() < 0.01


def test_adjusted_returns_match_exactly(stored, fresh_unadjusted_download, sample_dates):
    """Price ratios between sampled dates are what the backtest uses; they must agree tightly."""
    fresh = fresh_unadjusted_download["Adj Close"][TICKERS]
    for ticker in TICKERS:
        ratio = stored.loc[sample_dates, ticker] / fresh.loc[sample_dates, ticker]
        dispersion = ratio.max() / ratio.min() - 1
        print(f"{ticker}: stored/yahoo ratio in [{ratio.min():.8f}, {ratio.max():.8f}], dispersion {dispersion:.2e}")
        assert dispersion < 1e-4, f"{ticker} adjustment is not a constant factor"

    common = stored.index.intersection(fresh.dropna().index)[:-1]
    stored_returns = stored.loc[common].pct_change().dropna()
    fresh_returns = fresh.loc[common].pct_change().dropna()
    max_return_diff = (stored_returns - fresh_returns).abs().max()
    print("Max |daily return difference| over all common dates:", max_return_diff.round(8).to_dict())
    assert (max_return_diff < 1e-4).all()


def test_auto_adjust_true_close_equals_adj_close(stored, fresh_unadjusted_download, fresh_auto_adjusted_close):
    adj_close = fresh_unadjusted_download["Adj Close"][TICKERS].dropna()
    auto_close = fresh_auto_adjusted_close.dropna()
    # Settled bars only: a live intraday bar changes between the two downloads.
    common = adj_close.index.intersection(auto_close.index)
    common = common[common <= stored.index.max()]
    rel = (auto_close.loc[common] / adj_close.loc[common] - 1).abs().max()
    print("\nauto_adjust=True Close vs Adj Close, max relative difference:", rel.to_dict())
    assert (rel < 1e-5).all()  # yfinance stores prices as float32 (~7 significant digits)


def test_raw_close_is_a_different_definition(stored, fresh_unadjusted_download):
    """Unadjusted Close ignores distributions, so it must be well above Adj Close on old dates."""
    raw_close = fresh_unadjusted_download["Close"][TICKERS]
    date = pd.Timestamp("2010-01-04")
    gap = raw_close.loc[date] / stored.loc[date] - 1
    print("\nRaw Close / stored Adj Close - 1 on 2010-01-04:", gap.round(4).to_dict())
    assert gap["SPY"] > 0.10 and gap["AGG"] > 0.10   # years of dividends / coupons
    assert abs(gap["GLD"]) < 1e-6                   # GLD pays no distributions
