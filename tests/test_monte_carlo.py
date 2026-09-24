"""9. Monte Carlo simulation: reproducibility, dimensions, return sampling and sanity of outputs."""

import numpy as np
import pandas as pd
import pytest

import portfolio_engine as engine
from tests.conftest import DEFAULT_WEIGHTS


@pytest.fixture(scope="module")
def portfolio(prices):
    return engine.calculate_rebalanced_portfolio(prices, DEFAULT_WEIGHTS, 10_000, "annually")


def _synthetic_portfolio(returns: np.ndarray, start_value: float = 100.0) -> pd.DataFrame:
    values = start_value * np.concatenate(([1.0], np.cumprod(1 + returns)))
    pv = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))
    return pd.DataFrame({"Portfolio Value": pv, "Daily Return": pv.pct_change(), "Drawdown": pv / pv.cummax() - 1})


def test_fixed_seed_is_reproducible(portfolio):
    a = engine.run_monte_carlo(portfolio, 10, 1000, n_sample_paths=20, seed=42)
    b = engine.run_monte_carlo(portfolio, 10, 1000, n_sample_paths=20, seed=42)
    pd.testing.assert_frame_equal(a.percentiles, b.percentiles)
    np.testing.assert_array_equal(a.terminal_values, b.terminal_values)
    pd.testing.assert_frame_equal(a.sample_paths, b.sample_paths)
    c = engine.run_monte_carlo(portfolio, 10, 1000, seed=7)
    assert not np.array_equal(a.terminal_values, c.terminal_values)


def test_default_seed_used_by_app_is_fixed(portfolio):
    np.testing.assert_array_equal(
        engine.run_monte_carlo(portfolio, 5, 500).terminal_values,
        engine.run_monte_carlo(portfolio, 5, 500).terminal_values,
    )


@pytest.mark.parametrize("horizon, sims", [(5, 500), (10, 1000), (20, 2000), (30, 2000)])
def test_dimensions(portfolio, horizon, sims):
    result = engine.run_monte_carlo(portfolio, horizon, sims, n_sample_paths=40)
    steps = horizon * 252
    assert result.n_simulations == sims and result.horizon_years == horizon
    assert result.terminal_values.shape == (sims,)
    assert result.percentiles.shape == (steps + 1, 5)
    assert result.sample_paths.shape == (steps + 1, 40)
    assert result.percentiles.index[0] == portfolio.index[-1]
    assert result.percentiles.index.is_monotonic_increasing and result.percentiles.index.is_unique
    assert (result.percentiles.iloc[0] == result.starting_value).all()
    assert result.starting_value == pytest.approx(portfolio["Portfolio Value"].iloc[-1])


def test_simulation_count_is_capped(portfolio):
    result = engine.run_monte_carlo(portfolio, 5, 50_000)
    assert result.n_simulations == engine.MC_MAX_SIMS == 2000
    assert result.terminal_values.shape == (engine.MC_MAX_SIMS,)


def test_samples_daily_returns_not_price_levels():
    """With only two possible historical returns, every simulated step must be one of them."""
    returns = np.tile([0.01, -0.005], 100)
    result = engine.run_monte_carlo(_synthetic_portfolio(returns), 1, 300, n_sample_paths=300)
    steps = result.sample_paths.pct_change().iloc[1:].to_numpy()
    close_to_up = np.isclose(steps, 0.01, atol=1e-5)
    close_to_down = np.isclose(steps, -0.005, atol=1e-5)
    assert (close_to_up | close_to_down).all()
    share_up = close_to_up.mean()
    assert 0.47 < share_up < 0.53  # resampled uniformly with replacement


def test_constant_return_gives_deterministic_compounding():
    daily = 0.0004
    result = engine.run_monte_carlo(_synthetic_portfolio(np.full(100, daily), 1_000), 2, 200, starting_value=5_000)
    expected_final = 5_000 * (1 + daily) ** (2 * 252)
    np.testing.assert_allclose(result.terminal_values, expected_final, rtol=1e-4)  # float32 accumulation
    np.testing.assert_allclose(result.percentiles.iloc[-1], expected_final, rtol=1e-4)


def test_resampled_steps_come_from_the_portfolios_history(portfolio):
    result = engine.run_monte_carlo(portfolio, 1, 200, n_sample_paths=10)
    history = np.sort(portfolio["Daily Return"].dropna().to_numpy(dtype=np.float32))
    steps = (result.sample_paths.pct_change().iloc[1:].to_numpy()).ravel()
    nearest = history[np.clip(np.searchsorted(history, steps), 0, len(history) - 1)]
    nearest_below = history[np.clip(np.searchsorted(history, steps) - 1, 0, len(history) - 1)]
    distance = np.minimum(np.abs(nearest - steps), np.abs(nearest_below - steps))
    assert distance.max() < 1e-5


def test_percentiles_are_ordered_and_finite(portfolio):
    for horizon in (5, 30):
        result = engine.run_monte_carlo(portfolio, horizon, 2000)
        p = result.percentiles
        assert np.isfinite(p.to_numpy()).all() and np.isfinite(result.terminal_values).all()
        assert (p.to_numpy() > 0).all() and (result.terminal_values > 0).all()
        assert (p["p5"] <= p["p25"]).all() and (p["p25"] <= p["p50"]).all()
        assert (p["p50"] <= p["p75"]).all() and (p["p75"] <= p["p95"]).all()
        # Terminal percentiles agree with the stored fan's last row (float32 rounding only).
        np.testing.assert_allclose(
            p.iloc[-1].to_numpy(), np.percentile(result.terminal_values, [5, 25, 50, 75, 95]), rtol=1e-5
        )


def test_summary_statistics(portfolio):
    result = engine.run_monte_carlo(portfolio, 10, 2000)
    s = engine.summarize_monte_carlo(result)
    t = result.terminal_values
    assert s["Median Ending Value"] == pytest.approx(np.median(t))
    assert s["P5 Ending Value"] <= s["Median Ending Value"] <= s["P95 Ending Value"]
    assert s["Probability of Loss"] == pytest.approx((t < result.starting_value).mean())
    assert s["Probability of Doubling"] == pytest.approx((t >= 2 * result.starting_value).mean())
    assert s["Median CAGR"] == pytest.approx((np.median(t) / result.starting_value) ** (1 / 10) - 1)
    print(f"\n10Y/2000 sims from EUR{result.starting_value:,.0f}: median EUR{s['Median Ending Value']:,.0f}, "
          f"P5 EUR{s['P5 Ending Value']:,.0f}, P95 EUR{s['P95 Ending Value']:,.0f}, "
          f"P(loss) {s['Probability of Loss']:.1%}")
    # A bootstrap median CAGR should land near the historical CAGR (~9.5%).
    assert 0.06 < s["Median CAGR"] < 0.13


def test_insufficient_history_returns_none():
    assert engine.run_monte_carlo(_synthetic_portfolio(np.full(29, 0.001)), 5, 500) is None
    assert engine.run_monte_carlo(_synthetic_portfolio(np.full(30, 0.001)), 5, 500) is not None
    empty = engine.monte_carlo_forecast(_synthetic_portfolio(np.full(10, 0.001)))
    assert empty.empty and list(empty.columns) == ["p5", "p25", "p50", "p75", "p95"]
