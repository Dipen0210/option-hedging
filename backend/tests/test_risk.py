"""
Tests for Step 2 (regime detection) and Step 3 (per-stock risk analysis).
Unit tests use synthetic data to avoid network calls.
Integration smoke tests marked with @pytest.mark.integration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# company_data tests
# -----------------------------------------------------------------------

def test_company_data_lookup():
    from backend.data.company_data import get_company, get_sector, ticker_exists
    assert ticker_exists("AAPL")
    assert get_sector("AAPL") == "Technology"
    info = get_company("AAPL")
    assert info.name == "Apple Inc."
    assert info.market_cap > 0


def test_company_data_unknown_ticker():
    from backend.data.company_data import get_company, get_sector, ticker_exists
    assert not ticker_exists("FAKEXYZ")
    assert get_sector("FAKEXYZ") == "Unknown"
    assert get_company("FAKEXYZ") is None


def test_company_market_cap_category():
    from backend.data.company_data import get_company
    aapl = get_company("AAPL")
    assert aapl.market_cap_category == "mega_cap"


def test_company_sector_flags():
    from backend.data.company_data import get_company
    xom = get_company("XOM")
    assert xom is not None
    assert xom.is_commodity_exposed is True


# -----------------------------------------------------------------------
# VaR calculator tests (synthetic data — no network)
# -----------------------------------------------------------------------

def test_var_basic():
    from backend.core.risk.var_calculator import VaRCalculator
    np.random.seed(42)
    returns = pd.Series(np.random.normal(-0.001, 0.015, 500))
    calc = VaRCalculator()
    result = calc.historical_var(returns, notional=100_000, confidence=0.95, horizon_days=1)
    assert result["var_dollar"] > 0
    assert result["cvar_dollar"] >= result["var_dollar"]
    assert 0 < result["var_pct"] < 1


def test_var_scales_with_sqrt_of_TRADING_days_not_calendar_days():
    """
    Regression: horizon_days is CALENDAR days but sigma is a TRADING-day
    quantity. Scaling by sqrt(calendar days) overstated 180-day VaR by ~20%.
    """
    from backend.core.risk.var_calculator import VaRCalculator, _trading_days
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    calc = VaRCalculator()

    r1   = calc.parametric_cf_var(returns, 100_000, horizon_days=1)
    r180 = calc.parametric_cf_var(returns, 100_000, horizon_days=180)

    # Both horizons share the same sigma and z_CF, so the ratio is exactly
    # the ratio of the sqrt horizon scales.
    expected = np.sqrt(_trading_days(180) / _trading_days(1))
    ratio = r180["var_dollar"] / r1["var_dollar"]
    assert ratio == pytest.approx(expected, rel=1e-3)

    # And it must NOT be the calendar-day figure.
    assert ratio < np.sqrt(180) * 0.9

    # The conversion itself, and that it is reported to the caller.
    assert r180["horizon_trading_days"] == pytest.approx(124.19, abs=0.05)
    assert r1["horizon_trading_days"] == 1.0   # floored, never below one day


def test_var_reports_its_own_method_and_sigma_source():
    """A risk number must say how it was produced."""
    from backend.core.risk.var_calculator import VaRCalculator
    np.random.seed(3)
    returns = pd.Series(np.random.normal(0, 0.011, 400))
    calc = VaRCalculator()

    sample = calc.parametric_cf_var(returns, 100_000, horizon_days=90)
    assert sample["method"] == "parametric_cornish_fisher"
    assert sample["sigma_source"] == "sample_std"
    assert sample["n_observations"] == 400

    garch = calc.parametric_cf_var(returns, 100_000, horizon_days=90, garch_vol=0.22)
    assert garch["sigma_source"] == "garch_forecast"
    # A higher forward-looking vol than the sample must raise VaR.
    assert garch["var_dollar"] > sample["var_dollar"]


def test_implausible_garch_vol_falls_back_to_sample_std_and_discloses():
    from backend.core.risk.var_calculator import VaRCalculator
    from backend.data import data_quality as dq
    np.random.seed(5)
    returns = pd.Series(np.random.normal(0, 0.01, 300))
    calc = VaRCalculator()

    with dq.capture() as log:
        res = calc.parametric_cf_var(returns, 100_000, horizon_days=90,
                                     garch_vol=42.0, label="SPY")
    assert res["sigma_source"] == "sample_std"
    assert log.summary()["is_degraded"] is True
    assert any("GARCH" in w for w in log.summary()["warnings"])


def test_short_history_var_is_disclosed_as_an_assumption():
    """
    Regression: <30 observations silently returned a VaR built on a hardcoded
    20% vol, indistinguishable from a measured estimate.
    """
    from backend.core.risk.var_calculator import VaRCalculator
    from backend.data import data_quality as dq
    np.random.seed(11)
    returns = pd.Series(np.random.normal(0, 0.01, 12))
    calc = VaRCalculator()

    with dq.capture() as log:
        res = calc.parametric_cf_var(returns, 100_000, horizon_days=180, label="IPO")

    assert res["method"] == "assumed_vol_fallback"
    assert "assumed" in res["sigma_source"]
    assert res["n_observations"] == 12
    assert log.summary()["is_degraded"] is True
    # Warnings are grouped by condition, so the field and the subject are
    # named separately ("var: using fallback (IPO) — ...").
    assert any("var" in w and "IPO" in w for w in log.summary()["warnings"])
    # Still uses the trading-day horizon, like the main path.
    assert res["horizon_trading_days"] == pytest.approx(124.19, abs=0.05)


def test_cvar_is_never_below_var_and_names_its_source():
    from backend.core.risk.var_calculator import VaRCalculator
    np.random.seed(13)
    # Left-skewed, fat-tailed returns: the empirical tail path should engage.
    returns = pd.Series(np.random.standard_t(4, 600) * 0.008)
    calc = VaRCalculator()
    res = calc.parametric_cf_var(returns, 500_000, horizon_days=60)
    assert res["cvar_dollar"] >= res["var_dollar"]
    assert res["cvar_source"].startswith("empirical_tail_")


def test_zero_volatility_series_raises_rather_than_returning_zero_var():
    """A flat series cannot yield a defensible VaR — it must not report $0 risk."""
    from backend.core.risk.var_calculator import VaRCalculator
    calc = VaRCalculator()
    returns = pd.Series([0.0] * 100)
    with pytest.raises(ValueError, match="volatility"):
        calc.parametric_cf_var(returns, 100_000, horizon_days=180, label="FLAT")


def test_historical_var_alias_still_works():
    """Deprecated alias retained for existing callers."""
    from backend.core.risk.var_calculator import VaRCalculator
    np.random.seed(17)
    returns = pd.Series(np.random.normal(0, 0.01, 200))
    calc = VaRCalculator()
    assert (calc.historical_var(returns, 100_000, horizon_days=30)
            == calc.parametric_cf_var(returns, 100_000, horizon_days=30))


def test_tail_correlation_higher_in_crash():
    from backend.core.risk.var_calculator import VaRCalculator
    np.random.seed(7)
    # Simulate: normally low corr, but both crash together in tails
    n = 500
    ra = pd.Series(np.random.normal(0, 0.01, n))
    rb = pd.Series(np.random.normal(0, 0.01, n))
    # Inject correlated crashes in bottom 5%
    crash_idx = np.random.choice(n, 25)
    crash = np.random.uniform(-0.05, -0.03, 25)
    ra.iloc[crash_idx] = crash
    rb.iloc[crash_idx] = crash * 0.9

    calc = VaRCalculator()
    full_corr = float(np.corrcoef(ra.values, rb.values)[0, 1])
    tail_corr = calc.tail_correlation(ra, rb, percentile=0.05)
    assert tail_corr > full_corr


# -----------------------------------------------------------------------
# Hedge ratio tests (pure math — no network)
# -----------------------------------------------------------------------

def test_delta_adjusted_contracts():
    from backend.core.risk.hedge_ratio import HedgeRatioCalculator
    calc = HedgeRatioCalculator()
    # N = (β × V × target%) / (|Δ| × contract_size × S)
    # N = (1.2 × 600000) / (0.25 × 100 × 261) = 720000 / 6525 ≈ 110.3 → 110
    n = calc.delta_adjusted_contracts(600_000, 1.2, 0.25, 100, 261.0)
    assert n == 110


def test_beta_weighted_portfolio():
    from backend.core.risk.hedge_ratio import HedgeRatioCalculator
    calc = HedgeRatioCalculator()
    # N* = βp × (V / contract_value) × target%
    # N* = 1.1 × (500000 / 220000) = 1.1 × 2.2727 ≈ 2.5 → rounds to 2 or 3
    n = calc.beta_weighted_portfolio(1.1, 500_000, 220_000)
    assert n in (2, 3)


def test_partial_hedge_series():
    from backend.core.risk.hedge_ratio import HedgeRatioCalculator
    calc = HedgeRatioCalculator()
    series = calc.partial_hedge_series(10, steps=[0.5, 1.0], premium_per_contract=500)
    assert len(series) == 2
    assert series[0]["n_contracts"] == 5
    assert series[1]["n_contracts"] == 10
    assert series[1]["estimated_cost"] == 5000.0


# -----------------------------------------------------------------------
# GARCH model test (synthetic data — no network)
# -----------------------------------------------------------------------

def test_garch_fits_and_forecasts():
    from backend.ml.volatility.garch_model import GARCHVolModel
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    model = GARCHVolModel()
    vol = model.fit_and_forecast(returns)
    assert 0.05 < vol < 1.0   # annualized vol should be 5%–100%


def test_garch_conditional_vol():
    from backend.ml.volatility.garch_model import GARCHVolModel
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.015, 300))
    model = GARCHVolModel()
    model.fit(returns)
    cond_vol = model.current_conditional_vol()
    assert cond_vol > 0
