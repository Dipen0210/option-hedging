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


def test_var_scales_with_horizon():
    from backend.core.risk.var_calculator import VaRCalculator
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    calc = VaRCalculator()
    r1  = calc.historical_var(returns, 100_000, horizon_days=1)
    r20 = calc.historical_var(returns, 100_000, horizon_days=20)
    # VaR_20 should be ~sqrt(20) times VaR_1
    ratio = r20["var_dollar"] / r1["var_dollar"]
    assert abs(ratio - np.sqrt(20)) < 1.5  # allow some tolerance


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
