"""
Tests for Layer 4 — Instrument Selection (options pricing, Greeks, position sizing).

All tests are pure-math unit tests using synthetic inputs.
No network calls, no yfinance, no market data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import numpy as np


# ── BSM pricer ────────────────────────────────────────────────────────────────

class TestBSMPricer:
    """Black-Scholes-Merton pricing and Greeks."""

    def test_put_price_positive(self):
        from backend.instruments.options.bsm_pricer import bsm_price
        p = bsm_price(S=500, K=500, r=0.05, q=0.01, sigma=0.20, T=0.25, option_type="put")
        assert p > 0

    def test_call_price_positive(self):
        from backend.instruments.options.bsm_pricer import bsm_price
        p = bsm_price(S=500, K=490, r=0.05, q=0.01, sigma=0.20, T=0.25, option_type="call")
        assert p > 0

    def test_put_call_parity(self):
        """C - P ≈ S·e^(-qT) - K·e^(-rT)"""
        from backend.instruments.options.bsm_pricer import bsm_price
        S, K, r, q, sigma, T = 500, 500, 0.05, 0.01, 0.20, 0.25
        call = bsm_price(S, K, r, q, sigma, T, "call")
        put  = bsm_price(S, K, r, q, sigma, T, "put")
        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 0.01

    def test_put_delta_negative(self):
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=500, K=500, r=0.05, q=0.01, sigma=0.20, T=0.25, option_type="put")
        assert -1 < g["delta"] < 0

    def test_call_delta_positive(self):
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=500, K=500, r=0.05, q=0.01, sigma=0.20, T=0.25, option_type="call")
        assert 0 < g["delta"] < 1

    def test_gamma_positive(self):
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=500, K=500, r=0.05, q=0.01, sigma=0.20, T=0.25)
        assert g["gamma"] > 0

    def test_theta_negative(self):
        """Long options lose value over time."""
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=500, K=500, r=0.05, q=0.01, sigma=0.20, T=0.25)
        assert g["theta"] < 0

    def test_vega_positive(self):
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=500, K=500, r=0.05, q=0.01, sigma=0.20, T=0.25)
        assert g["vega"] > 0

    def test_atm_put_delta_near_minus_half(self):
        """ATM put delta ≈ -0.5 (with no carry)."""
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=100, K=100, r=0.00, q=0.00, sigma=0.20, T=1.0, option_type="put")
        assert abs(g["delta"] + 0.5) < 0.05

    def test_deep_itm_put_delta_near_minus_one(self):
        from backend.instruments.options.bsm_pricer import bsm_greeks
        g = bsm_greeks(S=80, K=100, r=0.03, q=0.00, sigma=0.20, T=1.0, option_type="put")
        assert g["delta"] < -0.80

    def test_implied_vol_roundtrip(self):
        """Solve IV from a BSM price and confirm it recovers input sigma."""
        from backend.instruments.options.bsm_pricer import bsm_price, implied_vol
        S, K, r, q, T = 500, 490, 0.05, 0.01, 0.25
        true_sigma = 0.22
        mkt_price = bsm_price(S, K, r, q, true_sigma, T, "put")
        recovered = implied_vol(mkt_price, S, K, r, q, T, "put")
        assert recovered is not None
        assert abs(recovered - true_sigma) < 1e-4

    def test_put_call_parity_check_passes(self):
        from backend.instruments.options.bsm_pricer import bsm_price, put_call_parity_check
        S, K, r, q, sigma, T = 500, 500, 0.05, 0.0, 0.20, 0.25
        call = bsm_price(S, K, r, q, sigma, T, "call")
        put  = bsm_price(S, K, r, q, sigma, T, "put")
        assert put_call_parity_check(call, put, S, K, r, T)

    def test_zero_time_to_expiry_returns_intrinsic(self):
        from backend.instruments.options.bsm_pricer import bsm_price
        # ITM put: intrinsic = K - S = 500 - 480 = 20
        p = bsm_price(S=480, K=500, r=0.05, q=0.0, sigma=0.20, T=0.0, option_type="put")
        assert abs(p - 20.0) < 0.01


# ── Binomial CRR pricer ───────────────────────────────────────────────────────

class TestBinomialPricer:
    """CRR Binomial Tree."""

    def test_european_converges_to_bsm(self):
        """European binomial price should match BSM within $0.05."""
        from backend.instruments.options.binomial_pricer import crr_price
        from backend.instruments.options.bsm_pricer import bsm_price
        S, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.20, 0.5
        b_price = bsm_price(S, K, r, q, sigma, T, "put")
        c_price = crr_price(S, K, r, q, sigma, T, "put", american=False, steps=500)
        assert abs(c_price - b_price) < 0.05

    def test_american_put_gte_european(self):
        """American put ≥ European put (early exercise premium ≥ 0)."""
        from backend.instruments.options.binomial_pricer import crr_price
        S, K, r, q, sigma, T = 100, 105, 0.05, 0.03, 0.25, 0.5
        american = crr_price(S, K, r, q, sigma, T, "put", american=True)
        european = crr_price(S, K, r, q, sigma, T, "put", american=False)
        assert american >= european - 1e-6

    def test_early_exercise_premium_positive_for_itm(self):
        """Deep ITM put with dividends should have positive EEP."""
        from backend.instruments.options.binomial_pricer import early_exercise_premium
        eep = early_exercise_premium(S=80, K=100, r=0.05, q=0.04, sigma=0.25, T=0.5)
        assert eep >= 0.0

    def test_crr_greeks_delta_negative_for_put(self):
        from backend.instruments.options.binomial_pricer import crr_greeks
        g = crr_greeks(S=100, K=100, r=0.05, q=0.0, sigma=0.20, T=0.5, option_type="put")
        assert g["delta"] < 0

    def test_crr_greeks_gamma_positive(self):
        from backend.instruments.options.binomial_pricer import crr_greeks
        g = crr_greeks(S=100, K=100, r=0.05, q=0.0, sigma=0.20, T=0.5, option_type="put")
        assert g["gamma"] > 0


# ── Black-76 pricer ───────────────────────────────────────────────────────────

class TestBlack76Pricer:
    """Black-76 model for options on futures."""

    def test_futures_put_price_positive(self):
        from backend.instruments.options.black76_pricer import black76_price
        p = black76_price(F=80, K=80, r=0.05, sigma=0.25, T=0.5, option_type="put")
        assert p > 0

    def test_futures_call_price_positive(self):
        from backend.instruments.options.black76_pricer import black76_price
        p = black76_price(F=80, K=78, r=0.05, sigma=0.25, T=0.5, option_type="call")
        assert p > 0

    def test_at_the_money_forward_parity(self):
        """At-the-money forward: put price ≈ call price (with same K=F)."""
        from backend.instruments.options.black76_pricer import black76_price
        F, K, r, sigma, T = 100, 100, 0.05, 0.20, 0.5
        put  = black76_price(F, K, r, sigma, T, "put")
        call = black76_price(F, K, r, sigma, T, "call")
        # C - P = e^(-rT) * (F - K) = 0 when F=K
        assert abs(call - put) < 0.01

    def test_black76_greeks_delta_negative_for_put(self):
        from backend.instruments.options.black76_pricer import black76_greeks
        g = black76_greeks(F=80, K=80, r=0.05, sigma=0.25, T=0.5, option_type="put")
        assert g["delta"] < 0

    def test_implied_vol_76_roundtrip(self):
        from backend.instruments.options.black76_pricer import black76_price, implied_vol_76
        F, K, r, T = 100, 98, 0.05, 0.5
        true_sig = 0.18
        mkt_price = black76_price(F, K, r, true_sig, T, "put")
        recovered = implied_vol_76(mkt_price, F, K, r, T, "put")
        assert recovered is not None
        assert abs(recovered - true_sig) < 1e-4


# ── Multi-leg Greeks ──────────────────────────────────────────────────────────

class TestGreeks:
    """Multi-leg net Greeks aggregation."""

    def test_net_greeks_two_legs(self):
        """Long put + short call: delta should be sum."""
        from backend.instruments.options.bsm_pricer import bsm_greeks
        from backend.instruments.options.greeks import net_greeks
        g_put  = bsm_greeks(500, 495, 0.05, 0.01, 0.20, 0.25, "put")
        g_call = bsm_greeks(500, 510, 0.05, 0.01, 0.20, 0.25, "call")
        net = net_greeks([(g_put, +1), (g_call, -1)])
        expected_delta = g_put["delta"] - g_call["delta"]
        assert abs(net["delta"] - expected_delta) < 1e-6

    def test_collar_reduces_net_cost(self):
        """Collar net premium < naked put premium."""
        from backend.instruments.options.bsm_pricer import bsm_greeks
        from backend.instruments.options.greeks import collar_greeks
        g_put  = bsm_greeks(500, 495, 0.05, 0.01, 0.20, 0.25, "put")
        g_call = bsm_greeks(500, 515, 0.05, 0.01, 0.20, 0.25, "call")
        net = collar_greeks(g_put, g_call)
        assert net["price"] < g_put["price"]

    def test_spread_net_delta_less_than_single_put(self):
        """Bear put spread delta magnitude < single ATM put."""
        from backend.instruments.options.bsm_pricer import bsm_greeks
        from backend.instruments.options.greeks import bear_put_spread_greeks
        g_long  = bsm_greeks(500, 500, 0.05, 0.0, 0.20, 0.25, "put")   # ATM
        g_short = bsm_greeks(500, 460, 0.05, 0.0, 0.20, 0.25, "put")   # 8% OTM
        net = bear_put_spread_greeks(g_long, g_short)
        assert abs(net["delta"]) < abs(g_long["delta"])

    def test_hedge_effectiveness(self):
        from backend.instruments.options.greeks import hedge_effectiveness
        result = hedge_effectiveness(1000, -400, 500_000, 5_000)
        assert result["delta_offset_pct"] == 40.0
        assert result["notional_ratio"] == 1.0


# ── Position sizer ────────────────────────────────────────────────────────────

class TestPositionSizer:
    """Position sizing: delta-adjusted, budget gate, partial series."""

    def test_size_by_delta_basic(self):
        """N = (β × V × target%) / (|Δ| × contract_size × S)"""
        from backend.instruments.options.position_sizer import size_by_delta
        # N = (1.2 × 500000 × 1.0) / (0.50 × 100 × 500) = 600000 / 25000 = 24
        n = size_by_delta(500_000, 1.2, -0.50, 500.0)
        assert n == 24

    def test_size_by_delta_floors_at_1(self):
        """Even for large notional with very small delta, should return ≥ 1."""
        from backend.instruments.options.position_sizer import size_by_delta
        n = size_by_delta(1_000, 0.1, -0.01, 500.0)
        assert n >= 1

    def test_budget_gate_caps_contracts(self):
        """Budget gate must reduce contracts to stay within max_cost_pct."""
        from backend.instruments.options.position_sizer import apply_budget_gate
        # 10 contracts × $5/share × 100 = $5,000 cost
        # budget = $1,000,000 × 0.002 = $2,000 → should cap at 4 contracts
        result = apply_budget_gate(10, 5.0, 1_000_000, 0.002)
        assert result["n_contracts_gated"] == 4
        assert result["capped"] is True
        assert result["total_cost"] == 2000.0

    def test_budget_gate_no_cap_needed(self):
        """If cost is below budget, no capping occurs."""
        from backend.instruments.options.position_sizer import apply_budget_gate
        result = apply_budget_gate(5, 2.0, 1_000_000, 0.02)
        assert result["n_contracts_gated"] == 5
        assert result["capped"] is False

    def test_partial_series_has_correct_length(self):
        from backend.instruments.options.position_sizer import partial_hedge_series
        series = partial_hedge_series(20, option_price=3.0, steps=[0.5, 0.75, 1.0])
        assert len(series) == 3

    def test_partial_series_100pct_is_full_contracts(self):
        from backend.instruments.options.position_sizer import partial_hedge_series
        series = partial_hedge_series(10, option_price=5.0, steps=[0.5, 1.0])
        assert series[-1]["n_contracts"] == 10
        assert series[-1]["estimated_cost"] == 5000.0

    def test_compute_position_full_pipeline(self):
        """End-to-end: delta size → budget gate → partial series → viable."""
        from backend.instruments.options.position_sizer import compute_position
        result = compute_position(
            portfolio_notional=500_000,
            asset_beta=1.2,
            option_delta=-0.50,
            option_price=3.50,
            spot_price=500.0,
            max_cost_pct=0.02,
            hedge_target_pct=1.0,
        )
        assert result["viable"]
        assert result["full_contracts"] > 0
        assert result["gated"]["total_cost"] <= 500_000 * 0.02 + 1   # within budget
        assert len(result["partial_series"]) > 0

    def test_compute_position_not_viable_when_too_expensive(self):
        """If even 1 contract exceeds budget, viable = False."""
        from backend.instruments.options.position_sizer import compute_position
        result = compute_position(
            portfolio_notional=1_000,    # tiny portfolio
            asset_beta=1.0,
            option_delta=-0.50,
            option_price=50.0,           # very expensive
            spot_price=500.0,
            max_cost_pct=0.001,          # 0.1% = $1 budget
        )
        assert not result["viable"]

    def test_rounding_error_zero_for_exact_match(self):
        from backend.instruments.options.position_sizer import rounding_error_pct
        assert rounding_error_pct(10.0, 10) == 0.0

    def test_rounding_error_nonzero(self):
        from backend.instruments.options.position_sizer import rounding_error_pct
        err = rounding_error_pct(10.5, 10)
        assert abs(err - 4.76) < 0.1


# ── Model consistency check ───────────────────────────────────────────────────

def test_instrument_candidate_model_fields():
    """InstrumentCandidate model accepts all expected fields."""
    from backend.models.hedge_models import InstrumentCandidate
    c = InstrumentCandidate(
        instrument_type="option",
        asset_class="equity",
        ticker="SPY",
        strategy="Protective Put",
        strike=490.0,
        expiry_date="2025-06-20",
        option_type="put",
        n_contracts=10,
        total_cost=3500.0,
        delta=-0.45,
        gamma=0.002,
        theta=-0.15,
        vega=0.25,
        lambda_leverage=-12.5,
        basis_risk_r2=0.82,
        efficiency_ratio=12.5,
        score=78.5,
        rationale="Test candidate",
        pros=["Pro 1"],
        cons=["Con 1"],
        partial_hedge_options=[{"hedge_pct": 0.5, "n_contracts": 5, "estimated_cost": 1750.0}],
    )
    assert c.n_contracts == 10
    assert c.score == 78.5


def test_hedge_output_model():
    """HedgeOutput + HedgeRecommendation round-trip."""
    from backend.models.hedge_models import HedgeOutput, HedgeRecommendation, InstrumentCandidate
    output = HedgeOutput(
        portfolio_notional=500_000,
        regime="mid_vol",
        is_anomaly=False,
        vix_level=18.5,
        recommendations=[
            HedgeRecommendation(
                rank=1,
                asset_ticker="AAPL",
                candidates=[],
            )
        ],
        run_time_seconds=0.42,
    )
    assert output.regime == "mid_vol"
    assert len(output.recommendations) == 1
