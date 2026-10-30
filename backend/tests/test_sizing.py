"""
Regression tests for position sizing (L4 selector / L7 sizer).

Every test here pins a defect that reached the user. L4 sized candidates
against the live chain, then L6 overwrote the cost with a theoretical price
and L7 re-sized with a second, divergent copy of the same formula. Because L7
ran last, its numbers were the ones displayed — so the sizing the product
showed was computed from a price that was not for sale, for a coverage level
the user had not asked for, against the wrong underlying's spot.

These are pure arithmetic: no network, no fixtures, no pipeline.
"""

import pytest

from backend.core.pricing.base_pricer import PriceResult
from backend.core.sizing.equity_sizer import EquitySizer
from backend.core.sizing.base_sizer import SizingContext
from backend.engines.layer_06_pricing_engine import _apply_price_result
from backend.engines.layer_07_position_sizer import (
    _apply_sizing_result,
    executable_premium,
)
from backend.instruments.options.position_sizer import (
    MIN_HEDGE_DELTA,
    MIN_PREMIUM_PER_SHARE,
    compute_position,
)
from backend.models.hedge_models import InstrumentCandidate


def candidate(**over) -> InstrumentCandidate:
    base = dict(
        instrument_type="option",
        asset_class="equity",
        ticker="SPY",
        strategy="Protective Put",
        strike=480.0,
        expiry_date="2026-12-18",
        option_type="put",
        n_contracts=0,
    )
    base.update(over)
    return InstrumentCandidate(**base)


def context(**over) -> SizingContext:
    base = dict(
        portfolio_notional=500_000.0,
        holding_notional=100_000.0,
        beta=1.0,
        hedge_ratio=0.15,          # the user's protection level
        max_premium_pct=0.05,
        extended={"spot_price": 500.0},
    )
    base.update(over)
    return SizingContext(**base)


def price(premium: float, delta: float = -0.30) -> PriceResult:
    return PriceResult(price=premium, delta=delta, lambda_leverage=-5.0)


SIZER = EquitySizer()


# ── The hedge is sized against the right instrument ──────────────────────────

class TestHedgeInstrumentSpot:
    def test_cross_hedge_uses_the_hedge_etfs_spot_not_the_holdings(self):
        """
        A QQQ put bought against an AAPL position must be divided by QQQ's
        price. The sizing context is built per holding, so `spot_price` is
        AAPL's; using it put the wrong number in the denominator of every
        proxy and macro hedge.
        """
        cand = candidate(ticker="QQQ",
                         extended_metrics={"sizing_ratio": 1.0,
                                           "hedge_spot": 400.0})
        ctx = context(extended={"spot_price": 200.0})   # the holding's price
        result = SIZER.size(cand, price(5.0), ctx)

        # N = 1.0 x 100,000 x 0.15 / (0.30 x 100 x 400) = 1.25 -> 1
        assert result.n_contracts == 1
        # Against the holding's 200.0 it would have been 2.5 -> 2.
        wrong = SIZER.size(
            candidate(ticker="QQQ", extended_metrics={"sizing_ratio": 1.0}),
            price(5.0), ctx,
        )
        assert wrong.n_contracts == 2, (
            "the fallback no longer reproduces the old behaviour, so this "
            "test is not pinning anything"
        )

    def test_notional_hedged_is_measured_in_the_hedge_instrument(self):
        cand = candidate(ticker="QQQ",
                         extended_metrics={"sizing_ratio": 1.0,
                                           "hedge_spot": 400.0})
        r = SIZER.size(cand, price(5.0), context(extended={"spot_price": 200.0}))
        assert r.notional_hedged == pytest.approx(r.n_contracts * 0.30 * 100 * 400)


# ── The protection level the user chose is the one applied ───────────────────

class TestProtectionLevel:
    @pytest.mark.parametrize("level,expected", [(0.05, 10), (0.10, 20), (0.20, 40)])
    def test_contract_count_scales_with_protection_level(self, level, expected):
        # N = floor(1.0 x 100,000 x level / (0.10 x 100 x 50)) = 200 x level,
        # well under the 50-lot budget cap so only the level is being measured.
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 50.0})
        r = SIZER.size(cand, price(1.0, delta=-0.10), context(hedge_ratio=level))
        assert r.n_contracts == expected

    def test_a_full_hedge_is_not_assumed(self):
        """L7 hardcoded hedge_ratio=1.0, so 15% and 100% produced one answer."""
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 50.0})
        partial = SIZER.size(cand, price(1.0, delta=-0.10), context(hedge_ratio=0.15))
        full = SIZER.size(cand, price(1.0, delta=-0.10), context(hedge_ratio=1.00))
        assert partial.n_contracts < full.n_contracts


# ── The budget is a cap, not a suggestion ────────────────────────────────────

class TestBudgetGate:
    def test_an_unaffordable_hedge_returns_nothing_not_one_contract(self):
        """
        `min(n, max(max_contracts, 1))` always let one lot through. On a
        $100,000 holding with a 0.5% budget ($500), a contract costing $9,000
        was still quoted — eighteen times the cap the user set.
        """
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        r = SIZER.size(cand, price(90.0), context(max_premium_pct=0.005))
        assert r.n_contracts == 0
        assert r.total_cost == 0.0

    def test_cost_never_exceeds_the_cap(self):
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        ctx = context(max_premium_pct=0.02)     # $2,000 of a $100,000 holding
        r = SIZER.size(cand, price(7.0), ctx)
        assert r.total_cost <= ctx.holding_notional * ctx.max_premium_pct

    def test_rounding_is_down_so_the_cap_is_not_crossed(self):
        # Exact N = 1.0 x 100,000 x 0.15 / (0.25 x 100 x 400) = 1.5
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 400.0})
        r = SIZER.size(cand, price(2.0, delta=-0.25), context())
        assert r.n_contracts == 1, "ceil() over-hedged by rounding 1.5 up to 2"


# ── Guards against absurd inputs ─────────────────────────────────────────────

class TestUnpriceableInputs:
    def test_a_contract_with_no_delta_cannot_hedge(self):
        """
        The 1e-6 floor was a division-by-zero check, not a statement about
        what can hedge. A put priced off a junk 1.56% IV came back with
        |delta| ~ 1e-5 and the sizer asked for 1,500,000 contracts.
        """
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        r = SIZER.size(cand, price(1.0, delta=-1e-5), context())
        assert r.n_contracts == 0

    def test_the_delta_floor_is_the_shared_constant(self):
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        below = SIZER.size(cand, price(1.0, delta=-(MIN_HEDGE_DELTA / 2)), context())
        above = SIZER.size(cand, price(1.0, delta=-(MIN_HEDGE_DELTA * 2)), context())
        assert below.n_contracts == 0
        assert above.n_contracts > 0

    def test_a_sub_penny_premium_is_rejected_not_divided_into_the_budget(self):
        """
        A premium below a cent is a bad pricing input, and dividing a budget
        by it bypasses the budget entirely.
        """
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        r = SIZER.size(cand, price(MIN_PREMIUM_PER_SHARE / 2), context())
        assert r.n_contracts == 0

    def test_a_missing_spot_is_refused(self):
        r = SIZER.size(candidate(), price(5.0), context(extended={}))
        assert r.n_contracts == 0


# ── Credit structures ────────────────────────────────────────────────────────

class TestCreditStructures:
    def test_a_net_credit_is_carried_through_as_a_negative_cost(self):
        """
        A collar financed past zero pays the user. L6 clamped the net price at
        0.0 and the sizer then read a free trade: the inflow never reached the
        screen and the budget gate saw no cost at all.
        """
        cand = candidate(strategy="Collar", short_strike=560.0,
                         short_option_type="call",
                         extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        r = SIZER.size(cand, price(-1.50), context())
        assert r.n_contracts > 0
        assert r.total_cost < 0, "a credit was reported as a cost of zero or more"
        assert r.total_cost == pytest.approx(r.n_contracts * -1.50 * 100)

    def test_a_credit_is_not_capped_by_the_cost_budget(self):
        cand = candidate(strategy="Collar", short_strike=560.0,
                         extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        tight = SIZER.size(cand, price(-1.50), context(max_premium_pct=0.005))
        loose = SIZER.size(cand, price(-1.50), context(max_premium_pct=0.20))
        assert tight.n_contracts == loose.n_contracts > 0


# ── One formula, two layers ──────────────────────────────────────────────────

class TestLayersAgree:
    @pytest.mark.parametrize("delta,premium,level", [
        (-0.30,  5.00, 0.15),
        (-0.12,  1.25, 0.50),
        (-0.55, 22.00, 0.10),
    ])
    def test_l7_reproduces_the_selectors_count(self, delta, premium, level):
        """
        L4 sizes to decide what to keep; L7 sizes to produce the number shown.
        When they disagree the user sees a candidate selected on one basis and
        priced on another.
        """
        ctx = context(hedge_ratio=level)
        cand = candidate(extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        l7 = SIZER.size(cand, price(premium, delta), ctx)

        l4 = compute_position(
            portfolio_notional=ctx.holding_notional,
            asset_beta=1.0,
            option_delta=delta,
            option_price=premium,
            spot_price=500.0,
            max_cost_pct=ctx.max_premium_pct,
            hedge_target_pct=level,
        )
        assert l7.n_contracts == l4["gated"]["n_contracts_gated"]
        assert l7.total_cost == pytest.approx(l4["gated"]["total_cost"])


# ── Cost follows the price the user would pay ────────────────────────────────

class TestExecutablePremium:
    def test_a_quoted_candidate_is_costed_at_the_market_mid(self):
        c = candidate(premium_source="market_mid", market_premium=11.50,
                      model_premium=8.00)
        assert executable_premium(c) == 11.50

    def test_an_unquoted_candidate_falls_back_to_the_model(self):
        c = candidate(premium_source="model", market_premium=0.0,
                      model_premium=8.00)
        assert executable_premium(c) == 8.00

    def test_sizing_a_quoted_candidate_uses_the_mid(self):
        """
        The premium used to be recovered as `total_cost / 100`, which after L6
        was the theoretical price. A contract quoted at 11.50 was therefore
        sized and budgeted as if it cost 8.00.
        """
        cand = candidate(premium_source="market_mid", market_premium=11.50,
                         model_premium=8.00,
                         extended_metrics={"sizing_ratio": 1.0, "hedge_spot": 500.0})
        from backend.engines.layer_07_position_sizer import _build_price_result
        cand.delta = -0.30
        assert _build_price_result(cand).price == 11.50

    def test_market_total_cost_follows_the_contract_count(self):
        from backend.core.sizing.base_sizer import SizingResult
        c = candidate(premium_source="market_mid", market_premium=11.50,
                      market_total_cost=4_600.0, n_contracts=4)
        _apply_sizing_result(c, SizingResult(n_contracts=2, total_cost=2_300.0,
                                             notional_hedged=0.0))
        assert c.n_contracts == 2
        assert c.total_cost == 2_300.0
        assert c.market_total_cost == 2_300.0, (
            "the response quoted two different costs for the same trade"
        )


# ── L6 owns the model price, not the cost ────────────────────────────────────

class TestPricingLayerDoesNotSetCost:
    def test_the_model_price_goes_to_model_premium(self):
        c = candidate(n_contracts=7, total_cost=9_999.0)
        _apply_price_result(c, PriceResult(price=12.34, delta=-0.3))
        assert c.model_premium == 12.34

    def test_the_executable_cost_survives_pricing(self):
        """
        `total_cost = price * 100` discarded both the market quote and the
        contract count, leaving one contract's theoretical cost where the
        position's executable cost had been.
        """
        c = candidate(n_contracts=7, total_cost=9_999.0,
                      premium_source="market_mid", market_premium=14.00)
        _apply_price_result(c, PriceResult(price=12.34, delta=-0.3))
        assert c.total_cost == 9_999.0

    def test_the_quote_is_compared_against_the_price_actually_shown(self):
        c = candidate(premium_source="market_mid", market_premium=14.00)
        _apply_price_result(c, PriceResult(price=10.00, delta=-0.3))
        assert c.premium_vs_model_pct == pytest.approx(0.40)
