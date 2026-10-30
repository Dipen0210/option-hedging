"""
Regression tests for candidate disclosure and portfolio-level budget enforcement.

Two defects are pinned here.

1. Moneyness was read in the wrong direction. Throughout the selector
   moneyness is strike / spot, so a protective put is out-of-the-money BELOW
   1.0. `_put_cons` guarded its OTM caveat on `moneyness > 1.05`, which no put
   the selector builds can satisfy, so the fact that an OTM put does not cover
   the first leg of a decline was disclosed on no recommendation at all — while
   the pros list beside it advertised unlimited downside protection. `_put_pros`
   had the mirror-image bug and was never called.

2. Nothing added the per-holding hedge costs up. Each holding is gated against
   a pro-rata share of the budget, but the portfolio total was neither enforced
   nor reported, so the response had no number answering "can I afford this?".

These are pure-function tests: no network, no fixtures, no pipeline.
"""
from __future__ import annotations

import pytest

from backend.engines.layer_10_evaluator import EvaluatorEngine
from backend.instruments.options.selector import OptionsSelector
from backend.models.hedge_models import (
    HedgeOutput,
    HedgeRecommendation,
    InstrumentCandidate,
)
from backend.models.risk_models import RegimeState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _regime(label: str = "mid_vol") -> RegimeState:
    return RegimeState(
        regime_label=label, regime_id=1, is_anomaly=False,
        anomaly_score=0.1, soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3, vix_level=16.91,
        vol_forecast_garch=0.1212, realized_vol_20d=0.0999,
    )


def _cand(cost: float, score: float, ticker: str = "SPY") -> InstrumentCandidate:
    return InstrumentCandidate(
        instrument_type="option",
        asset_class="equity",
        ticker=ticker,
        strategy=f"test {cost:.0f}",
        option_type="put",
        strike=100.0,
        n_contracts=1,
        total_cost=cost,
        score=score,
        efficiency_ratio=1.0,
    )


def _output(recs: dict[str, list[InstrumentCandidate]], notional: float) -> HedgeOutput:
    return HedgeOutput(
        portfolio_notional=notional,
        regime="mid_vol",
        is_anomaly=False,
        vix_level=17.5,
        run_time_seconds=0.0,
        recommendations=[
            HedgeRecommendation(rank=i, asset_ticker=t, candidates=c)
            for i, (t, c) in enumerate(recs.items(), start=1)
        ],
    )


# ── 1. Disclosure ─────────────────────────────────────────────────────────────

class TestMoneynessCaveat:
    def test_an_otm_put_discloses_the_uncovered_decline(self):
        # 3% OTM put on a $500 underlying: the holder absorbs the first
        # $15/share. That is the deductible and it has to be stated.
        note = OptionsSelector._moneyness_caveat(0.97, "put", 500.0)
        assert note is not None, (
            "an out-of-the-money protective put disclosed nothing about the "
            "decline it does not cover"
        )
        assert "not covered" in note
        assert "15.00" in note, f"the uncovered amount was not quantified: {note}"

    @pytest.mark.parametrize("moneyness", [0.99, 0.97, 0.90, 0.85])
    def test_every_otm_put_in_the_grid_gets_a_caveat(self, moneyness):
        # The old guard was `moneyness > 1.05`, so every one of these — the
        # whole put grid — produced no caveat.
        assert OptionsSelector._moneyness_caveat(moneyness, "put", 400.0) is not None

    def test_an_atm_put_has_nothing_to_disclose(self):
        assert OptionsSelector._moneyness_caveat(1.00, "put", 500.0) is None

    def test_an_itm_put_discloses_the_intrinsic_component(self):
        note = OptionsSelector._moneyness_caveat(1.05, "put", 500.0)
        assert note is not None
        assert "intrinsic" in note
        assert "25.00" in note

    def test_a_call_hedge_caveat_points_upward_not_downward(self):
        # The old text was reused verbatim for call hedges and told the user
        # the leg pays when the underlying "falls below strike" — backwards.
        note = OptionsSelector._moneyness_caveat(1.03, "call", 420.0)
        assert note is not None
        assert "rallies" in note
        assert "falls" not in note

    def test_direction_is_not_symmetric_between_puts_and_calls(self):
        # moneyness 1.03 is OTM for a call and ITM for a put. One helper,
        # opposite readings — which is exactly what the old code got wrong.
        call = OptionsSelector._moneyness_caveat(1.03, "call", 100.0)
        put = OptionsSelector._moneyness_caveat(1.03, "put", 100.0)
        assert "out-of-the-money" in call
        assert "in-the-money" in put


class TestOptionPros:
    def test_an_otm_put_is_not_called_atm(self):
        # `_put_pros` claimed "ATM delta — highest sensitivity to decline" for
        # every put, a 10%-OTM one included.
        pros = OptionsSelector._option_pros(0.90, "put", _regime())
        joined = " ".join(pros)
        assert "At the money" not in joined, (
            f"a 10% OTM put was described as at-the-money: {pros}"
        )
        assert "Out of the money" in joined

    def test_an_atm_put_is_called_atm(self):
        pros = OptionsSelector._option_pros(1.00, "put", _regime())
        assert any("At the money" in p for p in pros)

    def test_regime_pro_only_in_calm_regimes(self):
        calm = OptionsSelector._option_pros(1.00, "put", _regime("low_vol"))
        stressed = OptionsSelector._option_pros(1.00, "put", _regime("high_vol"))
        assert any("Low-vol regime" in p for p in calm)
        assert not any("Low-vol regime" in p for p in stressed)


class TestOptionCons:
    def test_the_uncovered_decline_leads_the_cons(self):
        cons = OptionsSelector._option_cons(0.95, "put", 500.0, 0.01, 0.05)
        assert "not covered" in cons[0], (
            f"the most material caveat was not stated first: {cons}"
        )

    def test_budget_pressure_is_quantified(self):
        cons = OptionsSelector._option_cons(1.00, "put", 500.0, 0.045, 0.05)
        assert any("90%" in c for c in cons), cons

    def test_theta_is_always_disclosed(self):
        cons = OptionsSelector._option_cons(1.00, "put", 500.0, 0.001, 0.05)
        assert any("Theta" in c for c in cons)

    def test_extra_caveats_are_appended_not_replaced(self):
        cons = OptionsSelector._option_cons(
            0.95, "put", 500.0, 0.001, 0.05, extra=["basis risk sentinel"]
        )
        assert "basis risk sentinel" in cons
        assert any("not covered" in c for c in cons)


# ── 2. Portfolio budget ───────────────────────────────────────────────────────

class TestPortfolioBudget:
    def setup_method(self):
        self.l10 = EvaluatorEngine()

    def test_the_portfolio_hedge_cost_is_reported(self):
        # Three holdings, one pick each. Before this check the response
        # carried no total at all.
        out = _output(
            {"AAPL": [_cand(1000, 80)],
             "MSFT": [_cand(800, 80)],
             "SPY": [_cand(600, 80)]},
            notional=200_000,
        )
        self.l10.enforce_portfolio_budget(out, 0.05)   # $10,000 budget
        assert out.hedge_portfolio_cost == pytest.approx(2400.0)
        assert out.hedge_budget_total == pytest.approx(10_000.0)
        assert out.hedge_budget_utilization == pytest.approx(0.24)

    def test_a_set_within_budget_is_left_alone(self):
        out = _output({"AAPL": [_cand(500, 90), _cand(400, 80)]}, notional=100_000)
        self.l10.enforce_portfolio_budget(out, 0.05)
        assert len(out.recommendations[0].candidates) == 2
        assert out.warnings == []

    def test_a_candidate_over_the_budget_is_removed(self):
        # $100k book, 2% cap → $2,000. The 5,000 pick breaches it and must not
        # be recommended; the 900 alternative stands in.
        out = _output({"AAPL": [_cand(5000, 90), _cand(900, 80)]}, notional=100_000)
        self.l10.enforce_portfolio_budget(out, 0.02)
        kept = out.recommendations[0].candidates
        assert [c.total_cost for c in kept] == [900]
        assert out.hedge_portfolio_cost == pytest.approx(900.0)
        assert out.hedge_budget_utilization <= 1.0
        assert any("hedge budget" in w for w in out.warnings)

    def test_a_holding_with_nothing_affordable_is_shown_unhedged(self):
        out = _output({"AAPL": [_cand(5000, 90)]}, notional=100_000)
        self.l10.enforce_portfolio_budget(out, 0.02)
        assert out.recommendations[0].candidates == []
        assert out.hedge_portfolio_cost == pytest.approx(0.0)
        assert any("unhedged" in w for w in out.warnings)

    def test_one_large_holding_cannot_starve_the_rest(self):
        # Walked most-expensive-first, the 1,900 pick is taken before the
        # cheaper ones, so what remains is what is genuinely left over.
        out = _output(
            {"SMALL": [_cand(50, 90)],
             "BIG": [_cand(1900, 90)],
             "MID": [_cand(400, 90)]},
            notional=100_000,
        )
        self.l10.enforce_portfolio_budget(out, 0.02)   # $2,000
        assert out.hedge_portfolio_cost <= out.hedge_budget_total
        by_ticker = {r.asset_ticker: r.candidates for r in out.recommendations}
        assert by_ticker["BIG"], "the largest holding was dropped"

    def test_the_total_never_exceeds_the_cap(self):
        out = _output(
            {f"H{i}": [_cand(900, 90), _cand(100, 10)] for i in range(5)},
            notional=100_000,
        )
        self.l10.enforce_portfolio_budget(out, 0.02)   # $2,000
        assert out.hedge_portfolio_cost <= out.hedge_budget_total + 0.01, (
            f"the recommended set costs ${out.hedge_portfolio_cost:,.0f} "
            f"against a ${out.hedge_budget_total:,.0f} budget"
        )

    def test_a_net_credit_does_not_consume_budget(self):
        # A financed collar pays the user. It cannot eat into the cap, and the
        # portfolio total has to carry the sign.
        out = _output({"AAPL": [_cand(-250, 90)]}, notional=100_000)
        self.l10.enforce_portfolio_budget(out, 0.02)
        assert out.recommendations[0].candidates, "a credit structure was dropped"
        assert out.hedge_portfolio_cost == pytest.approx(-250.0)

    def test_a_holding_with_no_candidates_is_skipped_cleanly(self):
        out = _output({"AAPL": [], "MSFT": [_cand(100, 90)]}, notional=100_000)
        self.l10.enforce_portfolio_budget(out, 0.05)
        assert out.hedge_portfolio_cost == pytest.approx(100.0)

    def test_zero_notional_does_not_divide_by_zero(self):
        out = _output({"AAPL": [_cand(100, 90)]}, notional=0.0)
        self.l10.enforce_portfolio_budget(out, 0.05)
        assert out.hedge_budget_utilization == 0.0
