"""
Regression tests for R10: the figures the pipeline substituted when a holding
had no market price.

Found 2026-10-31 while tracing R9. One holding without a `spot_price` entry
did not produce a warning, a decline, or a missing field. It produced a
confident hedge, sized and scored and simulated against numbers nobody
measured. Five call sites were re-deriving an exposure that L3 had already
computed as `current_price x shares`, and each had its own substitute:

  * the orchestrator's stock sizing context   -> shares x purchase_price
  * the orchestrator's `_holding_notionals`   -> shares x purchase_price
  * L7's missing-context fallback             -> the whole portfolio's notional
  * L9's simulation inputs                    -> spot $100, sigma 20%, and the
                                                 portfolio's notional (this one
                                                 fired on *every* run, because
                                                 nothing ever writes
                                                 `market_data[t]["notional"]`)
  * L10's scoring denominator                 -> the whole portfolio's notional

Cost basis is the price the user typed, which the product elsewhere calls
`cost_basis`; the error equals the unrealised gain, so it was worst on the
holdings the user had done best on. The portfolio notional is a different
quantity altogether, and substituting it scales a single holding's hedge by
the ratio of the portfolio to the position.

The approved fix, in the user's words: "decline to hedge it and say so", and
delete the invented 0.20 volatility default rather than keep it as a
last resort.

These tests assert on the figure each layer *receives* — the sizing context,
the scoring denominator, the simulation input. The 812-test suite passed
unchanged through every defect above, because it asserts that a
recommendation appears, and a recommendation built on a substituted number
appears exactly as readily as a correct one.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pytest

from backend.core.sizing.base_sizer import SizingContext, SizingResult
from backend.core.sizing.commodity_sizer import CommoditySizer
from backend.core.pricing.base_pricer import PriceResult
from backend.engines.layer_07_position_sizer import PositionSizerEngine
from backend.engines.layer_09_simulation_engine import SimulationEngine
from backend.engines.layer_10_evaluator import EvaluatorEngine
from backend.engines.pipeline_orchestrator import HedgeOrchestrator
from backend.models.hedge_models import (
    HedgeOutput,
    HedgeRecommendation,
    InstrumentCandidate,
    UnhedgedRisk,
)
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.risk_models import PortfolioRiskSummary, RegimeState, RiskProfile

SHARES = 100
PURCHASE_PRICE = 100.0
MARKET_PRICE = 400.0
COST_BASIS = SHARES * PURCHASE_PRICE        # 10_000 — the substitute
MARKET_NOTIONAL = SHARES * MARKET_PRICE     # 40_000 — what L3 measures
PORTFOLIO_NOTIONAL = 250_000.0              # the other substitute


def _portfolio(total: float = PORTFOLIO_NOTIONAL) -> PortfolioInput:
    return PortfolioInput(
        holdings=[StockPosition(
            ticker="MSFT",
            shares=SHARES,
            purchase_price=PURCHASE_PRICE,
            purchase_date=date.today() - timedelta(days=365),
        )],
        total_notional=total,
    )


def _profile(*, notional: float = MARKET_NOTIONAL, available: bool = True) -> RiskProfile:
    return RiskProfile(
        ticker="MSFT", asset_class="equity", notional_value=notional,
        beta_vs_spy=1.1, beta_available=True, data_available=available,
        var_5pct=notional * 0.08, cvar_5pct=notional * 0.11,
        var_pct=0.08, cvar_pct=0.11, notional_at_risk=notional * 0.08,
        optimal_hedge_ratio=1.0, regression_hedge_ratio=1.1,
        tail_correlation=0.6,
    )


def _summary(profile: RiskProfile) -> PortfolioRiskSummary:
    return PortfolioRiskSummary(
        total_notional=PORTFOLIO_NOTIONAL, portfolio_beta=1.1,
        portfolio_var_5pct=PORTFOLIO_NOTIONAL * 0.08,
        portfolio_cvar_5pct=PORTFOLIO_NOTIONAL * 0.11,
        concentration_top1=0.16, risk_profiles=[profile],
    )


def _regime() -> RegimeState:
    return RegimeState(
        regime_label="mid_vol", regime_id=1, is_anomaly=False,
        anomaly_score=0.2, soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3, vix_level=17.5,
        vol_forecast_garch=0.18, realized_vol_20d=0.16,
    )


def _candidate(**over) -> InstrumentCandidate:
    base = dict(
        instrument_type="option", asset_class="equity", ticker="MSFT",
        strategy="protective_put", n_contracts=2, strike=380.0,
        expiry_date="2027-01-15", option_type="put",
        premium=4.0, total_cost=800.0, max_protection=30_000.0,
    )
    base.update(over)
    return InstrumentCandidate(**base)


def _unhedged(notional: float = MARKET_NOTIONAL) -> UnhedgedRisk:
    return UnhedgedRisk(
        notional=notional, var_95=notional * 0.08, cvar_95=notional * 0.11,
        var_pct=0.08, beta=1.1,
        stress_loss_10pct=notional * 0.11, stress_loss_20pct=notional * 0.22,
    )


def _output(candidates=None, *, unhedged_notional: float = MARKET_NOTIONAL) -> HedgeOutput:
    return HedgeOutput(
        portfolio_notional=PORTFOLIO_NOTIONAL, regime="mid_vol", is_anomaly=False,
        vix_level=17.5, run_time_seconds=0.1,
        recommendations=[HedgeRecommendation(
            rank=1, asset_ticker="MSFT",
            candidates=[_candidate()] if candidates is None else candidates,
            unhedged_risk=_unhedged(unhedged_notional),
        )],
    )


def _market_data(**over) -> dict:
    md = {"MSFT": {"spot_price": MARKET_PRICE, "sigma": 0.28}, "_global": {}}
    md["MSFT"].update(over)
    return md


# ── The exposure handed to the sizer is the one L3 measured ─────────────────

class TestTheSizerReceivesTheMeasuredExposure:

    def test_the_sizing_context_carries_the_market_notional(self):
        ctxs, declined = HedgeOrchestrator()._build_sizing_contexts(
            _portfolio(), _market_data(), _summary(_profile()),
        )
        assert declined == {}
        assert ctxs["MSFT"].holding_notional == MARKET_NOTIONAL
        # The defect, named: 100 shares x the $100 purchase price.
        assert ctxs["MSFT"].holding_notional != COST_BASIS
        # And not the other substitute either.
        assert ctxs["MSFT"].holding_notional != PORTFOLIO_NOTIONAL

    def test_a_holding_with_no_price_is_declined_not_sized(self):
        """
        The substitution's whole danger is that it produced a context. An
        unmeasurable holding must produce none.
        """
        md = {"MSFT": {}, "_global": {}}
        ctxs, declined = HedgeOrchestrator()._build_sizing_contexts(
            _portfolio(), md, _summary(_profile()),
        )
        assert "MSFT" not in ctxs
        assert "MSFT" in declined
        assert "price" in declined["MSFT"]

    def test_a_holding_l3_could_not_measure_is_declined(self):
        """`data_available=False` is L3 saying it has no figure for this."""
        ctxs, declined = HedgeOrchestrator()._build_sizing_contexts(
            _portfolio(), _market_data(),
            _summary(_profile(notional=0.0, available=False)),
        )
        assert "MSFT" not in ctxs
        assert "MSFT" in declined

    def test_no_sizing_context_is_ever_built_on_the_portfolio_total(self):
        """
        A single holding is 16% of this portfolio. Whatever path is taken, the
        holding_notional must never arrive as the portfolio's own figure.
        """
        for md in (_market_data(), {"MSFT": {}, "_global": {}}):
            ctxs, _ = HedgeOrchestrator()._build_sizing_contexts(
                _portfolio(), md, _summary(_profile()),
            )
            for ctx in ctxs.values():
                assert ctx.holding_notional != PORTFOLIO_NOTIONAL


class TestTheDeclineIsVisibleToTheUser:

    def test_the_declined_holding_loses_its_candidates(self):
        out = _output()
        HedgeOrchestrator()._decline_unpriced(out, {"MSFT": "no price"})
        assert out.recommendations[0].candidates == []

    def test_the_decline_is_stated_in_a_warning_naming_the_holding(self):
        out = _output()
        HedgeOrchestrator()._decline_unpriced(out, {"MSFT": "no price"})
        blob = " ".join(out.warnings)
        assert "MSFT" in blob
        assert "unhedged" in blob
        # Silence was the original defect: a hedge appeared with nothing said.
        assert out.warnings

    def test_a_priced_holding_is_left_alone(self):
        out = _output()
        HedgeOrchestrator()._decline_unpriced(out, {})
        assert len(out.recommendations[0].candidates) == 1
        assert out.warnings == []


class TestTheNotionalMapOmitsWhatItCannotMeasure:

    def test_a_measured_holding_appears_at_its_market_notional(self):
        result = HedgeOrchestrator()._holding_notionals(
            _portfolio(), _market_data(), _summary(_profile()),
        )
        assert result["MSFT"] == MARKET_NOTIONAL

    def test_an_unmeasured_holding_is_absent_rather_than_cost_basis(self):
        result = HedgeOrchestrator()._holding_notionals(
            _portfolio(), {"MSFT": {}, "_global": {}},
            _summary(_profile(notional=0.0, available=False)),
        )
        assert "MSFT" not in result
        assert COST_BASIS not in result.values()


# ── L10: an unknown denominator leaves the score unset ──────────────────────

class TestScoringRefusesAPortfolioDenominator:

    def test_a_candidate_with_no_measured_notional_is_left_unscored(self):
        out = _output()
        before = out.recommendations[0].candidates[0].score
        EvaluatorEngine().run(
            hedge_output=out, regime=_regime(), holding_notionals={},
        )
        assert out.recommendations[0].candidates[0].score == before

    def test_a_candidate_with_a_measured_notional_is_scored(self):
        """The refusal must not swallow the runs that do have the figure."""
        out = _output()
        EvaluatorEngine().run(
            hedge_output=out, regime=_regime(),
            holding_notionals={"MSFT": MARKET_NOTIONAL},
        )
        assert out.recommendations[0].candidates[0].score > 0

    def test_the_score_differs_between_the_holding_and_the_portfolio(self):
        """
        Proof the denominator is load-bearing: if these two agreed, the old
        substitution would have been harmless and this test pointless.
        """
        scores = []
        for notional in (MARKET_NOTIONAL, PORTFOLIO_NOTIONAL):
            out = _output()
            EvaluatorEngine().run(
                hedge_output=out, regime=_regime(),
                holding_notionals={"MSFT": notional},
            )
            scores.append(out.recommendations[0].candidates[0].score)
        assert scores[0] != scores[1]


# ── L9: no invented spot, no invented volatility, no portfolio notional ────

class TestSimulationInputsAreMeasuredOrAbsent:

    def test_the_simulation_uses_the_holdings_notional_not_the_portfolios(self, monkeypatch):
        seen = []
        import backend.engines.layer_09_simulation_engine as l9
        real = l9._simulate_one

        def spy(cand, spot, sigma, notional, *a, **kw):
            seen.append({"spot": spot, "sigma": sigma, "notional": notional})
            return real(cand, spot, sigma, notional, *a, **kw)

        monkeypatch.setattr(l9, "_simulate_one", spy)
        SimulationEngine().run(
            hedge_output=_output(), market_data=_market_data(),
            holding_notionals={"MSFT": MARKET_NOTIONAL},
        )
        assert seen, "no candidate was simulated"
        assert seen[0]["notional"] == MARKET_NOTIONAL
        assert seen[0]["notional"] != PORTFOLIO_NOTIONAL
        assert seen[0]["spot"] == MARKET_PRICE
        assert seen[0]["sigma"] == 0.28

    @pytest.mark.parametrize("missing,substitute", [
        ("spot_price", 100.0),
        ("sigma", 0.20),
    ])
    def test_a_missing_input_is_not_replaced_by_its_old_default(
        self, monkeypatch, missing, substitute
    ):
        seen = []
        import backend.engines.layer_09_simulation_engine as l9
        monkeypatch.setattr(
            l9, "_simulate_one",
            lambda cand, spot, sigma, notional, *a, **kw: seen.append(
                {"spot": spot, "sigma": sigma}) or (cand, {}),
        )
        md = _market_data()
        del md["MSFT"][missing]
        SimulationEngine().run(
            hedge_output=_output(), market_data=md,
            holding_notionals={"MSFT": MARKET_NOTIONAL},
        )
        assert seen == [], f"simulated with a substituted {missing}={substitute}"

    def test_the_unsimulated_holding_is_disclosed(self):
        out = _output()
        md = _market_data()
        del md["MSFT"]["sigma"]
        SimulationEngine().run(
            hedge_output=out, market_data=md,
            holding_notionals={"MSFT": MARKET_NOTIONAL},
        )
        blob = " ".join(out.warnings)
        assert "MSFT" in blob
        assert "simulat" in blob.lower()

    def test_an_unmeasured_exposure_stops_the_simulation(self, monkeypatch):
        """
        `md["notional"]` is never written by any caller, so L9's old
        `md.get("notional", hedge_output.portfolio_notional)` fell through on
        every run. The replacement takes the orchestrator's measured map and
        declines when the holding is not in it.

        The first attempt at this fix read `rec.unhedged_risk.notional`, which
        is populated by L12 — three layers downstream — so it was absent on
        every real run and L9 silently simulated nothing. Caught over HTTP,
        not by this suite; hence the assertion below is on the *simulating*
        case too, in `test_the_simulation_uses_the_holdings_notional...`.
        """
        seen = []
        import backend.engines.layer_09_simulation_engine as l9
        monkeypatch.setattr(
            l9, "_simulate_one",
            lambda *a, **kw: seen.append(a) or (a[0], {}),
        )
        SimulationEngine().run(
            hedge_output=_output(), market_data=_market_data(),
            holding_notionals={},
        )
        assert seen == []


# ── L7: no context means no size, not a portfolio-sized one ────────────────

class TestSizingWithoutAContextIsRefused:

    def test_a_candidate_with_no_context_is_withdrawn(self):
        out = _output()
        PositionSizerEngine().run(hedge_output=out, sizing_contexts={})
        assert out.recommendations[0].candidates == []
        blob = " ".join(out.warnings)
        assert "MSFT" in blob

    def test_a_candidate_with_a_context_is_sized(self):
        out = _output()
        ctx = PositionSizerEngine.build_sizing_context(
            portfolio_notional=PORTFOLIO_NOTIONAL,
            holding_notional=MARKET_NOTIONAL,
            extended={"spot_price": MARKET_PRICE},
        )
        PositionSizerEngine().run(hedge_output=out, sizing_contexts={"MSFT": ctx})
        assert len(out.recommendations[0].candidates) == 1


# ── The invented volatilities are gone ─────────────────────────────────────

class TestNoVolatilityIsInvented:

    def test_the_sizing_context_has_no_default_volatility(self):
        ctx = SizingContext(
            portfolio_notional=PORTFOLIO_NOTIONAL, holding_notional=MARKET_NOTIONAL,
        )
        assert ctx.spot_vol is None
        assert ctx.futures_vol is None

    def test_the_context_builder_has_no_default_volatility(self):
        ctx = PositionSizerEngine.build_sizing_context(
            portfolio_notional=PORTFOLIO_NOTIONAL, holding_notional=MARKET_NOTIONAL,
        )
        assert ctx.spot_vol is None
        assert ctx.futures_vol is None

    def test_the_commodity_sizer_declines_without_a_measured_volatility(self):
        """
        h* = rho x sigma_s / sigma_f sets the contract count directly, so a
        substituted 20% produced a contract count that looked derived.
        """
        cand = _candidate(asset_class="commodity", ticker="USO", strike=70.0)
        result = CommoditySizer().size(
            cand, PriceResult(price=2.0, delta=-0.3, lambda_leverage=-5.0),
            SizingContext(
                portfolio_notional=PORTFOLIO_NOTIONAL,
                holding_notional=MARKET_NOTIONAL,
                extended={"futures_price": 70.0},
            ),
        )
        assert result.n_contracts == 0
        assert result.total_cost == 0.0

    def test_the_commodity_sizer_sizes_with_a_measured_volatility(self):
        cand = _candidate(asset_class="commodity", ticker="USO", strike=70.0)
        result = CommoditySizer().size(
            cand, PriceResult(price=2.0, delta=-0.3, lambda_leverage=-5.0),
            SizingContext(
                portfolio_notional=PORTFOLIO_NOTIONAL,
                holding_notional=MARKET_NOTIONAL,
                spot_vol=0.35, futures_vol=0.33,
                extended={"futures_price": 70.0},
            ),
        )
        assert result.n_contracts > 0
