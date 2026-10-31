"""
Regression tests for the figures L11 hands the language model, and for the
run where it should not be asked at all.

Two defects, found 2026-10-30 while verifying the stub-disclosure fix over
HTTP. A futures-only run — zero candidates — came back with

    top_recommendation:
    "No candidates are currently available to hedge the $30,000 AAPL
     position under the current VIX of 17.5."

against a response whose own `portfolio_notional` was **$40,000**.

The $30,000 was not a hallucination. L11 computed the holding exposure it put
in the prompt as `shares x purchase_price` — 200 x $150, the **cost basis** —
while L3 measures notional as `current_price x shares` and every other figure
in the response (portfolio_notional, VaR, hedge sizing, budget) is computed on
that. So the narrative described a $30,000 position sitting beside a $40,000
one, and on a profitable holding it understates the user's exposure by exactly
the unrealised gain. `portfolio_valuation.py` already calls that same
expression `cost_basis`; L11 was the only caller treating it as exposure. This
was wrong on *every* run, not only the empty ones — a successful options run
was narrated against cost basis too.

The empty run compounded it. With no candidate anywhere, the model was still
asked for "the single best hedge action to take now", and the prompt's own
first rule — every claim must cite a number — pushes it to supply a figure for
a recommendation that cannot exist.

The fixes: L11 passes the measured `notional_value`; the prompt carries null
rather than a substitute when there is no measurement; the system prompts
state that a null field is omitted and never estimated; and a run with no
candidate on any holding skips the provider and gets the deterministic
narrative with a notice saying so.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pytest

import backend.engines.layer_11_llm_explainer as l11
from backend.core.explainer.base_explainer import PortfolioExplanation
from backend.core.explainer.prompt_builder import (
    CANDIDATES_SYSTEM,
    COMBINED_SYSTEM,
    PORTFOLIO_SYSTEM,
    PromptBuilder,
)
from backend.models.hedge_models import (
    HedgeOutput,
    HedgeRecommendation,
    InstrumentCandidate,
)
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.risk_models import PortfolioRiskSummary, RegimeState, RiskProfile

SHARES = 200
PURCHASE_PRICE = 150.0
MARKET_PRICE = 200.0
COST_BASIS = SHARES * PURCHASE_PRICE        # 30_000 — what the prompt used to get
MARKET_NOTIONAL = SHARES * MARKET_PRICE     # 40_000 — what L3 measures


def _portfolio() -> PortfolioInput:
    return PortfolioInput(
        holdings=[StockPosition(
            ticker="AAPL",
            shares=SHARES,
            purchase_price=PURCHASE_PRICE,
            purchase_date=date.today() - timedelta(days=365),
        )],
        total_notional=MARKET_NOTIONAL,
    )


def _regime() -> RegimeState:
    return RegimeState(
        regime_label="mid_vol", regime_id=1, is_anomaly=False,
        anomaly_score=0.2, soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3, vix_level=17.5,
        vol_forecast_garch=0.18, realized_vol_20d=0.16,
    )


def _profile(notional: float = MARKET_NOTIONAL) -> RiskProfile:
    return RiskProfile(
        ticker="AAPL", asset_class="equity", notional_value=notional,
        beta_vs_spy=1.2, beta_available=True,
        var_5pct=notional * 0.08, cvar_5pct=notional * 0.11,
        var_pct=0.08, cvar_pct=0.11, notional_at_risk=notional * 0.08,
        optimal_hedge_ratio=1.0, regression_hedge_ratio=1.2,
        tail_correlation=0.6,
    )


def _summary(profile: RiskProfile) -> PortfolioRiskSummary:
    return PortfolioRiskSummary(
        total_notional=MARKET_NOTIONAL, portfolio_beta=1.2,
        portfolio_var_5pct=MARKET_NOTIONAL * 0.08,
        portfolio_cvar_5pct=MARKET_NOTIONAL * 0.11,
        concentration_top1=1.0, risk_profiles=[profile],
    )


def _candidate() -> InstrumentCandidate:
    return InstrumentCandidate(
        instrument_type="option", asset_class="equity", ticker="AAPL",
        strategy="protective_put", n_contracts=2, strike=190.0,
    )


def _output(candidates) -> HedgeOutput:
    return HedgeOutput(
        portfolio_notional=MARKET_NOTIONAL, regime="mid_vol", is_anomaly=False,
        vix_level=17.5, run_time_seconds=0.1,
        recommendations=[HedgeRecommendation(
            rank=1, asset_ticker="AAPL", candidates=candidates,
        )],
    )


class _Recorder:
    """
    Available, returns a complete narrative, and keeps the context it was
    given so the test can read the figures that would have reached the model.
    """
    provider_name = "recorder"

    def __init__(self):
        self.contexts = []

    def is_available(self) -> bool:
        return True

    def supports_combined_call(self) -> bool:
        return True

    def explain_all(self, ctx):
        self.contexts.append(ctx)
        return type("R", (), {
            "candidates": {},
            "portfolio": PortfolioExplanation(
                summary="Model summary.",
                key_risks=["Model risk."],
                regime_commentary="Model regime view.",
                top_recommendation="Model pick.",
            ),
        })()


def _run(monkeypatch, explainer, output, profile=None):
    monkeypatch.setattr(l11, "get_explainer", lambda provider=None: explainer)
    return l11.FinalLLMExplainerEngine().run(
        hedge_output=output,
        portfolio=_portfolio(),
        risk_summary=_summary(profile or _profile()),
        regime=_regime(),
    )


class TestTheExposureInThePromptIsTheOneTheProductComputed:

    def test_the_prompt_gets_the_measured_notional_not_the_cost_basis(self, monkeypatch):
        rec = _Recorder()
        _run(monkeypatch, rec, _output([_candidate()]))

        holding = rec.contexts[0]["positions"][0]["holding"]
        assert holding["notional_usd"] == MARKET_NOTIONAL
        # The defect, named: 200 x the $150 purchase price.
        assert holding["notional_usd"] != COST_BASIS

    def test_the_holding_figure_agrees_with_the_portfolio_figure(self, monkeypatch):
        """
        A single-holding portfolio is the clearest case: the position and the
        portfolio are the same money, so the two figures in one prompt must
        not disagree. They did — $30,000 and $40,000.
        """
        rec = _Recorder()
        _run(monkeypatch, rec, _output([_candidate()]))

        ctx = rec.contexts[0]
        assert (ctx["positions"][0]["holding"]["notional_usd"]
                == ctx["portfolio"]["total_notional_usd"])

    def test_an_unmeasured_notional_is_passed_as_null_not_substituted(self, monkeypatch):
        """
        When L3 measured no notional, the prompt says so. The tempting
        substitutes — cost basis, or the portfolio total — are each a figure
        the product did not compute for this holding.
        """
        rec = _Recorder()
        _run(monkeypatch, rec, _output([_candidate()]),
             profile=_profile(notional=0.0))

        holding = rec.contexts[0]["positions"][0]["holding"]
        assert holding["notional_usd"] is None
        assert holding["notional_usd"] != COST_BASIS

    def test_the_per_holding_prompt_builder_tolerates_a_missing_notional(self):
        """
        `round(None)` raises. The builder has to carry the absence, not crash
        — a crash here would be caught by L11's catch-all and reported to the
        user as an LLM failure.
        """
        ctx = PromptBuilder().build_candidates_context(
            asset_ticker="AAPL",
            holding_notional=None,
            risk_profile=_profile(),
            candidates=[_candidate()],
            regime=_regime(),
        )
        assert ctx["holding"]["notional_usd"] is None

    @pytest.mark.parametrize("prompt,name", [
        (CANDIDATES_SYSTEM, "candidates"),
        (COMBINED_SYSTEM, "combined"),
        (PORTFOLIO_SYSTEM, "portfolio"),
    ])
    def test_every_system_prompt_forbids_inventing_a_missing_figure(self, prompt, name):
        """
        A null is only safe if the model is told what it means. The combined
        prompt's first rule demands a number for every claim, which without
        this instruction reads as licence to produce one.
        """
        assert "null" in prompt, name
        lowered = prompt.lower()
        # Stems, so the three prompts can phrase it naturally ("never
        # estimate" / "rather than estimating one").
        assert "estimat" in lowered, name
        assert "never" in lowered, name


class TestAModelIsNotAskedToRecommendAHedgeThatDoesNotExist:

    def test_the_provider_is_not_called_when_no_holding_has_a_candidate(self, monkeypatch):
        rec = _Recorder()
        _run(monkeypatch, rec, _output([]))
        assert rec.contexts == [], "the model was asked to narrate an empty set"

    def test_the_run_is_still_narrated_and_the_skip_is_disclosed(self, monkeypatch):
        out = _run(monkeypatch, _Recorder(), _output([]))

        assert out.portfolio_summary.strip()
        assert out.top_recommendation.strip()
        assert out.llm_notice.strip()
        assert "not called" in out.llm_notice
        # And it is not passed off as the model's analysis.
        assert out.portfolio_summary != "Model summary."
        assert out.top_recommendation != "Model pick."

    def test_the_narrative_carries_no_figure_the_product_did_not_compute(self, monkeypatch):
        """
        The deterministic text restates computed values only. $30,000 is the
        figure that appeared in the defect; it must not appear from any path.
        """
        out = _run(monkeypatch, _Recorder(), _output([]))
        blob = " ".join([
            out.portfolio_summary, out.top_recommendation,
            out.regime_commentary, *out.key_risks,
        ])
        assert "30,000" not in blob and "30000" not in blob
        assert "$30.0K" not in blob

    def test_a_provider_is_still_called_when_one_holding_has_a_candidate(self, monkeypatch):
        """The skip must not swallow runs that do have something to explain."""
        rec = _Recorder()
        out = _run(monkeypatch, rec, _output([_candidate()]))
        assert len(rec.contexts) == 1
        assert out.portfolio_summary == "Model summary."
        assert out.llm_notice == ""


class TestAnEmptyResultIsNotExplainedByAPriceScreen:
    """
    The same ⑤ defect as the stub endpoints, in L11's deterministic text:
    budget, liquidity and basis-risk screens are one reason a candidate list
    can be empty, and they were named for every reason.
    """

    def test_the_no_hedge_sentence_names_no_financial_cause(self, monkeypatch):
        out = _run(monkeypatch, _Recorder(), _output([]))
        top = out.top_recommendation.lower()
        assert "no hedge is recommended" in top
        assert "budget" not in top
        assert "liquidity" not in top
        assert "basis-risk" not in top
        # It points at the reasons L4 actually recorded instead.
        assert "warnings" in top

    def test_the_summary_does_not_call_an_unevaluated_hedge_unaffordable(self, monkeypatch):
        out = _run(monkeypatch, _Recorder(), _output([]))
        assert "affordable" not in out.portfolio_summary
        assert "unhedged" in out.portfolio_summary
