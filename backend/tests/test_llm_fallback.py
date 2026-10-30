"""
Tests for Layer 11's deterministic narrative fallback.

Three paths through `FinalLLMExplainerEngine.run` used to leave the narrative
fields at their empty defaults:

  * an unavailable provider returned `hedge_output` untouched, with no notice
    at all;
  * a rate limit set `llm_notice` but wrote no text;
  * the catch-all `except Exception` logged and carried on.

In each case `portfolio_summary`, `key_risks`, `regime_commentary` and
`top_recommendation` came back empty, and the UI rendered nothing — so "the
model had nothing to say about your risk" and "the explanation service is
down" looked identical. These tests pin that the narrative is always
populated, that the notice always says which it was, and — the part that
matters for financial correctness — that the deterministic text only ever
restates figures the pipeline actually computed.

No network: the explainer is a stub, and `get_explainer` is substituted at the
point L11 imports it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backend.engines import layer_11_llm_explainer as l11
from backend.engines.layer_11_llm_explainer import (
    FinalLLMExplainerEngine,
    _deterministic_narrative,
    _fmt_money,
    _FAILED_NOTICE,
    _RATE_LIMIT_NOTICE,
    _UNAVAILABLE_NOTICE,
)
from backend.core.explainer.huggingface_explainer import HFRateLimitError

# The existing fixtures are the authority on how these models are built;
# duplicating them here is how they drift.
from backend.tests.test_explainer import (
    _make_candidate,
    _make_hedge_output,
    _make_portfolio,
    _make_portfolio_risk_summary,
    _make_regime,
)


# ── Stub explainers ───────────────────────────────────────────────────────────

class _Stub:
    """Minimal explainer stand-in. Only what L11 actually calls."""

    provider_name = "stub"

    def __init__(self, available=True, raises=None, blank=False):
        self._available = available
        self._raises = raises
        self._blank = blank

    def is_available(self):
        return self._available

    def supports_combined_call(self):
        return True

    def explain_all(self, ctx):
        if self._raises:
            raise self._raises
        from backend.core.explainer.base_explainer import PortfolioExplanation

        class _Result:
            candidates = {}
            portfolio = PortfolioExplanation(
                summary="" if self._blank else "LLM summary.",
                key_risks=[] if self._blank else ["LLM risk"],
                regime_commentary="" if self._blank else "LLM regime note.",
                top_recommendation="" if self._blank else "LLM top pick.",
            )

        return _Result()


@pytest.fixture
def run(monkeypatch):
    """Run L11 with a stub provider and hand back the HedgeOutput."""
    def _run(stub, output=None, risk=None, regime=None):
        monkeypatch.setattr(l11, "get_explainer", lambda provider=None: stub)
        return FinalLLMExplainerEngine().run(
            hedge_output=output or _make_hedge_output(),
            portfolio=_make_portfolio(),
            risk_summary=risk or _make_portfolio_risk_summary(),
            regime=regime or _make_regime(),
        )
    return _run


def _narrative_fields(out):
    return [
        out.portfolio_summary,
        out.key_risks,
        out.regime_commentary,
        out.top_recommendation,
    ]


# ── The three silent-failure paths ────────────────────────────────────────────

class TestFallbackPaths:

    def test_unavailable_provider_still_produces_a_narrative(self, run):
        out = run(_Stub(available=False))
        assert all(_narrative_fields(out)), (
            "an unavailable provider left the analysis with no narrative at all"
        )
        assert out.llm_notice == _UNAVAILABLE_NOTICE

    def test_rate_limit_still_produces_a_narrative(self, run):
        out = run(_Stub(raises=HFRateLimitError("quota")))
        assert all(_narrative_fields(out))
        assert out.llm_notice == _RATE_LIMIT_NOTICE

    def test_unexpected_failure_still_produces_a_narrative_and_says_so(self, run):
        out = run(_Stub(raises=RuntimeError("upstream exploded")))
        assert all(_narrative_fields(out))
        assert out.llm_notice == _FAILED_NOTICE, (
            "a failed explainer produced no notice, so the blank narrative "
            "was indistinguishable from a model with nothing to say"
        )

    def test_a_blank_llm_response_is_backfilled(self, run):
        """A call that succeeds but returns empty strings is still a blank UI."""
        out = run(_Stub(blank=True))
        assert all(_narrative_fields(out))

    def test_the_notice_is_silent_when_the_llm_worked(self, run):
        out = run(_Stub())
        assert out.llm_notice == ""
        assert out.portfolio_summary == "LLM summary."

    def test_llm_text_is_never_overwritten(self, run):
        """The backstop must not clobber a real model response."""
        out = run(_Stub())
        assert out.portfolio_summary  == "LLM summary."
        assert out.key_risks          == ["LLM risk"]
        assert out.regime_commentary  == "LLM regime note."
        assert out.top_recommendation == "LLM top pick."


# ── What the deterministic text is allowed to say ─────────────────────────────

class TestNarrativeContent:

    def _fallback(self, output=None, risk=None, regime=None):
        out = output or _make_hedge_output()
        _deterministic_narrative(
            out, risk or _make_portfolio_risk_summary(), regime or _make_regime()
        )
        return out

    def test_summary_states_the_computed_var_not_a_rounded_invention(self):
        out = self._fallback()
        # 45,000 → "$45.0K"; the figure must be the one risk analysis produced.
        assert "$45.0K" in out.portfolio_summary
        assert "$60.0K" in out.portfolio_summary       # expected shortfall
        assert "1.30" in out.portfolio_summary         # portfolio beta

    def test_summary_reports_budget_usage_when_a_budget_was_enforced(self):
        out = _make_hedge_output()
        out.hedge_portfolio_cost      = 3500.0
        out.hedge_budget_total        = 10_000.0
        out.hedge_budget_utilization  = 0.35
        self._fallback(output=out)
        assert "35%" in out.portfolio_summary
        assert "$10.0K" in out.portfolio_summary

    def test_a_net_credit_is_described_as_a_credit_not_as_spend(self):
        out = _make_hedge_output()
        out.hedge_portfolio_cost     = -1200.0
        out.hedge_budget_total       = 10_000.0
        out.hedge_budget_utilization = -0.12
        self._fallback(output=out)
        assert "net credit" in out.portfolio_summary
        assert "-$" not in out.portfolio_summary

    def test_unhedged_holdings_are_counted_in_the_summary(self):
        out = _make_hedge_output()
        from backend.models.hedge_models import HedgeRecommendation
        out.recommendations.append(
            HedgeRecommendation(rank=2, asset_ticker="TLT", candidates=[])
        )
        self._fallback(output=out)
        assert "1 of 2 holdings have no affordable hedge" in out.portfolio_summary

    def test_a_model_priced_premium_is_disclosed_in_the_top_recommendation(self):
        out = _make_hedge_output()
        cand = out.recommendations[0].candidates[0]
        cand.premium_source = "model"
        self._fallback(output=out)
        assert "model estimate" in out.top_recommendation, (
            "the headline recommendation quoted a theoretical premium as "
            "though it were an executable price"
        )

    def test_a_market_priced_premium_carries_no_such_caveat(self):
        out = _make_hedge_output()
        out.recommendations[0].candidates[0].premium_source = "market_mid"
        self._fallback(output=out)
        assert "model estimate" not in out.top_recommendation

    def test_the_top_recommendation_identifies_the_actual_contract(self):
        out = _make_hedge_output()
        out.recommendations[0].candidates[0].contract_symbol = "SPY250620P00490000"
        self._fallback(output=out)
        assert "SPY250620P00490000" in out.top_recommendation

    def test_no_hedge_is_stated_plainly_rather_than_forced(self):
        """`Do not force a recommendation when no valid hedge exists.`"""
        out = _make_hedge_output()
        out.recommendations[0].candidates = []
        self._fallback(output=out)
        assert "No hedge is recommended" in out.top_recommendation

    def test_the_highest_scoring_candidate_is_the_one_recommended(self):
        out = _make_hedge_output()
        weak, strong = _make_candidate(strategy="Weak"), _make_candidate(strategy="Strong")
        weak.score, strong.score = 30.0, 91.0
        out.recommendations[0].candidates = [weak, strong]
        self._fallback(output=out)
        assert "Strong" in out.top_recommendation
        assert "91/100" in out.top_recommendation


class TestKeyRisks:

    def _risks(self, mutate=None, risk=None, regime=None):
        out = _make_hedge_output()
        if mutate:
            mutate(out)
        _deterministic_narrative(
            out, risk or _make_portfolio_risk_summary(), regime or _make_regime()
        )
        return " | ".join(out.key_risks)

    def test_weak_basis_risk_is_surfaced(self):
        def m(out):
            out.recommendations[0].candidates[0].basis_risk_r2 = 0.22
        assert "Basis risk" in self._risks(m)

    def test_strong_basis_risk_is_not_flagged(self):
        assert "Basis risk" not in self._risks()      # fixture R2 is 0.82

    def test_model_priced_premiums_are_counted(self):
        def m(out):
            out.recommendations[0].candidates[0].premium_source = "model"
        assert "model estimates" in self._risks(m)

    def test_a_failed_liquidity_screen_is_surfaced(self):
        def m(out):
            out.recommendations[0].candidates[0].liquidity_ok = False
        assert "liquidity screen" in self._risks(m)

    def test_unknown_liquidity_is_not_reported_as_a_failure(self):
        """None means the venue published nothing — not that it failed."""
        def m(out):
            out.recommendations[0].candidates[0].liquidity_ok = None
        assert "liquidity screen" not in self._risks(m)

    def test_concentration_is_surfaced_from_the_computed_figure(self):
        assert "Concentration" in self._risks()       # fixture top1 is 1.0

    def test_an_anomalous_regime_is_surfaced(self):
        regime = _make_regime()
        regime.is_anomaly = True
        assert "anomalous" in self._risks(regime=regime)


class TestRegimeCommentary:

    def test_a_degraded_regime_says_how_it_was_detected(self):
        out = _make_hedge_output()
        out.regime_degraded = True
        out.regime_source = "vix_fallback"
        out.regime_degradation_reason = "GMM received no usable history."
        _deterministic_narrative(out, _make_portfolio_risk_summary(), _make_regime())
        assert "vix_fallback" in out.regime_commentary
        assert "GMM received no usable history." in out.regime_commentary

    def test_an_undegraded_regime_reports_confidence_instead(self):
        out = _make_hedge_output()
        _deterministic_narrative(out, _make_portfolio_risk_summary(), _make_regime())
        assert "Confidence" in out.regime_commentary
        assert "fallback" not in out.regime_commentary

    def test_the_commentary_carries_the_computed_vols(self):
        out = _make_hedge_output()
        _deterministic_narrative(out, _make_portfolio_risk_summary(), _make_regime())
        assert "16.0%" in out.regime_commentary   # realized_vol_20d
        assert "18.0%" in out.regime_commentary   # garch forecast
        assert "18.5" in out.regime_commentary    # VIX


class TestMoneyFormatting:

    @pytest.mark.parametrize("value,expected", [
        (1_250_000, "$1.25M"),
        (34_500,    "$34.5K"),
        (412,       "$412"),
        (0,         "$0"),
        (-5_000,    "-$5.0K"),
    ])
    def test_scales(self, value, expected):
        assert _fmt_money(value) == expected
