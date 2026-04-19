"""
Tests for Layer 5 — LLM Explainer.

All tests use a mock explainer — no real API calls.
Tests cover:
  - PromptBuilder output shape
  - CandidateExplanation / PortfolioExplanation dataclasses
  - BaseLLMExplainer contract (mock subclass)
  - LLMExplainerEngine merging logic
  - Graceful fallback when LLM unavailable or returns bad JSON
  - JSON extraction utility (_extract_json)
  - ClaudeExplainer / OllamaExplainer parse helpers (with synthetic JSON)
  - get_explainer factory routing
  - HedgeOutput model accepts new LLM fields
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from typing import Any, Dict, List

from backend.core.explainer.base_explainer import (
    BaseLLMExplainer,
    CandidateExplanation,
    PortfolioExplanation,
)


# ── Mock explainer used across all engine tests ────────────────────────────────

class MockExplainer(BaseLLMExplainer):
    """Deterministic stub — returns predictable data without any API call."""

    def __init__(self, available: bool = True, fail: bool = False):
        self._available = available
        self._fail = fail

    def is_available(self) -> bool:
        return self._available

    def explain_candidates(self, context: Dict[str, Any]) -> List[CandidateExplanation]:
        if self._fail:
            raise RuntimeError("mock failure")
        ticker = context.get("holding", {}).get("ticker", "MOCK")
        return [
            CandidateExplanation(
                ticker="SPY",
                strategy="Protective Put",
                asset_ticker=ticker,
                when_works_best="Works best in sudden market selloffs.",
                when_fails="Underperforms in slow grinding declines with high theta.",
                rationale="ATM put on SPY provides direct downside protection.",
                pros=["Simple", "Liquid"],
                cons=["Theta decay"],
            )
        ]

    def explain_portfolio(self, context: Dict[str, Any]) -> PortfolioExplanation:
        if self._fail:
            raise RuntimeError("mock failure")
        return PortfolioExplanation(
            summary="Portfolio carries elevated systematic risk.",
            key_risks=["High beta", "VIX spike risk"],
            regime_commentary="Mid-vol regime favours moderate hedging.",
            top_recommendation="Buy SPY puts to cap downside.",
        )


# ── Helpers to build synthetic Pydantic objects ──────────────────────────────

def _make_regime():
    from backend.models.risk_models import RegimeState
    return RegimeState(
        regime_label="mid_vol",
        regime_id=1,
        is_anomaly=False,
        anomaly_score=0.1,
        soft_membership=[0.1, 0.9, 0.0],
        n_discovered_regimes=3,
        vix_level=18.5,
        vol_forecast_garch=0.18,
        realized_vol_20d=0.16,
        sentiment_score=0.0,
    )


def _make_risk_profile(ticker="IWM"):
    from backend.models.risk_models import RiskProfile
    return RiskProfile(
        ticker=ticker,
        asset_class="equity",
        notional_value=500_000,
        beta_vs_spy=1.3,
        var_5pct=45_000,
        cvar_5pct=60_000,
        var_pct=0.09,
        cvar_pct=0.12,
        notional_at_risk=45_000,
        optimal_hedge_ratio=1.2,
        regression_hedge_ratio=1.1,
        tail_correlation=0.85,
    )


def _make_portfolio_risk_summary():
    from backend.models.risk_models import PortfolioRiskSummary
    return PortfolioRiskSummary(
        total_notional=500_000,
        portfolio_beta=1.3,
        portfolio_var_5pct=45_000,
        portfolio_cvar_5pct=60_000,
        concentration_top1=1.0,
        risk_profiles=[_make_risk_profile()],
        correlation_matrix=[[1.0]],
        tickers=["IWM"],
    )


def _make_portfolio():
    from backend.models.portfolio_models import PortfolioInput, HoldingInput
    return PortfolioInput(
        holdings=[HoldingInput(
            ticker="IWM", shares=2295, purchase_price=217.0,
            purchase_date="2024-01-01", asset_class="equity",
        )],
        total_notional=500_000,
        hedge_horizon_days=90,
        protection_level=0.15,
        max_hedge_cost_pct=0.02,
        upside_preservation=True,
        execution_mode="paper",
    )


def _make_candidate(ticker="SPY", strategy="Protective Put"):
    from backend.models.hedge_models import InstrumentCandidate
    return InstrumentCandidate(
        instrument_type="option",
        asset_class="equity",
        ticker=ticker,
        strategy=strategy,
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
        score=72.0,
        rationale="Rule-based rationale.",
        pros=["Pro 1"],
        cons=["Con 1"],
    )


def _make_hedge_output():
    from backend.models.hedge_models import HedgeOutput, HedgeRecommendation
    return HedgeOutput(
        portfolio_notional=500_000,
        regime="mid_vol",
        is_anomaly=False,
        vix_level=18.5,
        recommendations=[
            HedgeRecommendation(
                rank=1,
                asset_ticker="IWM",
                candidates=[_make_candidate()],
            )
        ],
        run_time_seconds=0.5,
    )


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class TestPromptBuilder:

    def test_candidates_context_has_regime_and_holding(self):
        from backend.core.explainer.prompt_builder import PromptBuilder
        pb = PromptBuilder()
        ctx = pb.build_candidates_context(
            asset_ticker="IWM",
            holding_notional=500_000,
            risk_profile=_make_risk_profile(),
            candidates=[_make_candidate()],
            regime=_make_regime(),
        )
        assert ctx["regime"]["label"] == "mid_vol"
        assert ctx["holding"]["ticker"] == "IWM"
        assert len(ctx["candidates"]) == 1

    def test_candidates_context_has_expected_candidate_fields(self):
        from backend.core.explainer.prompt_builder import PromptBuilder
        pb = PromptBuilder()
        ctx = pb.build_candidates_context(
            "IWM", 500_000, _make_risk_profile(),
            [_make_candidate()], _make_regime(),
        )
        c = ctx["candidates"][0]
        assert "ticker" in c
        assert "strategy" in c
        assert "delta" in c
        assert "score" in c

    def test_portfolio_context_has_holdings_list(self):
        from backend.core.explainer.prompt_builder import PromptBuilder
        pb = PromptBuilder()
        ctx = pb.build_portfolio_context(
            portfolio=_make_portfolio(),
            risk_summary=_make_portfolio_risk_summary(),
            hedge_output=_make_hedge_output(),
            regime=_make_regime(),
        )
        assert "holdings" in ctx
        assert ctx["portfolio"]["total_notional_usd"] == 500_000
        assert ctx["regime"]["vix_level"] == 18.5
        assert len(ctx["top_hedge_per_holding"]) == 1

    def test_portfolio_context_top_candidates_shape(self):
        from backend.core.explainer.prompt_builder import PromptBuilder
        ctx = PromptBuilder().build_portfolio_context(
            _make_portfolio(), _make_portfolio_risk_summary(),
            _make_hedge_output(), _make_regime(),
        )
        top = ctx["top_hedge_per_holding"][0]
        assert "holding" in top
        assert "strategy" in top
        assert "cost_usd" in top


# ── CandidateExplanation + PortfolioExplanation dataclasses ──────────────────

def test_candidate_explanation_defaults():
    exp = CandidateExplanation(ticker="SPY", strategy="Protective Put", asset_ticker="IWM")
    assert exp.when_works_best == ""
    assert exp.pros == []

def test_portfolio_explanation_defaults():
    exp = PortfolioExplanation()
    assert exp.summary == ""
    assert exp.key_risks == []

def test_candidate_explanation_fully_populated():
    exp = CandidateExplanation(
        ticker="SPY", strategy="Collar", asset_ticker="AAPL",
        when_works_best="Rising VIX", when_fails="Grinding rally",
        rationale="Hedges beta exposure cheaply.",
        pros=["Low cost"], cons=["Capped upside"],
    )
    assert exp.pros == ["Low cost"]
    assert exp.when_fails == "Grinding rally"


# ── Mock explainer contract ───────────────────────────────────────────────────

def test_mock_explainer_candidates_returns_list():
    ctx = {"holding": {"ticker": "IWM"}, "candidates": []}
    result = MockExplainer().explain_candidates(ctx)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].ticker == "SPY"

def test_mock_explainer_portfolio_returns_obj():
    result = MockExplainer().explain_portfolio({})
    assert isinstance(result, PortfolioExplanation)
    assert "systematic" in result.summary.lower()

def test_unavailable_explainer_returns_empty():
    expl = MockExplainer(available=False)
    assert not expl.is_available()


# ── LLMExplainerEngine ────────────────────────────────────────────────────────

class TestLLMExplainerEngine:

    def _make_engine(self, **mock_kwargs):
        from backend.engines.layer_05_llm_explainer import LLMExplainerEngine
        engine = LLMExplainerEngine.__new__(LLMExplainerEngine)
        from backend.core.explainer.prompt_builder import PromptBuilder
        engine.explainer = MockExplainer(**mock_kwargs)
        engine.prompt_builder = PromptBuilder()
        return engine

    def test_explain_populates_when_works_best(self):
        engine = self._make_engine()
        output = engine.explain(
            _make_hedge_output(), _make_portfolio(),
            _make_portfolio_risk_summary(), _make_regime(),
        )
        candidate = output.recommendations[0].candidates[0]
        assert "selloff" in candidate.when_works_best.lower()

    def test_explain_populates_portfolio_summary(self):
        engine = self._make_engine()
        output = engine.explain(
            _make_hedge_output(), _make_portfolio(),
            _make_portfolio_risk_summary(), _make_regime(),
        )
        assert "risk" in output.portfolio_summary.lower()

    def test_explain_sets_llm_provider(self):
        engine = self._make_engine()
        output = engine.explain(
            _make_hedge_output(), _make_portfolio(),
            _make_portfolio_risk_summary(), _make_regime(),
        )
        assert output.llm_provider == "mock"

    def test_explain_skips_when_unavailable(self):
        engine = self._make_engine(available=False)
        output = engine.explain(
            _make_hedge_output(), _make_portfolio(),
            _make_portfolio_risk_summary(), _make_regime(),
        )
        # Should return unchanged output (no portfolio_summary)
        assert output.portfolio_summary == ""
        # Rule-based rationale preserved
        assert output.recommendations[0].candidates[0].rationale == "Rule-based rationale."

    def test_explain_preserves_rule_rationale_when_llm_empty(self):
        """If LLM returns no matches, rule-based text stays."""
        class EmptyExplainer(MockExplainer):
            def explain_candidates(self, ctx):
                return []  # no matches
        from backend.engines.layer_05_llm_explainer import LLMExplainerEngine
        from backend.core.explainer.prompt_builder import PromptBuilder
        engine = LLMExplainerEngine.__new__(LLMExplainerEngine)
        engine.explainer = EmptyExplainer()
        engine.prompt_builder = PromptBuilder()
        output = engine.explain(
            _make_hedge_output(), _make_portfolio(),
            _make_portfolio_risk_summary(), _make_regime(),
        )
        assert output.recommendations[0].candidates[0].rationale == "Rule-based rationale."


# ── JSON extraction utility ───────────────────────────────────────────────────

def test_extract_json_strips_markdown_fence():
    from backend.core.explainer.claude_explainer import _extract_json
    raw = '```json\n[{"key": "value"}]\n```'
    out = _extract_json(raw)
    parsed = json.loads(out)
    assert parsed[0]["key"] == "value"

def test_extract_json_plain_passthrough():
    from backend.core.explainer.claude_explainer import _extract_json
    raw = '[{"key": "value"}]'
    assert _extract_json(raw) == raw

def test_extract_json_strips_fence_without_language():
    from backend.core.explainer.claude_explainer import _extract_json
    raw = '```\n{"a": 1}\n```'
    out = _extract_json(raw)
    assert json.loads(out) == {"a": 1}


# ── ClaudeExplainer parse helpers (no API call) ───────────────────────────────

class TestClaudeExplainerParsers:

    def _make_explainer(self):
        from backend.core.explainer.claude_explainer import ClaudeExplainer
        e = ClaudeExplainer.__new__(ClaudeExplainer)
        e._model = "claude-sonnet-4-6"
        e._api_key = "test"
        e._client = None
        return e

    def test_parse_candidates_valid_json(self):
        e = self._make_explainer()
        raw = json.dumps([{
            "ticker": "SPY", "strategy": "Protective Put",
            "when_works_best": "Crash", "when_fails": "Slow drift",
            "rationale": "Good hedge.", "pros": ["A"], "cons": ["B"],
        }])
        result = e._parse_candidates(raw, {"holding": {"ticker": "IWM"}})
        assert len(result) == 1
        assert result[0].ticker == "SPY"
        assert result[0].pros == ["A"]

    def test_parse_candidates_bad_json_returns_empty(self):
        e = self._make_explainer()
        result = e._parse_candidates("this is not json", {})
        assert result == []

    def test_parse_portfolio_valid_json(self):
        e = self._make_explainer()
        raw = json.dumps({
            "summary": "Portfolio is risky.",
            "key_risks": ["Beta", "VIX"],
            "regime_commentary": "High vol.",
            "top_recommendation": "Buy puts.",
        })
        result = e._parse_portfolio(raw)
        assert result.summary == "Portfolio is risky."
        assert "Beta" in result.key_risks

    def test_parse_portfolio_bad_json_returns_empty(self):
        e = self._make_explainer()
        result = e._parse_portfolio("not json at all")
        assert result.summary == ""
        assert result.key_risks == []

    def test_parse_candidates_single_object_not_array(self):
        """Claude sometimes returns a JSON object instead of array — handle it."""
        e = self._make_explainer()
        raw = json.dumps({
            "ticker": "QQQ", "strategy": "Collar",
            "when_works_best": "X", "when_fails": "Y",
            "rationale": "Z", "pros": [], "cons": [],
        })
        result = e._parse_candidates(raw, {"holding": {"ticker": "MSFT"}})
        assert len(result) == 1
        assert result[0].ticker == "QQQ"


# ── get_explainer factory ─────────────────────────────────────────────────────

def test_get_explainer_default_returns_claude():
    from backend.core.explainer import get_explainer
    from backend.core.explainer.claude_explainer import ClaudeExplainer
    expl = get_explainer("claude")
    assert isinstance(expl, ClaudeExplainer)

def test_get_explainer_huggingface():
    from backend.core.explainer import get_explainer
    from backend.core.explainer.huggingface_explainer import HuggingFaceExplainer
    expl = get_explainer("huggingface")
    assert isinstance(expl, HuggingFaceExplainer)


# ── HedgeOutput model accepts LLM fields ─────────────────────────────────────

def test_hedge_output_llm_fields():
    from backend.models.hedge_models import HedgeOutput
    output = HedgeOutput(
        portfolio_notional=500_000,
        regime="mid_vol",
        is_anomaly=False,
        vix_level=18.5,
        recommendations=[],
        run_time_seconds=1.0,
        portfolio_summary="Test summary",
        key_risks=["Risk 1", "Risk 2"],
        regime_commentary="Mid-vol.",
        top_recommendation="Buy puts.",
        llm_provider="claude",
    )
    assert output.portfolio_summary == "Test summary"
    assert len(output.key_risks) == 2
    assert output.llm_provider == "claude"
