"""
Layer 11 — Final LLM Explainer

Runs AFTER L6–L10 so the LLM has full quantitative context:
  - Actual option prices, Greeks (Δ/Γ/Θ/V/Λ)
  - Position sizes (n_contracts, total_cost)
  - Simulation results (VaR reduction, payoff at 10%/20% drop)
  - Composite scores (0–100) with sub-scores
  - Stress scenario outcomes

This replaces the preliminary L5 explanations with richer,
number-grounded narratives.

Flow:
  1. For each holding: build candidates context → call explainer.explain_candidates()
     → merge CandidateExplanation into InstrumentCandidate
  2. Build portfolio context → call explainer.explain_portfolio()
     → write to HedgeOutput top-level fields

Graceful degradation:
  - If LLM unavailable: keeps L5 rule-based rationale, returns unchanged
  - Per-candidate failures don't block portfolio explanation
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.models.portfolio_models import PortfolioInput
from backend.models.risk_models import RegimeState, PortfolioRiskSummary
from backend.core.explainer import get_explainer, PromptBuilder
from backend.core.explainer.base_explainer import CandidateExplanation
from backend.core.explainer.huggingface_explainer import HFRateLimitError

_RATE_LIMIT_NOTICE = (
    "AI explanations are temporarily unavailable — "
    "the free-tier usage limit has been reached. "
    "All hedge recommendations and risk data are unaffected."
)

logger = logging.getLogger(__name__)


def _merge_explanation(
    candidate: InstrumentCandidate,
    explanation: CandidateExplanation,
) -> None:
    """Write LLM explanation fields back onto candidate in-place."""
    if explanation.rationale:
        candidate.rationale = explanation.rationale
    if explanation.pros:
        candidate.pros = explanation.pros
    if explanation.cons:
        candidate.cons = explanation.cons
    if explanation.when_works_best:
        candidate.when_works_best = explanation.when_works_best
    if explanation.when_fails:
        candidate.when_fails = explanation.when_fails


def _build_risk_profile_lookup(
    risk_summary: PortfolioRiskSummary,
) -> Dict[str, object]:
    """Map ticker → RiskProfile for quick lookup."""
    return {p.ticker: p for p in risk_summary.risk_profiles}


class FinalLLMExplainerEngine:
    """
    L11 Final LLM Explainer Engine.

    run() — enrich all candidates + portfolio narrative with LLM explanations.
    """

    def run(
        self,
        hedge_output: HedgeOutput,
        portfolio: PortfolioInput,
        risk_summary: PortfolioRiskSummary,
        regime: RegimeState,
        provider: Optional[str] = None,
    ) -> HedgeOutput:
        t0 = time.perf_counter()

        explainer = get_explainer(provider)
        if not explainer.is_available():
            logger.warning(
                "L11: LLM provider '%s' unavailable — keeping L5 explanations",
                explainer.provider_name,
            )
            return hedge_output

        builder = PromptBuilder()
        risk_by_ticker = _build_risk_profile_lookup(risk_summary)

        # Collect (ticker, notional, risk_profile, candidates) for every holding
        position_tuples = []
        for rec in hedge_output.recommendations:
            ticker = rec.asset_ticker
            risk_profile = risk_by_ticker.get(ticker)
            if risk_profile is None:
                continue
            holding_notional = next(
                (h.shares * h.purchase_price for h in portfolio.stock_positions
                 if h.ticker == ticker),
                None,
            )
            if holding_notional is None:
                holding_notional = next(
                    (p.notional_value for p in risk_summary.risk_profiles
                     if p.ticker == ticker),
                    hedge_output.portfolio_notional,
                )
            position_tuples.append((ticker, holding_notional, risk_profile, rec.candidates))

        try:
            if explainer.supports_combined_call():
                # ── Single API call for entire portfolio ──────────────────────
                logger.info("L11: combined single-call mode (%d positions)", len(position_tuples))
                combined_ctx = builder.build_combined_context(
                    positions=position_tuples,
                    portfolio=portfolio,
                    risk_summary=risk_summary,
                    hedge_output=hedge_output,
                    regime=regime,
                )
                result = explainer.explain_all(combined_ctx)

                # Distribute candidate explanations back to each holding
                for rec in hedge_output.recommendations:
                    position_exps = result.candidates.get(rec.asset_ticker, {})
                    for cand in rec.candidates:
                        exp = position_exps.get(cand.strategy)
                        if exp:
                            _merge_explanation(cand, exp)

                # Write portfolio narrative
                hedge_output.portfolio_summary   = result.portfolio.summary
                hedge_output.key_risks           = result.portfolio.key_risks
                hedge_output.regime_commentary   = result.portfolio.regime_commentary
                hedge_output.top_recommendation  = result.portfolio.top_recommendation
                hedge_output.llm_provider        = explainer.provider_name
                logger.info("L11: combined call done (provider=%s)", explainer.provider_name)

            else:
                # ── Per-holding calls (fallback for Claude etc.) ───────────────
                for ticker, holding_notional, risk_profile, candidates in position_tuples:
                    ctx = builder.build_candidates_context(
                        asset_ticker=ticker,
                        holding_notional=holding_notional,
                        risk_profile=risk_profile,
                        candidates=candidates[:3],
                        regime=regime,
                    )
                    explanations = explainer.explain_candidates(ctx)
                    exp_map = {(e.ticker, e.strategy): e for e in explanations}
                    for rec in hedge_output.recommendations:
                        if rec.asset_ticker != ticker:
                            continue
                        for cand in rec.candidates:
                            exp = exp_map.get((cand.ticker, cand.strategy))
                            if exp:
                                _merge_explanation(cand, exp)

                port_ctx = builder.build_portfolio_context(
                    portfolio=portfolio,
                    risk_summary=risk_summary,
                    hedge_output=hedge_output,
                    regime=regime,
                )
                port_exp = explainer.explain_portfolio(port_ctx)
                hedge_output.portfolio_summary   = port_exp.summary
                hedge_output.key_risks           = port_exp.key_risks
                hedge_output.regime_commentary   = port_exp.regime_commentary
                hedge_output.top_recommendation  = port_exp.top_recommendation
                hedge_output.llm_provider        = explainer.provider_name
                logger.info("L11: per-position calls done (provider=%s)", explainer.provider_name)

        except HFRateLimitError:
            logger.warning("L11: HuggingFace free-tier rate limit hit")
            hedge_output.llm_notice   = _RATE_LIMIT_NOTICE
            hedge_output.llm_provider = explainer.provider_name

        except Exception as exc:
            logger.error("L11: explanation failed: %s", exc, exc_info=True)

        elapsed = time.perf_counter() - t0
        logger.info("L11 FinalLLMExplainerEngine done in %.3fs", elapsed)
        return hedge_output
