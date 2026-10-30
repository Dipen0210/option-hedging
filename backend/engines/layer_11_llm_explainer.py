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
from backend.core import disclosures
from backend.models.portfolio_models import PortfolioInput
from backend.models.risk_models import RegimeState, PortfolioRiskSummary
from backend.core.explainer import get_explainer, PromptBuilder
from backend.core.explainer.base_explainer import CandidateExplanation
from backend.core.explainer.huggingface_explainer import HFRateLimitError

_RATE_LIMIT_NOTICE = (
    "AI explanations are temporarily unavailable — "
    "the free-tier usage limit has been reached. "
    "The narrative below was generated from the computed figures instead. "
    "All hedge recommendations and risk data are unaffected."
)

_SUBSTITUTED_NOTICE = (
    "The language model returned no analysis for this run, so the narrative "
    "below was generated directly from the computed figures instead. All "
    "hedge recommendations and risk data are unaffected."
)

_PARTIAL_NOTICE = (
    "The language model returned only part of its analysis for this run; the "
    "remaining sections below ({fields}) were generated directly from the "
    "computed figures. All hedge recommendations and risk data are unaffected."
)

_NARRATIVE_FIELD_LABELS = {
    "portfolio_summary":  "portfolio summary",
    "key_risks":          "key risks",
    "regime_commentary":  "regime commentary",
    "top_recommendation": "top recommendation",
}

_UNAVAILABLE_NOTICE = (
    "AI explanations are unavailable (no language-model provider is "
    "configured or reachable). The narrative below was generated directly "
    "from the computed figures instead. All hedge recommendations and risk "
    "data are unaffected."
)

_FAILED_NOTICE = (
    "AI explanations could not be generated for this run. The narrative "
    "below was generated directly from the computed figures instead. All "
    "hedge recommendations and risk data are unaffected."
)

logger = logging.getLogger(__name__)


def _fmt_money(x: float) -> str:
    """$1.2M / $34.5K / $412 — readable at a glance, no false precision."""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1_000_000:
        return f"{sign}${a / 1_000_000:,.2f}M"
    if a >= 1_000:
        return f"{sign}${a / 1_000:,.1f}K"
    return f"{sign}${a:,.0f}"


def _var_horizon_phrase(risk_summary: PortfolioRiskSummary) -> str:
    """
    The leading phrase for a VaR sentence, naming the horizon the figure
    actually covers.

    The VaR in `risk_summary` is scaled to the portfolio's hedge horizon,
    which the user chooses (default 180 calendar days). This sentence used to
    open "One-day 95% VaR is ...", which understated a four-month tail loss as
    an overnight one by roughly the square root of the horizon — about 9x for
    a 120-day hedge. Returns an empty string when the horizon is unknown, so
    the sentence reads "95% VaR is ..." with no horizon claimed at all rather
    than defaulting to a wrong one.
    """
    days = getattr(risk_summary, "var_horizon_calendar_days", 0) or 0
    if days <= 0:
        return ""
    if days == 1:
        return "One-day "
    return f"Over the {days}-day hedge horizon, "


def _deterministic_narrative(
    hedge_output: HedgeOutput,
    risk_summary: PortfolioRiskSummary,
    regime: RegimeState,
) -> List[str]:
    """
    Compose the portfolio narrative from the numbers the pipeline computed.

    Three paths through `run()` used to leave the narrative fields completely
    empty: an unavailable provider returned `hedge_output` untouched, a rate
    limit set a notice but no text, and the catch-all logged and moved on. In
    every one of those cases `portfolio_summary`, `key_risks`,
    `regime_commentary` and `top_recommendation` stayed at their empty
    defaults, so the UI rendered nothing at all — no narrative and, in two of
    the three cases, no explanation of why. A user cannot distinguish "the
    model had nothing to say about your risk" from "the explanation service
    is down", and the first reading is the dangerous one.

    Every sentence written here is a restatement of a figure already in
    `hedge_output` or `risk_summary` — no interpretation, no market view, no
    forecast. That is the point: it is the one form of explanation that
    cannot be wrong in a way the numbers are not already wrong. It is
    deliberately plainer than the LLM text, and `llm_notice` says so, so
    nobody mistakes it for a model's analysis.

    Fields that already hold text are left alone, so this also serves as a
    backstop for a partial LLM response.

    Returns the names of the fields it filled. The caller needs them because
    this function running at all is a disclosable event: an end-to-end run
    found the provider reachable and raising nothing, yet returning blank
    narrative fields, so this backstop wrote all four and the response went
    out with `llm_provider="huggingface"` and an empty `llm_notice`. Template
    text was presented as a model's analysis, which is the silent substitution
    the rest of this pipeline is built to avoid. An empty return value means
    the model genuinely wrote everything.
    """
    filled: List[str] = []
    notional = hedge_output.portfolio_notional
    n_holdings = len(hedge_output.recommendations)
    hedged = [r for r in hedge_output.recommendations if r.candidates]

    # ── Summary ──────────────────────────────────────────────────────────────
    if not hedge_output.portfolio_summary:
        parts = [
            f"{_fmt_money(notional)} across {n_holdings} "
            f"holding{'s' if n_holdings != 1 else ''}."
        ]
        if risk_summary.portfolio_var_5pct:
            parts.append(
                f"{_var_horizon_phrase(risk_summary)}95% VaR is "
                f"{_fmt_money(abs(risk_summary.portfolio_var_5pct))}"
                f" and 95% expected shortfall "
                f"{_fmt_money(abs(risk_summary.portfolio_cvar_5pct))}."
            )
        if risk_summary.portfolio_beta:
            parts.append(
                f"Portfolio beta to SPY is {risk_summary.portfolio_beta:.2f}."
            )
        if hedge_output.hedge_budget_total > 0:
            cost = hedge_output.hedge_portfolio_cost
            if cost < 0:
                parts.append(
                    f"The selected hedges are a net credit of "
                    f"{_fmt_money(abs(cost))}."
                )
            else:
                parts.append(
                    f"The selected hedges cost {_fmt_money(cost)}, "
                    f"{hedge_output.hedge_budget_utilization:.0%} of the "
                    f"{_fmt_money(hedge_output.hedge_budget_total)} budget."
                )
        if len(hedged) < n_holdings:
            parts.append(
                f"{n_holdings - len(hedged)} of {n_holdings} holdings have no "
                f"affordable hedge and are shown unhedged."
            )
        hedge_output.portfolio_summary = " ".join(parts)
        filled.append("portfolio_summary")

    # ── Key risks ────────────────────────────────────────────────────────────
    if not hedge_output.key_risks:
        risks: List[str] = []

        if risk_summary.concentration_top1 > 0.30:
            risks.append(
                f"Concentration: largest holding is "
                f"{risk_summary.concentration_top1:.0%} of the portfolio"
            )
        if risk_summary.portfolio_beta > 1.1:
            risks.append(
                f"Beta {risk_summary.portfolio_beta:.2f} — amplifies market moves"
            )

        # Basis risk, premium provenance and liquidity are per-candidate facts
        # the user would otherwise have to open every card to find.
        tops = [
            sorted(r.candidates, key=lambda c: c.score, reverse=True)[0]
            for r in hedged
        ]
        weak_basis = [c for c in tops if 0 < c.basis_risk_r2 < 0.50]
        if weak_basis:
            risks.append(
                f"Basis risk: {len(weak_basis)} recommended hedge(s) explain "
                f"under half the variance of what they hedge"
            )
        modelled = [c for c in tops if c.premium_source != "market_mid"]
        if modelled:
            risks.append(
                f"{len(modelled)} of {len(tops)} premiums are model estimates, "
                f"not executable quotes"
            )
        illiquid = [c for c in tops if c.liquidity_ok is False]
        if illiquid:
            risks.append(
                f"{len(illiquid)} recommended contract(s) failed the liquidity "
                f"screen — exiting may be costly"
            )
        if regime.is_anomaly:
            risks.append(
                "Market conditions flagged as anomalous — historical "
                "relationships are less reliable"
            )
        hedge_output.key_risks = risks
        filled.append("key_risks")

    # ── Regime commentary ────────────────────────────────────────────────────
    if not hedge_output.regime_commentary:
        label = regime.regime_label.replace("_", " ")
        bits = [f"Regime: {label}, VIX {regime.vix_level:.1f}."]
        if regime.realized_vol_20d:
            bits.append(f"20-day realized vol {regime.realized_vol_20d:.1%}.")
        if regime.vol_forecast_garch:
            bits.append(f"GARCH(1,1) forecast {regime.vol_forecast_garch:.1%}.")
        if hedge_output.regime_degraded:
            bits.append(
                f"Detected via {hedge_output.regime_source} rather than the "
                f"full model: {hedge_output.regime_degradation_reason}"
            )
        elif regime.soft_membership:
            # The GMM's posterior over regimes; the weight on the one it chose
            # is what "confidence" means here. `RegimeState` exposes no scalar
            # confidence field, so taking the max of the membership vector is
            # the honest reading rather than inventing a number.
            bits.append(f"Confidence {max(regime.soft_membership):.0%}.")
        hedge_output.regime_commentary = " ".join(bits)
        filled.append("regime_commentary")

    # ── Top recommendation ───────────────────────────────────────────────────
    if not hedge_output.top_recommendation:
        best_rec, best_cand = None, None
        for rec in hedged:
            top = sorted(rec.candidates, key=lambda c: c.score, reverse=True)[0]
            if best_cand is None or top.score > best_cand.score:
                best_rec, best_cand = rec, top

        if best_cand is None:
            hedge_output.top_recommendation = (
                "No hedge is recommended: no candidate met the budget, "
                "liquidity and basis-risk requirements for any holding."
            )
        else:
            ident = best_cand.contract_symbol or (
                f"{best_cand.ticker} {best_cand.expiry_date} "
                f"{best_cand.strike} {(best_cand.option_type or '').upper()}"
            ).strip()
            cost = best_cand.total_cost
            hedge_output.top_recommendation = (
                f"Highest-scoring hedge ({best_cand.score:.0f}/100): "
                f"{best_cand.strategy} on {best_rec.asset_ticker} using "
                f"{ident}, {best_cand.n_contracts} contract"
                f"{'s' if best_cand.n_contracts != 1 else ''} for "
                f"{_fmt_money(abs(cost))}"
                f"{' credit' if cost < 0 else ''}"
                f"{f', R2={best_cand.basis_risk_r2:.2f}' if best_cand.basis_risk_r2 > 0 else ''}"
                f"{'' if best_cand.premium_source == 'market_mid' else ' (premium is a model estimate, not a live quote)'}."
            )
        filled.append("top_recommendation")

    return filled


def _merge_explanation(
    candidate: InstrumentCandidate,
    explanation: CandidateExplanation,
) -> None:
    """
    Write LLM explanation fields back onto candidate in-place.

    `cons` is replaced rather than merged, which is right for prose and was
    wrong for the product's own disclosures: they live in the same list, so a
    model that returned its own bullets used to delete "premium shown is
    theoretical" and leave a model price on screen looking like a quote. The
    restore puts those back. See backend/core/disclosures.py.
    """
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
    disclosures.restore(candidate)


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
                "L11: LLM provider '%s' unavailable — generating the narrative "
                "from the computed figures instead",
                explainer.provider_name,
            )
            hedge_output.llm_notice = _UNAVAILABLE_NOTICE
            _deterministic_narrative(hedge_output, risk_summary, regime)
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
            # Was: log and carry on with every narrative field still empty, and
            # no notice. The user saw a blank analysis with nothing to say the
            # explainer had failed.
            hedge_output.llm_notice = _FAILED_NOTICE

        # Backstop for every path, including a successful call that returned
        # some fields blank: anything still empty is filled from the computed
        # figures. Populated fields are left untouched.
        filled = _deterministic_narrative(hedge_output, risk_summary, regime)

        # If the backstop wrote anything and no notice already explains why,
        # say so. Without this the response carried a provider name, no
        # notice, and template text in every narrative field — indistinguishable
        # from a model that had actually analysed the portfolio.
        if filled and not hedge_output.llm_notice:
            if len(filled) == len(_NARRATIVE_FIELD_LABELS):
                hedge_output.llm_notice = _SUBSTITUTED_NOTICE
            else:
                hedge_output.llm_notice = _PARTIAL_NOTICE.format(
                    fields=", ".join(
                        _NARRATIVE_FIELD_LABELS.get(f, f) for f in filled
                    )
                )

        elapsed = time.perf_counter() - t0
        logger.info("L11 FinalLLMExplainerEngine done in %.3fs", elapsed)
        return hedge_output
