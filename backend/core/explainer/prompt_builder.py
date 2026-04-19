"""
Builds structured context dicts used as LLM prompt inputs.

Keeps all prompt logic in one place so swapping LLM providers
(Claude ↔ Ollama ↔ OpenAI) only requires changing the explainer class,
not the prompt construction.

Two prompt types:
  1. candidates_context — per-holding, per-candidate explanations
  2. portfolio_context  — portfolio-level summary and risk narrative
"""
from typing import Dict, Any, List


# ── System prompts ─────────────────────────────────────────────────────────────

CANDIDATES_SYSTEM = """\
You are a senior portfolio risk manager and derivatives specialist.
Your job is to explain hedge recommendations to a sophisticated investor
in plain, precise language — no jargon without definition, no fluff.

You will receive a JSON object describing:
  - The current market regime and volatility environment
  - A stock holding with its risk metrics (beta, VaR, tail correlation)
  - 1–5 ranked hedge candidates (options strategies)

For EACH candidate, return a JSON array with objects containing:
  "ticker":         hedge instrument ticker (e.g. "SPY")
  "strategy":       strategy name (e.g. "Protective Put")
  "when_works_best": one sentence on conditions where this hedge excels
  "when_fails":      one sentence on conditions where this hedge underperforms
  "rationale":       2–3 sentences: why this hedge fits this position and regime
  "pros":            list of 2–3 specific advantages (strings)
  "cons":            list of 2–3 specific risks or drawbacks (strings)

Be specific. Reference the actual numbers (beta, VaR %, cost %, delta, VIX level).
Do not repeat generic boilerplate.
Return ONLY valid JSON — no markdown, no preamble, no trailing text.\
"""

PORTFOLIO_SYSTEM = """\
You are a senior portfolio risk manager.
You will receive a JSON summary of a multi-asset portfolio with risk metrics
and the top-ranked hedge recommendation for each holding.

Return a JSON object with exactly these keys:
  "summary":            2–4 sentences: overall portfolio risk picture and
                        why hedging is warranted right now
  "key_risks":          list of 3–5 specific risks (strings), most important first
  "regime_commentary":  1–2 sentences: what the current market regime implies
                        for this portfolio specifically
  "top_recommendation": one sentence naming the single best hedge action to take now

Reference the actual data: VIX level, portfolio beta, total notional, largest positions.
Return ONLY valid JSON — no markdown, no preamble, no trailing text.\
"""


class PromptBuilder:
    """
    Builds context dicts from Pydantic model objects.
    Context dicts are serialised to JSON in the explainer layer.
    """

    def build_candidates_context(
        self,
        asset_ticker: str,
        holding_notional: float,
        risk_profile,          # RiskProfile Pydantic model
        candidates: list,      # List[InstrumentCandidate]
        regime,                # RegimeState Pydantic model
    ) -> Dict[str, Any]:
        """
        Build context for per-candidate explanation of one holding.
        """
        return {
            "regime": {
                "label":           regime.regime_label,
                "vix_level":       round(regime.vix_level, 1),
                "vol_forecast_pct": round(regime.vol_forecast_garch * 100, 1),
                "is_anomaly":      regime.is_anomaly,
                "realized_vol_20d": round(regime.realized_vol_20d * 100, 1),
            },
            "holding": {
                "ticker":          asset_ticker,
                "asset_class":     risk_profile.asset_class,
                "notional_usd":    round(holding_notional),
                "beta_vs_spy":     round(risk_profile.beta_vs_spy, 3),
                "var_5pct_usd":    round(risk_profile.var_5pct),
                "var_pct":         round(risk_profile.var_pct * 100, 2),
                "cvar_pct":        round(risk_profile.cvar_pct * 100, 2),
                "tail_correlation": round(risk_profile.tail_correlation, 3),
                "optimal_hedge_ratio": round(risk_profile.optimal_hedge_ratio, 3),
            },
            "candidates": [
                _candidate_to_dict(c) for c in candidates
            ],
        }

    def build_portfolio_context(
        self,
        portfolio,             # PortfolioInput
        risk_summary,          # PortfolioRiskSummary
        hedge_output,          # HedgeOutput
        regime,                # RegimeState
    ) -> Dict[str, Any]:
        """
        Build context for portfolio-level narrative.
        """
        # Top candidate per holding
        top_candidates = []
        for rec in hedge_output.recommendations:
            if rec.candidates:
                top = rec.candidates[0]
                top_candidates.append({
                    "holding":    rec.asset_ticker,
                    "strategy":   top.strategy,
                    "hedge_etf":  top.ticker,
                    "n_contracts": top.n_contracts,
                    "cost_usd":   round(top.total_cost),
                    "score":      top.score,
                })

        return {
            "portfolio": {
                "total_notional_usd": round(portfolio.total_notional),
                "n_holdings":         len(portfolio.holdings),
                "beta":               round(risk_summary.portfolio_beta, 3),
                "var_5pct_usd":       round(risk_summary.portfolio_var_5pct),
                "cvar_5pct_usd":      round(risk_summary.portfolio_cvar_5pct),
                "concentration_top1": round(risk_summary.concentration_top1 * 100, 1),
                "hedge_horizon_days": portfolio.hedge_horizon_days,
                "max_hedge_cost_pct": round(portfolio.max_hedge_cost_pct * 100, 2),
            },
            "regime": {
                "label":           regime.regime_label,
                "vix_level":       round(regime.vix_level, 1),
                "vol_forecast_pct": round(regime.vol_forecast_garch * 100, 1),
                "is_anomaly":      regime.is_anomaly,
            },
            "holdings": [
                {
                    "ticker":    p.ticker,
                    "notional":  round(p.notional_value),
                    "beta":      round(p.beta_vs_spy, 3),
                    "var_pct":   round(p.var_pct * 100, 2),
                }
                for p in risk_summary.risk_profiles
            ],
            "top_hedge_per_holding": top_candidates,
        }


def _candidate_to_dict(c) -> Dict[str, Any]:
    """Serialise InstrumentCandidate to a compact dict for the prompt."""
    return {
        "ticker":           c.ticker,
        "strategy":         c.strategy,
        "option_type":      c.option_type,
        "strike":           c.strike,
        "expiry_date":      c.expiry_date,
        "n_contracts":      c.n_contracts,
        "total_cost_usd":   round(c.total_cost),
        "delta":            round(c.delta, 3),
        "gamma":            round(c.gamma, 5),
        "theta_daily":      round(c.theta, 3),
        "vega":             round(c.vega, 3),
        "lambda_leverage":  round(c.lambda_leverage, 2),
        "basis_risk_r2":    round(c.basis_risk_r2, 3),
        "score":            round(c.score, 1),
        "rule_rationale":   c.rationale,
    }
