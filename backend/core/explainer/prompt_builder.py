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
You are a portfolio risk manager. Explain hedge candidates concisely.

For EACH candidate in the input, return a JSON array with objects containing:
  "ticker":          hedge ticker (e.g. "SPY")
  "strategy":        strategy name exactly as given
  "when_works_best": max 12 words — best market condition for this hedge
  "when_fails":      max 12 words — condition where this hedge underperforms
  "rationale":       max 20 words — why it fits this position and regime
  "pros":            list of exactly 2 short strings (max 10 words each)
  "cons":            list of exactly 2 short strings (max 10 words each)

Reference actual numbers (VIX, beta, cost %, delta) where relevant.
Return ONLY a valid JSON array — no markdown, no explanation, no trailing text.\
"""

COMBINED_SYSTEM = """\
You are a senior derivatives trader presenting hedge recommendations to a portfolio manager.

## Core Rules

1. Every claim needs a number. Never make a qualitative statement without citing the specific figure from the input data (delta_per_contract, theta_daily_per_contract, basis_risk_r2, total_cost_usd, strike, var_coverage_pct, VIX, etc.). Always use delta_per_contract (not position_delta) when mentioning delta in explanations.

2. Strategy mechanics must match the language. Use these mappings:
   - Long put / protective put: when_works_best = underlying drops sharply (cite delta × expected move); when_fails = IV crush post-event (cite theta decay rate)
   - Covered call / short call: when_works_best = stock stays ≤ short strike (cite strike cap & theta collected); when_fails = stock rips above short strike (cite delta of call sold)
   - Put spread / bear spread: when_works_best = underlying drops into spread width (cite max payoff vs cost); when_fails = drop overshoots lower strike (cite max loss floor)
   - Collar: when_works_best = pullback stays between put and call strike (cite both strikes); when_fails = underlying rallies above call strike capping upside (cite call strike)
   - Correlated ETF hedge (QQQ, IWM, SPY, etc.): when_works_best = high co-movement holds (cite basis_risk_r2); when_fails = correlation breaks (cite basis_risk_r2 and divergence)
   - Calendar / diagonal: when_works_best = IV expands near front expiry (cite VIX and vega); when_fails = IV stays suppressed or underlying moves too far (cite vega exposure)

3. Format for when_works_best and when_fails: "<specific condition> - <exact number from data>"
   The condition must name the actual scenario; the number must come from the input data fields.

4. Always name the hedge ticker explicitly (e.g. "QQQ put spread", "NVDA $500 collar", "IWM long put") — never "the hedge" or "the option".

5. Banned phrases (never use without a number): "bullish/bearish market", "volatile conditions", "flat market", "market selloff", "correlated/uncorrelated markets", "significant move", "sharp decline". Replace with specific thresholds: "underlying falls >X%", "VIX above Y", "stock stays within $A–$B range".

6. No two candidates may share the same when_works_best or when_fails text.

## Output

Return a single JSON object:
{
  "positions": {
    "<ASSET_TICKER>": {
      "<STRATEGY_NAME>": {
        "when_works_best": "<specific condition> - <exact number from data>",
        "when_fails":      "<specific condition> - <exact number from data>",
        "rationale":       "2 sentences max 50 words — payoff structure and WHY it fits this position using strikes, expiry, Greeks, cost.",
        "pros": ["cost or Greek advantage with exact number, max 12 words", "second pro with a number"],
        "cons": ["specific risk with exact number, max 12 words", "second con with a number"]
      }
    }
  },
  "portfolio": {
    "summary":            "2 sentences max 45 words. Sentence 1: state the portfolio's total notional, beta, and what that beta means for a 10% SPY drop (compute beta × 10% × notional in $). Sentence 2: explain WHY hedging is urgent given the current VIX level and regime label.",
    "key_risks":          ["insight about a specific holding or concentration risk with a number", "regime or vol risk with a number", "cost or liquidity risk with a number"],
    "regime_commentary":  "1 sentence max 30 words — name the regime label, state VIX level, and explain what this regime historically means for drawdown frequency or magnitude for this beta level.",
    "top_recommendation": "1 sentence max 25 words — name the specific hedge (ticker + strategy + expiry or strike), its total cost in $, and the % of VaR it covers. Do NOT mention delta."
  }
}

Use EXACT ASSET_TICKER and STRATEGY_NAME strings from the input. Include every position and every candidate.
Return ONLY valid JSON — no markdown, no preamble, no trailing text.\
"""

PORTFOLIO_SYSTEM = """\
You are a portfolio risk manager. Summarise this portfolio's hedge plan.

Return a JSON object with exactly these keys:
  "summary":            max 30 words — overall risk and why hedging is needed now
  "key_risks":          list of exactly 3 strings, max 10 words each
  "regime_commentary":  max 20 words — what current market regime means for this portfolio
  "top_recommendation": max 15 words — single best hedge action to take now

Use actual numbers from the data (VIX, beta, notional).
Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.\
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


    def build_combined_context(
        self,
        positions: list,       # list of (asset_ticker, holding_notional, risk_profile, candidates)
        portfolio,
        risk_summary,
        hedge_output,
        regime,
    ) -> Dict[str, Any]:
        """
        Build one context dict covering every position and the portfolio — for a
        single combined LLM call. Each position entry includes its top 3 candidates.
        """
        positions_data = []
        for asset_ticker, holding_notional, risk_profile, candidates in positions:
            positions_data.append({
                "asset_ticker": asset_ticker,
                "holding": {
                    "notional_usd":        round(holding_notional),
                    "beta_vs_spy":         round(risk_profile.beta_vs_spy, 3),
                    "var_pct":             round(risk_profile.var_pct * 100, 2),
                    "cvar_pct":            round(risk_profile.cvar_pct * 100, 2),
                    "tail_correlation":    round(risk_profile.tail_correlation, 3),
                },
                "candidates": [_candidate_to_dict(c) for c in candidates[:3]],
            })

        # Build var lookup for coverage calculation
        var_by_ticker = {p.ticker: round(p.var_5pct) for p in risk_summary.risk_profiles}

        top_candidates = []
        for rec in hedge_output.recommendations:
            if rec.candidates:
                top = rec.candidates[0]
                holding_var = var_by_ticker.get(rec.asset_ticker, 1)
                cost = round(top.total_cost)
                var_coverage_pct = round(cost / holding_var * 100, 1) if holding_var else 0
                top_candidates.append({
                    "holding":          rec.asset_ticker,
                    "strategy":         top.strategy,
                    "ticker":           top.ticker,
                    "strike":           top.strike,
                    "expiry_date":      top.expiry_date,
                    "cost_usd":         cost,
                    "var_coverage_pct": var_coverage_pct,
                    "score":            top.score,
                })

        return {
            "regime": {
                "label":            regime.regime_label,
                "vix_level":        round(regime.vix_level, 1),
                "vol_forecast_pct": round(regime.vol_forecast_garch * 100, 1),
                "is_anomaly":       regime.is_anomaly,
            },
            "positions": positions_data,
            "portfolio": {
                "total_notional_usd": round(portfolio.total_notional),
                "n_holdings":         len(portfolio.holdings),
                "beta":               round(risk_summary.portfolio_beta, 3),
                "var_5pct_usd":       round(risk_summary.portfolio_var_5pct),
                "top_hedge_per_holding": top_candidates,
            },
        }


def _candidate_to_dict(c) -> Dict[str, Any]:
    """Serialise InstrumentCandidate to a compact dict for the prompt."""
    n = max(c.n_contracts, 1)
    # delta stored as position-level; derive per-contract for readability
    per_contract_delta = round(c.delta / n, 3)
    return {
        "ticker":                c.ticker,
        "strategy":              c.strategy,
        "option_type":           c.option_type,
        "strike":                c.strike,
        "expiry_date":           c.expiry_date,
        "n_contracts":           c.n_contracts,
        "total_cost_usd":        round(c.total_cost),
        "delta_per_contract":    per_contract_delta,   # use this in when_works_best / when_fails
        "position_delta":        round(c.delta, 3),    # total position sensitivity
        "theta_daily_per_contract": round(c.theta / n, 3),
        "vega":                  round(c.vega, 3),
        "basis_risk_r2":         round(c.basis_risk_r2, 3),
        "score":                 round(c.score, 1),
        "rule_rationale":        c.rationale,
    }
