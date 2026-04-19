"""
Layer 10 — Hedge Evaluator & Scorer

Computes a 0–100 composite score for every candidate using a
weighted multi-factor model. Runs after L6/L7/L8/L9 (all data available).

Scoring dimensions (weights sum to 100):
  ┌──────────────────────────────┬────────┐
  │ Dimension                    │ Weight │
  ├──────────────────────────────┼────────┤
  │ Cost efficiency              │  25    │
  │ Downside protection          │  25    │
  │ Greeks alignment             │  15    │
  │ Basis risk (R²)              │  15    │
  │ Regime fit                   │  10    │
  │ Simulation (VaR reduction)   │  10    │
  └──────────────────────────────┴────────┘

Each dimension is normalised 0–1 before weighting.
Final score is rounded to 1 decimal place.
Also computes efficiency_ratio (|lambda| per $ of premium).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.models.risk_models import RegimeState

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
_W_COST         = 0.25
_W_PROTECTION   = 0.25
_W_GREEKS       = 0.15
_W_BASIS        = 0.15
_W_REGIME       = 0.10
_W_SIMULATION   = 0.10

# Regime → preferred asset classes / strategies
_REGIME_PREFERRED = {
    "high_vol":  {"equity", "put", "bear_put_spread", "collar"},
    "mid_vol":   {"equity", "collar", "commodity", "bond"},
    "low_vol":   {"bond", "fx", "credit", "collar", "covered_call"},
    "unknown":   {"equity"},
}


def _cost_score(candidate: InstrumentCandidate, holding_notional: float) -> float:
    """
    Higher score = cheaper hedge relative to protection offered.
    efficiency = max_protection / total_cost (higher is better)
    Normalised: sigmoid-like scale capped at 1.0.
    """
    if candidate.total_cost <= 0:
        return 0.5  # forward / free instrument — neutral cost
    if candidate.max_protection <= 0:
        return 0.0
    ratio = candidate.max_protection / candidate.total_cost
    # score = min(ratio / 20, 1.0)  — ratio of 20x → full score
    return min(ratio / 20.0, 1.0)


def _protection_score(candidate: InstrumentCandidate, holding_notional: float) -> float:
    """
    How much of the holding's downside does this hedge cover?
    Uses sim_payoff_10pct_drop if available, else max_protection.
    """
    payoff_10 = candidate.extended_metrics.get("sim_payoff_10pct_drop", 0.0)
    expected_loss_10 = holding_notional * 0.10

    if payoff_10 > 0 and expected_loss_10 > 0:
        coverage = payoff_10 / expected_loss_10
        return min(coverage, 1.0)

    # Fallback: max_protection vs notional
    if holding_notional > 0 and candidate.max_protection > 0:
        return min(candidate.max_protection / holding_notional, 1.0)
    return 0.0


def _greeks_score(candidate: InstrumentCandidate) -> float:
    """
    Delta alignment: for a protective put, delta should be ≈ -0.30 to -0.50.
    Penalty if delta is near zero (OTM) or near -1 (deep ITM overpay).
    IR/Credit instruments (no delta): score by DV01 presence.
    """
    asset_class = candidate.asset_class

    if asset_class in ("bond", "credit"):
        # Score by DV01 / CS01 presence
        has_dv01 = candidate.extended_metrics.get("dv01", 0.0) > 0
        has_cs01 = candidate.extended_metrics.get("cs01", 0.0) > 0
        return 0.8 if (has_dv01 or has_cs01) else 0.3

    delta = abs(candidate.delta)
    if delta <= 0:
        return 0.0

    # Optimal delta range for protection: 0.20 – 0.50
    if 0.20 <= delta <= 0.50:
        return 1.0
    elif delta < 0.20:
        return delta / 0.20          # too OTM
    else:
        return max(0.0, 1.0 - (delta - 0.50) / 0.50)   # too deep ITM


def _basis_score(candidate: InstrumentCandidate) -> float:
    """R² of hedge vs asset. Higher = lower basis risk."""
    r2 = candidate.basis_risk_r2
    if r2 <= 0:
        return 0.5   # unknown basis — neutral
    return min(r2, 1.0)


def _regime_score(candidate: InstrumentCandidate, regime: RegimeState) -> float:
    """
    Does this instrument fit the current market regime?
    Simple rule: preferred instruments get 1.0, others get 0.4.
    """
    regime_label = (regime.regime_label if regime else "unknown").lower()
    preferred = _REGIME_PREFERRED.get(regime_label, _REGIME_PREFERRED["unknown"])

    asset_class = candidate.asset_class.lower()
    strategy    = candidate.strategy.lower().replace(" ", "_")
    opt_type    = (candidate.option_type or "").lower()

    # Check any overlap
    matches = {asset_class, strategy, opt_type} & preferred
    if matches:
        return 1.0

    # High vol: all puts score well regardless
    if regime_label == "high_vol" and opt_type == "put":
        return 0.9

    return 0.4


def _simulation_score(candidate: InstrumentCandidate) -> float:
    """VaR reduction % from simulation. 0 = no reduction, 1 = full."""
    return float(candidate.extended_metrics.get("sim_var_reduction_pct", 0.0))


def score_candidate(
    candidate: InstrumentCandidate,
    holding_notional: float,
    regime: RegimeState,
) -> float:
    """Compute composite score 0–100."""
    s_cost   = _cost_score(candidate, holding_notional)
    s_prot   = _protection_score(candidate, holding_notional)
    s_greeks = _greeks_score(candidate)
    s_basis  = _basis_score(candidate)
    s_regime = _regime_score(candidate, regime)
    s_sim    = _simulation_score(candidate)

    composite = (
        _W_COST       * s_cost   +
        _W_PROTECTION * s_prot   +
        _W_GREEKS     * s_greeks +
        _W_BASIS      * s_basis  +
        _W_REGIME     * s_regime +
        _W_SIMULATION * s_sim
    )

    score = round(composite * 100, 1)

    # Store sub-scores for transparency
    candidate.extended_metrics.update({
        "score_cost":       round(s_cost, 3),
        "score_protection": round(s_prot, 3),
        "score_greeks":     round(s_greeks, 3),
        "score_basis":      round(s_basis, 3),
        "score_regime":     round(s_regime, 3),
        "score_simulation": round(s_sim, 3),
    })

    # Efficiency ratio: |lambda| per $ of premium (normalised)
    if candidate.total_cost > 0:
        candidate.efficiency_ratio = round(
            abs(candidate.lambda_leverage) / max(candidate.total_cost, 1.0) * 1000, 4
        )

    return score


class EvaluatorEngine:
    """
    L10 Evaluator Engine.

    run() — score all candidates, write candidate.score
    """

    def run(
        self,
        hedge_output: HedgeOutput,
        regime: RegimeState,
        holding_notionals: Dict[str, float],
    ) -> HedgeOutput:
        t0 = time.perf_counter()
        n_scored = 0

        for rec in hedge_output.recommendations:
            notional = holding_notionals.get(rec.asset_ticker, hedge_output.portfolio_notional)
            for cand in rec.candidates:
                cand.score = score_candidate(cand, notional, regime)
                n_scored += 1
                logger.debug(
                    "Scored %s [%s] %.1f (cost=%.2f prot=%.2f greeks=%.2f)",
                    cand.ticker, cand.strategy, cand.score,
                    cand.extended_metrics.get("score_cost", 0),
                    cand.extended_metrics.get("score_protection", 0),
                    cand.extended_metrics.get("score_greeks", 0),
                )

        elapsed = time.perf_counter() - t0
        logger.info("L10 EvaluatorEngine: scored %d candidates in %.3fs", n_scored, elapsed)
        return hedge_output
