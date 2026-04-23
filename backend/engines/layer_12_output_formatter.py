"""
Layer 12 — Output Formatter

Final layer. Sorts, ranks, and assembles the clean HedgeOutput
ready for the API response.

Operations:
  1. Sort candidates within each recommendation by score (desc)
  2. Assign rank (1 = best) to each HedgeRecommendation
  3. Trim to TOP_N_CANDIDATES_PER_HOLDING if over limit
  4. Sort recommendations by asset risk (highest VaR first)
  5. Compute portfolio-level summary stats
  6. Stamp run_time_seconds on HedgeOutput
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from backend.models.hedge_models import HedgeOutput, HedgeRecommendation, InstrumentCandidate
from backend.models.risk_models import PortfolioRiskSummary

logger = logging.getLogger(__name__)

TOP_N_CANDIDATES_PER_HOLDING = 5


def _sort_candidates(candidates: List[InstrumentCandidate]) -> List[InstrumentCandidate]:
    """Sort by score desc, then by efficiency_ratio desc as tiebreaker."""
    return sorted(
        candidates,
        key=lambda c: (c.score, c.efficiency_ratio),
        reverse=True,
    )


def _holding_var(ticker: str, risk_summary: Optional[PortfolioRiskSummary]) -> float:
    """Return VaR for a ticker from risk_summary; 0 if not found."""
    if risk_summary is None:
        return 0.0
    for p in risk_summary.risk_profiles:
        if p.ticker == ticker:
            return p.var_5pct
    return 0.0


class OutputFormatterEngine:
    """
    L12 Output Formatter.

    run() — final sort, rank, trim, and summary assembly.
    """

    def run(
        self,
        hedge_output: HedgeOutput,
        pipeline_start_time: float,
        risk_summary: Optional[PortfolioRiskSummary] = None,
    ) -> HedgeOutput:
        t0_fmt = time.perf_counter()

        # ── 1. Sort + trim candidates within each recommendation ──────────────
        for rec in hedge_output.recommendations:
            rec.candidates = _sort_candidates(rec.candidates)[:TOP_N_CANDIDATES_PER_HOLDING]

        # ── 2. Sort recommendations by holding VaR (most at-risk first) ───────
        hedge_output.recommendations = sorted(
            hedge_output.recommendations,
            key=lambda r: _holding_var(r.asset_ticker, risk_summary),
            reverse=True,
        )

        # ── 3. Assign recommendation ranks (1 = highest VaR holding) ──────────
        for rank, rec in enumerate(hedge_output.recommendations, start=1):
            rec.rank = rank

        # ── 4. Aggregate portfolio-level hedge Greeks (top candidate per holding)
        port_delta = port_gamma = port_vega = 0.0
        for rec in hedge_output.recommendations:
            if rec.candidates:
                top = rec.candidates[0]  # already sorted by score desc
                port_delta += top.delta
                port_gamma += top.gamma
                port_vega  += top.vega
        hedge_output.hedge_portfolio_delta = round(port_delta, 4)
        hedge_output.hedge_portfolio_gamma = round(port_gamma, 6)
        hedge_output.hedge_portfolio_vega  = round(port_vega,  4)

        # ── 5. Stamp total run time ────────────────────────────────────────────
        hedge_output.run_time_seconds = round(time.perf_counter() - pipeline_start_time, 3)

        # ── 6. Log summary ─────────────────────────────────────────────────────
        total_candidates = sum(len(r.candidates) for r in hedge_output.recommendations)
        top_scores = [
            r.candidates[0].score for r in hedge_output.recommendations if r.candidates
        ]
        avg_top_score = round(sum(top_scores) / len(top_scores), 1) if top_scores else 0.0

        logger.info(
            "L12 OutputFormatter: %d recommendations, %d candidates, "
            "avg top score=%.1f, total time=%.3fs",
            len(hedge_output.recommendations),
            total_candidates,
            avg_top_score,
            hedge_output.run_time_seconds,
        )

        return hedge_output
