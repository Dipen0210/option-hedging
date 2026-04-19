"""
Layer 4 — Instrument Selection Engine (Step 4 in pipeline)

For each holding in the portfolio:
  1. Route to all applicable asset-class selectors
  2. Each selector returns ranked InstrumentCandidates
  3. Merge and re-rank across asset classes (options first by default)
  4. Cap at MAX_CANDIDATES_PER_HOLDING
  5. Return HedgeOutput with one HedgeRecommendation per holding

Asset classes:
  OPTIONS     ← fully implemented (Phase 1)
  FUTURES     ← stub (Phase 2)
  FORWARDS    ← stub (Phase 2)
  SWAPS       ← stub (Phase 2)
  INVERSE ETFs← stub (Phase 2)
"""
import logging
import time
from typing import List

from backend.models.portfolio_models import PortfolioInput
from backend.models.risk_models import RegimeState, PortfolioRiskSummary
from backend.models.hedge_models import HedgeOutput, HedgeRecommendation, InstrumentCandidate
from backend.instruments import ALL_SELECTORS

logger = logging.getLogger(__name__)

MAX_CANDIDATES_PER_HOLDING = 5


class InstrumentSelectionEngine:

    def __init__(self, selectors=None):
        """
        Args:
            selectors: list of InstrumentSelector instances.
                       Defaults to ALL_SELECTORS (options + stubs).
                       Pass a subset for testing.
        """
        self.selectors = selectors if selectors is not None else ALL_SELECTORS

    def select(
        self,
        portfolio: PortfolioInput,
        regime: RegimeState,
        risk_summary: PortfolioRiskSummary,
    ) -> HedgeOutput:
        """
        Run instrument selection for all holdings.

        Args:
            portfolio:    original PortfolioInput (holdings, constraints)
            regime:       RegimeState from Layer 2
            risk_summary: PortfolioRiskSummary from Layer 3

        Returns:
            HedgeOutput with ranked candidates per holding.
        """
        t0 = time.perf_counter()
        recommendations: List[HedgeRecommendation] = []

        # Iterate all risk profiles (stocks + option positions) directly
        for i, profile in enumerate(risk_summary.risk_profiles):
            ticker = profile.ticker
            logger.info(f"Selecting instruments for {ticker} ({profile.asset_class})...")
            candidates = self._select_for_holding(profile, portfolio, regime)

            if not candidates:
                logger.warning(f"No viable candidates found for {ticker}")

            recommendations.append(HedgeRecommendation(
                rank=i + 1,
                asset_ticker=ticker,
                candidates=candidates,
            ))

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"Instrument selection done in {elapsed}s — "
                    f"{sum(len(r.candidates) for r in recommendations)} total candidates")

        return HedgeOutput(
            portfolio_notional=portfolio.total_notional,
            regime=regime.regime_label,
            is_anomaly=regime.is_anomaly,
            vix_level=regime.vix_level,
            recommendations=recommendations,
            run_time_seconds=elapsed,
        )

    def _select_for_holding(
        self,
        profile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        """
        Run all applicable selectors for a single holding and merge results.

        Direction rule (safety net — also enforced in each selector):
          direct_hedge candidates MUST have delta opposite to the user's position.
          Same-direction same-asset suggestions are never valid hedges.
        """
        all_candidates: List[InstrumentCandidate] = []

        for selector in self.selectors:
            try:
                if not selector.is_applicable(profile, regime):
                    continue
                candidates = selector.find_candidates(profile, portfolio, regime)
                all_candidates.extend(candidates)
            except Exception as e:
                logger.error(
                    f"{selector.__class__.__name__} failed for {profile.ticker}: {e}"
                )

        # Direction validation: for direct hedges, delta sign must be opposite
        user_sign = getattr(profile, "effective_delta_sign", 1)   # default +1 (long stock)
        direction_valid: List[InstrumentCandidate] = []
        for c in all_candidates:
            if c.hedge_category == "direct_hedge" and c.delta != 0:
                if c.delta * user_sign >= 0:
                    # Same direction as user's position — not a valid hedge
                    logger.warning(
                        "L4: removed %s %s for %s — same-direction direct hedge "
                        "(candidate delta=%.3f, user sign=%d)",
                        c.ticker, c.strategy, profile.ticker, c.delta, user_sign,
                    )
                    continue
            direction_valid.append(c)

        # Sort all candidates by score descending
        direction_valid.sort(key=lambda c: c.score, reverse=True)

        # Deduplicate: keep best candidate per (strategy, ticker) pair
        seen = set()
        unique: List[InstrumentCandidate] = []
        for c in direction_valid:
            key = (c.strategy, c.ticker, c.strike)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique[:MAX_CANDIDATES_PER_HOLDING]
