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
from typing import List, Tuple

from backend.models.portfolio_models import PortfolioInput
from backend.models.risk_models import RegimeState, PortfolioRiskSummary
from backend.models.hedge_models import HedgeOutput, HedgeRecommendation, InstrumentCandidate
from backend.instruments import ALL_SELECTORS
from backend.instruments.options.selector import is_executably_quoted
from backend.data import data_quality as dq

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
        warnings: List[str] = []

        # Iterate all risk profiles (stocks + option positions) directly
        for i, profile in enumerate(risk_summary.risk_profiles):
            ticker = profile.ticker

            # A profile L3 could not measure carries placeholder risk figures,
            # not risk. Sizing a hedge off them produces a contract count, a
            # premium and a risk-reduction estimate that look exactly like the
            # real ones and are derived from nothing, so the holding is
            # reported as unhedgeable instead.
            if not getattr(profile, "data_available", True):
                reason = getattr(profile, "unavailable_reason", "") or "risk analysis failed"
                logger.warning(
                    "L4: no recommendation for %s — risk profile unavailable (%s)",
                    ticker, reason,
                )
                warnings.append(
                    f"{ticker}: no hedge could be recommended because its risk could "
                    f"not be analysed ({reason}). This holding is unhedged and its "
                    f"risk is not included in the portfolio figures."
                )
                recommendations.append(HedgeRecommendation(
                    rank=i + 1,
                    asset_ticker=ticker,
                    candidates=[],
                ))
                continue

            logger.info(f"Selecting instruments for {ticker} ({profile.asset_class})...")
            candidates, evaluated, unbuilt, failed = self._select_for_holding(
                profile, portfolio, regime
            )

            # A selector that raised is an error, not an absence, and it is
            # disclosed even when another instrument class did produce a
            # hedge — otherwise a partial run looks like a complete one. The
            # exception text itself stays in the log; responses carry none.
            if failed:
                warnings.append(
                    f"{ticker}: hedge selection failed for "
                    f"{', '.join(failed)}. The error was logged; any hedge "
                    f"shown for this holding comes from the other instrument "
                    f"types only."
                )

            if not candidates and not failed:
                logger.warning(f"No viable candidates found for {ticker}")
                # Several different facts produce an empty list, and the user
                # is owed the one that is true. Naming no cause is better
                # than naming the wrong one.
                if evaluated:
                    warnings.append(
                        f"{ticker}: no hedge candidate was found among the "
                        f"instrument types evaluated ({', '.join(evaluated)}). "
                        f"This holding is shown unhedged."
                    )
                elif unbuilt:
                    warnings.append(
                        f"{ticker}: hedging with {', '.join(unbuilt)} is not "
                        f"implemented in this version — no instrument was "
                        f"evaluated. This holding is shown unhedged."
                    )
                else:
                    warnings.append(
                        f"{ticker}: no instrument type in this run applies to "
                        f"this {profile.asset_class} holding. It is shown "
                        f"unhedged."
                    )

            recommendations.append(HedgeRecommendation(
                rank=i + 1,
                asset_ticker=ticker,
                candidates=candidates,
            ))

        # Whether an unquoted hedge may still be offered depends on something
        # no single holding can see. Decided here, across the whole run.
        self._refuse_unquotable_hedges(recommendations, warnings)

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"Instrument selection done in {elapsed}s — "
                    f"{sum(len(r.candidates) for r in recommendations)} total candidates")

        return HedgeOutput(
            portfolio_notional=portfolio.total_notional,
            regime=regime.regime_label,
            is_anomaly=regime.is_anomaly,
            vix_level=regime.vix_level,
            # The label alone hides whether the GMM produced it or the VIX
            # fallback did; L2 knows, and the response has to say so.
            regime_source=regime.regime_source,
            regime_degraded=regime.degraded,
            regime_degradation_reason=regime.degradation_reason,
            recommendations=recommendations,
            run_time_seconds=elapsed,
            warnings=warnings,
        )

    @staticmethod
    def _refuse_unquotable_hedges(
        recommendations: List[HedgeRecommendation],
        warnings: List[str],
    ) -> None:
        """
        Withhold a hedge for a holding whose contracts have no tradeable
        market, while the market is demonstrably open.

        The selector, finding nothing quotable for a holding, keeps the
        candidates and discloses that their premiums are theoretical. That is
        right for one reason and wrong for the other, and until now the two
        were indistinguishable:

          * The provider publishes no quotes at all right now — outside
            market hours, on a holiday, or during an outage. The contracts
            are real and listed; the user is researching a hedge they would
            place when the market opens. Withholding everything would make
            the product useless for two thirds of the day, so a disclosed
            theoretical premium is the honest and useful answer.

          * The market is open and quoting, and *this* instrument still has
            no two-sided market. Then there is no hedge to recommend. Ranking
            a model premium here invites the user to work out a position in a
            contract nobody will fill, which is the "forced recommendation"
            the brief prohibits.

        Telling them apart needs no market calendar, which is the point: a
        calendar would have to be maintained and would refuse hedges wrongly
        on every holiday it missed. The run answers the question itself. If
        any instrument anywhere in this run carries an executable quote, the
        venue is open and quoting, so a holding with none has no market. If
        nothing is quoted anywhere, the provider is silent and nothing can be
        concluded about any single contract.

        The asymmetry is deliberate. A missed refusal leaves a disclosed
        theoretical premium on screen; a wrong refusal silently removes a
        hedge the user could have placed. So the refusal fires only on
        positive evidence that quoting works right now.
        """
        quoted_anywhere = any(
            is_executably_quoted(c)
            for rec in recommendations
            for c in rec.candidates
        )
        if not quoted_anywhere:
            logger.info(
                "L4: no executable quote anywhere in this run — treating the "
                "venue as not quoting, so theoretical premiums stand with "
                "their disclosure rather than being refused",
            )
            return

        for rec in recommendations:
            if not rec.candidates:
                continue
            if any(is_executably_quoted(c) for c in rec.candidates):
                continue

            n = len(rec.candidates)
            logger.warning(
                "L4: withholding %d candidate(s) for %s — none has an "
                "executable two-sided quote while other instruments in this "
                "run are quoting normally",
                n, rec.asset_ticker,
            )
            rec.candidates = []
            warnings.append(
                f"{rec.asset_ticker}: no hedge is recommended. None of the "
                f"{n} contract(s) considered has a tradeable two-sided quote, "
                f"while other instruments priced in this same run are quoting "
                f"normally — so this is an absent market for these contracts, "
                f"not a closed one. A premium could only be a model estimate, "
                f"and a position sized from it could not be filled. The "
                f"holding is shown unhedged."
            )
            dq.record(
                f"no_market:{rec.asset_ticker}", dq.SOURCE_UNAVAILABLE,
                detail=(
                    f"{rec.asset_ticker}: hedge withheld — no executable "
                    f"two-sided quote on any candidate contract while the "
                    f"venue is quoting"
                ),
            )

    def _select_for_holding(
        self,
        profile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> Tuple[List[InstrumentCandidate], List[str], List[str], List[str]]:
        """
        Run all applicable selectors for a single holding and merge results.

        Returns the candidates, then the labels of the selectors that
        evaluated this holding, those that were applicable but unbuilt, and
        those that raised. The caller needs the last three to explain an
        empty list without inventing a reason for it, and to disclose a
        selector error instead of swallowing it.

        Direction rule (safety net — also enforced in each selector):
          direct_hedge candidates MUST have delta opposite to the user's position.
          Same-direction same-asset suggestions are never valid hedges.
        """
        all_candidates: List[InstrumentCandidate] = []
        evaluated: List[str] = []
        unbuilt: List[str] = []
        failed: List[str] = []

        for selector in self.selectors:
            # getattr, not attribute access: a selector injected by a caller
            # or a test need only satisfy the two-method protocol, and reading
            # `.label` inside the try below would turn a missing attribute
            # into a reported selector failure.
            label = getattr(selector, "label", "instrument")
            try:
                if not selector.is_applicable(profile, regime):
                    continue
                if not getattr(selector, "implemented", True):
                    unbuilt.append(label)
                    continue
                candidates = selector.find_candidates(profile, portfolio, regime)
            except Exception as e:
                failed.append(label)
                logger.error(
                    f"{selector.__class__.__name__} failed for {profile.ticker}: {e}"
                )
                continue
            evaluated.append(label)
            all_candidates.extend(candidates)

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

        return unique[:MAX_CANDIDATES_PER_HOLDING], evaluated, unbuilt, failed
