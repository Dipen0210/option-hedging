"""
Layer 7 — Position Sizer Engine

Dispatches per-candidate position sizing to asset-class-specific sizers concurrently.
Runs after L6 (prices are already filled in on candidates).

Input:  HedgeOutput (candidates with price + Greeks from L6)
        portfolio_context dict (notional, beta, budgets, etc.)
Output: HedgeOutput with n_contracts, total_cost, max_protection,
        and partial_hedge_options filled in for every candidate.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.core.sizing import SIZER_REGISTRY
from backend.core.sizing.base_sizer import SizingContext, SizingResult
from backend.core.pricing.base_pricer import PriceResult

logger = logging.getLogger(__name__)

_MAX_WORKERS = 16


def executable_premium(candidate: InstrumentCandidate) -> float:
    """
    The per-share price the user would actually transact at.

    Positive is a debit, negative a credit. When the chain gave a two-sided
    quote this is the market mid, because that is what the trade costs; only
    without one does the theoretical price stand in, and `premium_source`
    already tells the user which they are looking at.

    Sizing and cost must both come from this number. The premium was
    previously recovered as `candidate.total_cost / 100`, which after L6 was
    the model price for a single contract — so a candidate displaying a live
    bid/ask was sized and costed off the model, and the budget the user set
    was enforced against a price that did not exist in the market.
    """
    if candidate.premium_source == "market_mid" and candidate.market_premium:
        return candidate.market_premium
    return candidate.model_premium


def _build_price_result(candidate: InstrumentCandidate) -> PriceResult:
    """Reconstruct a PriceResult from a priced candidate (post-L6)."""
    return PriceResult(
        price=executable_premium(candidate),
        delta=candidate.delta,
        gamma=candidate.gamma,
        theta=candidate.theta,
        vega=candidate.vega,
        rho=candidate.rho,
        lambda_leverage=candidate.lambda_leverage,
        extended=dict(candidate.extended_metrics),
    )


def _size_one(
    candidate: InstrumentCandidate,
    price_result: PriceResult,
    ctx: SizingContext,
) -> tuple[InstrumentCandidate, SizingResult]:
    asset_class = candidate.asset_class
    sizer = SIZER_REGISTRY.get(asset_class)
    if sizer is None:
        logger.warning("No sizer for asset_class=%s ticker=%s", asset_class, candidate.ticker)
        return candidate, SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)

    try:
        result = sizer.size(candidate, price_result, ctx)
    except Exception as exc:
        logger.error("Sizing failed for %s: %s", candidate.ticker, exc, exc_info=True)
        result = SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)

    return candidate, result


def _apply_sizing_result(candidate: InstrumentCandidate, r: SizingResult) -> None:
    candidate.n_contracts = r.n_contracts
    candidate.total_cost = round(r.total_cost, 2)
    # market_total_cost is quoted alongside total_cost in the API response, so
    # it has to follow the contract count L7 settled on. Leaving it at L4's
    # pre-sizing figure made the two disagree on the same screen.
    if candidate.premium_source == "market_mid":
        candidate.market_total_cost = candidate.total_cost
    candidate.max_protection = round(r.notional_hedged, 2)
    candidate.partial_hedge_options = r.partial_hedge_options
    candidate.extended_metrics["hedge_effectiveness"] = round(r.hedge_effectiveness, 4)


class PositionSizerEngine:
    """
    L7 Position Sizer Engine.

    build_sizing_context() — helper to build a per-holding SizingContext
    run()                  — size all candidates concurrently
    """

    @staticmethod
    def build_sizing_context(
        portfolio_notional: float,
        holding_notional: float,
        beta: float = 1.0,
        hedge_ratio: float = 1.0,
        max_premium_pct: float = 0.02,
        correlation: float = 0.85,
        spot_vol: Optional[float] = None,
        futures_vol: Optional[float] = None,
        portfolio_dv01: float = 0.0,
        portfolio_cs01: float = 0.0,
        fx_exposure: float = 0.0,
        extended: Dict[str, Any] | None = None,
    ) -> SizingContext:
        return SizingContext(
            portfolio_notional=portfolio_notional,
            holding_notional=holding_notional,
            beta=beta,
            hedge_ratio=hedge_ratio,
            max_premium_pct=max_premium_pct,
            correlation=correlation,
            spot_vol=spot_vol,
            futures_vol=futures_vol,
            portfolio_dv01=portfolio_dv01,
            portfolio_cs01=portfolio_cs01,
            fx_exposure=fx_exposure,
            extended=extended or {},
        )

    def run(
        self,
        hedge_output: HedgeOutput,
        sizing_contexts: Dict[str, SizingContext],
    ) -> HedgeOutput:
        """
        Size all candidates concurrently.

        sizing_contexts: {asset_ticker → SizingContext}
        Each holding may have different beta / notional / budgets.
        Falls back to a default context if no ticker-specific context found.
        """
        t0 = time.perf_counter()

        tasks: List[tuple[int, int, InstrumentCandidate]] = []
        for ri, rec in enumerate(hedge_output.recommendations):
            for ci, cand in enumerate(rec.candidates):
                tasks.append((ri, ci, cand))

        logger.info("L7 PositionSizerEngine: sizing %d candidates", len(tasks))

        futures_map = {}
        unsizable: List[tuple[int, int]] = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for ri, ci, cand in tasks:
                # Pick context: by asset ticker being hedged (parent holding)
                rec = hedge_output.recommendations[ri]
                ctx = sizing_contexts.get(rec.asset_ticker)
                if ctx is None:
                    # No context means nobody measured this holding's exposure.
                    # The old fallback built one with
                    # `holding_notional=hedge_output.portfolio_notional`, so a
                    # single holding was sized as if it were the entire
                    # portfolio — the larger the portfolio, the larger the
                    # over-hedge, and the result looked deliberate.
                    unsizable.append((ri, ci))
                    continue

                pr = _build_price_result(cand)
                fut = pool.submit(_size_one, cand, pr, ctx)
                futures_map[fut] = (ri, ci)

            for fut in as_completed(futures_map):
                ri, ci = futures_map[fut]
                try:
                    candidate, result = fut.result()
                    _apply_sizing_result(candidate, result)
                    logger.debug(
                        "Sized %s → n=%d cost=$%.0f",
                        candidate.ticker, result.n_contracts, result.total_cost,
                    )
                except Exception as exc:
                    logger.error("Future failed for rec[%d] cand[%d]: %s", ri, ci, exc)

        # Drop in reverse index order so earlier indices stay valid.
        for ri, ci in sorted(unsizable, reverse=True):
            rec = hedge_output.recommendations[ri]
            dropped = rec.candidates.pop(ci)
            logger.warning(
                "L7: dropping %s %s for %s — no sizing context, so the "
                "holding's exposure was never measured",
                dropped.strategy, dropped.ticker, rec.asset_ticker,
            )
        if unsizable:
            tickers = sorted({
                hedge_output.recommendations[ri].asset_ticker
                for ri, _ in unsizable
            })
            hedge_output.warnings.append(
                f"{', '.join(tickers)}: candidates were withdrawn because the "
                "holding's exposure could not be measured, so no position size "
                "could be derived for them."
            )

        elapsed = time.perf_counter() - t0
        logger.info("L7 PositionSizerEngine done in %.3fs", elapsed)
        return hedge_output
