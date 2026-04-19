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
from typing import Any, Dict, List

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.core.sizing import SIZER_REGISTRY
from backend.core.sizing.base_sizer import SizingContext, SizingResult
from backend.core.pricing.base_pricer import PriceResult

logger = logging.getLogger(__name__)

_MAX_WORKERS = 16


def _build_price_result(candidate: InstrumentCandidate) -> PriceResult:
    """Reconstruct a PriceResult from a priced candidate (post-L6)."""
    return PriceResult(
        price=candidate.total_cost / 100 if candidate.total_cost > 0 else 0.0,
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
        spot_vol: float = 0.20,
        futures_vol: float = 0.22,
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
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for ri, ci, cand in tasks:
                # Pick context: by asset ticker being hedged (parent holding)
                rec = hedge_output.recommendations[ri]
                ctx = sizing_contexts.get(rec.asset_ticker)
                if ctx is None:
                    # fallback minimal context
                    ctx = SizingContext(
                        portfolio_notional=hedge_output.portfolio_notional,
                        holding_notional=hedge_output.portfolio_notional,
                    )
                    # inject spot price for equity sizer
                    ctx.extended["spot_price"] = cand.extended_metrics.get("spot_price", 0.0)

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

        elapsed = time.perf_counter() - t0
        logger.info("L7 PositionSizerEngine done in %.3fs", elapsed)
        return hedge_output
