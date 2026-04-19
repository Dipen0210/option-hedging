"""
Layer 6 — Pricing Engine

Dispatches per-candidate pricing to asset-class-specific pricers concurrently.
All candidates across all holdings run in a ThreadPoolExecutor simultaneously.

Input:  HedgeOutput (candidates from L5, fully populated except price/Greeks)
Output: HedgeOutput with price, delta, gamma, theta, vega, rho,
        lambda_leverage, and extended_metrics filled in for every candidate.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.core.pricing import PRICER_REGISTRY
from backend.core.pricing.base_pricer import PricingContext, PriceResult

logger = logging.getLogger(__name__)

_MAX_WORKERS = 16   # concurrent pricer threads


def _price_one(
    candidate: InstrumentCandidate,
    ctx: PricingContext,
) -> tuple[InstrumentCandidate, PriceResult]:
    """Price a single candidate and return (candidate, result)."""
    asset_class = candidate.asset_class
    pricer = PRICER_REGISTRY.get(asset_class)
    if pricer is None:
        logger.warning("No pricer for asset_class=%s ticker=%s", asset_class, candidate.ticker)
        return candidate, PriceResult(price=0.0, model_used="no_pricer")

    try:
        result = pricer.price(candidate, ctx)
    except Exception as exc:
        logger.error("Pricing failed for %s: %s", candidate.ticker, exc, exc_info=True)
        result = PriceResult(price=0.0, model_used="error")

    return candidate, result


def _apply_price_result(candidate: InstrumentCandidate, r: PriceResult) -> None:
    """Write PriceResult fields back into the candidate in-place."""
    candidate.total_cost = round(r.price * 100, 2)  # per-contract cost
    candidate.delta = r.delta
    candidate.gamma = r.gamma
    candidate.theta = r.theta
    candidate.vega = r.vega
    candidate.rho = r.rho
    candidate.lambda_leverage = r.lambda_leverage
    # Merge extended metrics
    if r.extended:
        candidate.extended_metrics.update(r.extended)
    candidate.extended_metrics["model_used"] = r.model_used


class PricingEngine:
    """
    L6 Pricing Engine.

    build_context()  — helper to assemble PricingContext from market data dicts
    run()            — price all candidates concurrently, return updated HedgeOutput
    """

    @staticmethod
    def build_context(
        spot_prices: dict,
        risk_free_rate: float = 0.053,
        regime_vol: float = 0.20,
        vix_level: float = 20.0,
        vol_overrides: dict | None = None,
        dividend_yields: dict | None = None,
        yield_curve: dict | None = None,
        fx_rates: dict | None = None,
        futures_prices: dict | None = None,
        credit_spreads: dict | None = None,
    ) -> PricingContext:
        return PricingContext(
            spot_prices=spot_prices,
            risk_free_rate=risk_free_rate,
            regime_vol=regime_vol,
            vix_level=vix_level,
            vol_overrides=vol_overrides or {},
            dividend_yields=dividend_yields or {},
            yield_curve=yield_curve or {},
            fx_rates=fx_rates or {},
            futures_prices=futures_prices or {},
            credit_spreads=credit_spreads or {},
        )

    def run(
        self,
        hedge_output: HedgeOutput,
        ctx: PricingContext,
    ) -> HedgeOutput:
        """
        Price all candidates concurrently.
        Returns the same HedgeOutput object with candidates updated in-place.
        """
        t0 = time.perf_counter()

        # Collect all (rec_idx, cand_idx, candidate) triples
        tasks: List[tuple[int, int, InstrumentCandidate]] = []
        for ri, rec in enumerate(hedge_output.recommendations):
            for ci, cand in enumerate(rec.candidates):
                tasks.append((ri, ci, cand))

        logger.info("L6 PricingEngine: pricing %d candidates across %d holdings",
                    len(tasks), len(hedge_output.recommendations))

        futures_map = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for ri, ci, cand in tasks:
                fut = pool.submit(_price_one, cand, ctx)
                futures_map[fut] = (ri, ci)

            for fut in as_completed(futures_map):
                ri, ci = futures_map[fut]
                try:
                    candidate, result = fut.result()
                    _apply_price_result(candidate, result)
                    logger.debug(
                        "Priced %s [%s] → price=%.4f model=%s",
                        candidate.ticker, candidate.asset_class,
                        result.price, result.model_used,
                    )
                except Exception as exc:
                    logger.error("Future failed for rec[%d] cand[%d]: %s", ri, ci, exc)

        elapsed = time.perf_counter() - t0
        logger.info("L6 PricingEngine done in %.3fs", elapsed)
        return hedge_output
