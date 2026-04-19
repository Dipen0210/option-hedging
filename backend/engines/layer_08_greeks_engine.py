"""
Layer 8 — Greeks Engine

Computes portfolio-level Greeks for every candidate concurrently.
Runs after L7 (n_contracts is filled in).

Output per candidate:
  Equity    → Δ, Γ, Θ, V, Λ  (BSM, scaled × n × 100)
  Commodity → Δ, Γ, V + theta_effective (roll-adjusted)
  IR        → DV01, Duration, Convexity  (no BSM Greeks)
  FX        → Δ, Γ, Θ, V + rho_domestic + rho_foreign
  Credit    → CS01, DV01  (no BSM Greeks)

All results are written into candidate.delta / .gamma / .theta / .vega / .rho
/ .lambda_leverage and candidate.extended_metrics.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.core.greeks import GREEKS_REGISTRY
from backend.core.greeks.base_greeks import GreeksResult
from backend.core.pricing.base_pricer import PricingContext, PriceResult

logger = logging.getLogger(__name__)

_MAX_WORKERS = 16


def _build_price_result(candidate: InstrumentCandidate) -> PriceResult:
    return PriceResult(
        price=candidate.total_cost / max(candidate.n_contracts, 1) / 100,
        delta=candidate.delta,
        gamma=candidate.gamma,
        theta=candidate.theta,
        vega=candidate.vega,
        rho=candidate.rho,
        lambda_leverage=candidate.lambda_leverage,
        extended=dict(candidate.extended_metrics),
    )


def _greeks_one(
    candidate: InstrumentCandidate,
    price_result: PriceResult,
    ctx: PricingContext,
) -> tuple[InstrumentCandidate, GreeksResult]:
    asset_class = candidate.asset_class
    engine = GREEKS_REGISTRY.get(asset_class)
    if engine is None:
        logger.warning("No greeks engine for asset_class=%s", asset_class)
        return candidate, GreeksResult()

    try:
        result = engine.compute(candidate, price_result, ctx, candidate.n_contracts)
    except Exception as exc:
        logger.error("Greeks failed for %s: %s", candidate.ticker, exc, exc_info=True)
        result = GreeksResult()

    return candidate, result


def _apply_greeks_result(candidate: InstrumentCandidate, g: GreeksResult) -> None:
    candidate.delta = g.delta
    candidate.gamma = g.gamma
    candidate.theta = g.theta
    candidate.vega = g.vega
    candidate.rho = g.rho
    candidate.lambda_leverage = g.lambda_leverage
    if g.extended:
        candidate.extended_metrics.update(g.extended)


class GreeksEngine:
    """
    L8 Greeks Engine.

    run() — compute portfolio-level Greeks for all candidates concurrently.
    """

    def run(
        self,
        hedge_output: HedgeOutput,
        ctx: PricingContext,
    ) -> HedgeOutput:
        t0 = time.perf_counter()

        tasks: List[tuple[int, int, InstrumentCandidate]] = []
        for ri, rec in enumerate(hedge_output.recommendations):
            for ci, cand in enumerate(rec.candidates):
                tasks.append((ri, ci, cand))

        logger.info("L8 GreeksEngine: computing Greeks for %d candidates", len(tasks))

        futures_map = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for ri, ci, cand in tasks:
                pr = _build_price_result(cand)
                fut = pool.submit(_greeks_one, cand, pr, ctx)
                futures_map[fut] = (ri, ci)

            for fut in as_completed(futures_map):
                ri, ci = futures_map[fut]
                try:
                    candidate, result = fut.result()
                    _apply_greeks_result(candidate, result)
                    logger.debug(
                        "Greeks %s [%s] → Δ=%.4f Γ=%.6f Θ=%.4f V=%.4f",
                        candidate.ticker, candidate.asset_class,
                        result.delta, result.gamma, result.theta, result.vega,
                    )
                except Exception as exc:
                    logger.error("Future failed for rec[%d] cand[%d]: %s", ri, ci, exc)

        elapsed = time.perf_counter() - t0
        logger.info("L8 GreeksEngine done in %.3fs", elapsed)
        return hedge_output
