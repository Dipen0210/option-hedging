"""
Commodity Greeks engine — Modified Greeks (L8).

Output: Δ, Γ, V  (standard) + roll_yield_daily replaces Θ.
Rho is not meaningful for commodity futures (no interest-rate carry in Black-76).

extended:
  roll_yield_daily   — daily carry cost from rolling futures
  theta_effective    — theta + roll_yield_daily (effective daily time decay)
"""
from __future__ import annotations

from backend.core.greeks.base_greeks import BaseGreeksEngine, GreeksResult

# Commodity futures contract multipliers
_MULTIPLIER = {
    "GC": 100,    "SI": 5000,  "CL": 1000,
    "NG": 10000,  "ZC": 5000,  "ZS": 5000,
    "GLD": 100,   "USO": 100,
}
_DEFAULT_MULT = 100


class CommodityGreeksEngine(BaseGreeksEngine):
    asset_class = "commodity"

    def compute(self, candidate, price_result, ctx, n_contracts: int) -> GreeksResult:
        ticker = candidate.ticker.upper()
        mult = _MULTIPLIER.get(ticker, _DEFAULT_MULT) * n_contracts

        delta = price_result.delta * mult
        gamma = price_result.gamma * mult
        vega  = price_result.vega  * mult

        roll_daily = price_result.extended.get("roll_yield_daily", 0.0)
        theta_bsm = price_result.theta * mult
        theta_effective = theta_bsm + roll_daily * mult

        return GreeksResult(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta_effective, 4),   # includes roll cost
            vega=round(vega, 4),
            rho=0.0,
            lambda_leverage=round(price_result.lambda_leverage, 4),
            extended={
                "roll_yield_daily": round(roll_daily * mult, 4),
                "theta_bsm": round(theta_bsm, 4),
                "theta_effective": round(theta_effective, 4),
            },
        )
