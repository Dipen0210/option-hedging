"""
Equity Greeks engine — Full BSM Greeks scaled to portfolio level (L8).

Output: Δ, Γ, Θ, V, Λ (all 5) × n_contracts × 100 shares/contract.
"""
from __future__ import annotations

from backend.core.greeks.base_greeks import BaseGreeksEngine, GreeksResult


class EquityGreeksEngine(BaseGreeksEngine):
    asset_class = "equity"

    def compute(self, candidate, price_result, ctx, n_contracts: int) -> GreeksResult:
        mult = n_contracts * 100   # 100 shares per equity option contract

        delta = price_result.delta * mult
        gamma = price_result.gamma * mult
        theta = price_result.theta * mult
        vega  = price_result.vega  * mult
        rho   = price_result.rho   * mult

        # Lambda is per-contract (leverage ratio), not scaled by n
        lambda_lev = price_result.lambda_leverage

        return GreeksResult(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
            lambda_leverage=round(lambda_lev, 4),
        )
