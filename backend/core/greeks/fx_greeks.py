"""
FX Greeks engine — Garman-Kohlhagen Greeks scaled to notional (L8).

Output: Δ, Γ, Θ, V + TWO Rhos (domestic and foreign).
Lambda is meaningful: Δ × S / V = leverage per unit of premium paid.

extended:
  rho_domestic   — sensitivity to domestic rate × notional
  rho_foreign    — sensitivity to foreign rate × notional
"""
from __future__ import annotations

from backend.core.greeks.base_greeks import BaseGreeksEngine, GreeksResult


class FXGreeksEngine(BaseGreeksEngine):
    asset_class = "fx"

    def compute(self, candidate, price_result, ctx, n_contracts: int) -> GreeksResult:
        # For FX, n_contracts represents notional in thousands
        notional_k = n_contracts
        scale = notional_k * 1000   # back to full notional

        # GK Greeks are per unit of notional (e.g., per $1 of FX exposure)
        delta = price_result.delta * scale
        gamma = price_result.gamma * scale
        theta = price_result.theta * scale
        vega  = price_result.vega  * scale

        rho_d = price_result.extended.get("rho_domestic", price_result.rho) * scale
        rho_f = price_result.extended.get("rho_foreign", 0.0) * scale

        return GreeksResult(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho_d, 4),           # primary = domestic
            lambda_leverage=round(price_result.lambda_leverage, 4),
            extended={
                "rho_domestic": round(rho_d, 4),
                "rho_foreign": round(rho_f, 4),
                "r_domestic": price_result.extended.get("r_domestic", 0.0),
                "r_foreign": price_result.extended.get("r_foreign", 0.0),
            },
        )
