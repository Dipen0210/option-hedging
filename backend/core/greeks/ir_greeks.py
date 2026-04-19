"""
Interest-rate Greeks engine — Duration / DV01 / Convexity (L8).

No BSM Greeks. IR risk is measured as:
  dv01           — $ change per 1bp parallel shift (scaled by n_contracts)
  duration       — Macaulay duration (years)
  mod_duration   — Modified duration
  convexity      — Convexity (years²)
  yield_level    — current yield

BSM delta/gamma/theta/vega are all set to 0.
"""
from __future__ import annotations

from backend.core.greeks.base_greeks import BaseGreeksEngine, GreeksResult


class IRGreeksEngine(BaseGreeksEngine):
    asset_class = "bond"

    def compute(self, candidate, price_result, ctx, n_contracts: int) -> GreeksResult:
        ext = price_result.extended
        face_per_contract = 1000   # $1000 face value per contract (simplified)
        scale = n_contracts * face_per_contract / 100   # bond price is per $100 face

        dv01_portfolio = ext.get("dv01", 0.0) * n_contracts
        duration = ext.get("duration", 0.0)
        mod_dur  = ext.get("mod_duration", 0.0)
        convexity = ext.get("convexity", 0.0)
        yield_lvl = ext.get("yield_level", 0.0)

        return GreeksResult(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            lambda_leverage=0.0,
            extended={
                "dv01": round(dv01_portfolio, 4),
                "duration": duration,
                "mod_duration": mod_dur,
                "convexity": convexity,
                "yield_level": yield_lvl,
                "n_contracts": n_contracts,
            },
        )
