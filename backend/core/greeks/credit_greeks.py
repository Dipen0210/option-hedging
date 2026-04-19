"""
Credit Greeks engine — CS01 / DV01 only (L8).

No BSM Greeks.  Credit risk is measured as:
  cs01         — $ change per 1bp spread widening × n_contracts
  dv01         — $ change per 1bp rate move × n_contracts
  hazard_rate  — implied default intensity
  spread_bps   — current CDS spread
  recovery     — assumed recovery rate
"""
from __future__ import annotations

from backend.core.greeks.base_greeks import BaseGreeksEngine, GreeksResult


class CreditGreeksEngine(BaseGreeksEngine):
    asset_class = "credit"

    def compute(self, candidate, price_result, ctx, n_contracts: int) -> GreeksResult:
        ext = price_result.extended

        cs01_portfolio = ext.get("cs01", 0.0) * n_contracts
        dv01_portfolio = ext.get("dv01", 0.0) * n_contracts

        return GreeksResult(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            lambda_leverage=0.0,
            extended={
                "cs01": round(cs01_portfolio, 4),
                "dv01": round(dv01_portfolio, 4),
                "hazard_rate": ext.get("hazard_rate", 0.0),
                "spread_bps": ext.get("spread_bps", 0.0),
                "recovery": ext.get("recovery", 0.40),
                "n_contracts": n_contracts,
            },
        )
