"""
Credit instrument pricer — Hazard-rate model (L6).

CDS pricing via constant hazard rate:
  CDS spread ≈ (1 - R) × λ   where λ = hazard rate, R = recovery rate

For CDS protection leg:
  Value = notional × (1 - R) × (1 - exp(-λT)) × exp(-rT)

Extended metrics:
  cs01          — dollar value of 1bp spread move
  dv01          — interest-rate DV01
  hazard_rate   — implied λ from spread
  spread_bps    — CDS spread in basis points
  recovery      — assumed recovery rate
"""
from __future__ import annotations

import math
from backend.core.pricing.base_pricer import BaseAssetPricer, PriceResult, PricingContext
from backend.core.pricing.base_pricer import years_to_expiry

# Default recovery rates by sector
_RECOVERY = {
    "IG": 0.40,    # Investment grade
    "HY": 0.35,    # High yield
    "EM": 0.25,    # Emerging markets
    "FIN": 0.45,   # Financials
}

# Default 5yr CDS spreads by ticker (bps) — fallback estimates
_DEFAULT_SPREAD_BPS = {
    "HYG": 350.0,
    "JNK": 400.0,
    "LQD": 80.0,
    "BKLN": 300.0,
    "EMB": 250.0,
}


def _cds_price(
    notional: float,
    spread_bps: float,
    r: float,
    T: float,
    recovery: float = 0.40,
) -> dict:
    """Price a CDS protection leg (buyer of protection perspective)."""
    spread = spread_bps / 10_000
    lgd = 1.0 - recovery              # loss given default

    if lgd <= 0 or spread <= 0:
        return {"price": 0.0, "hazard_rate": 0.0, "cs01": 0.0, "dv01": 0.0}

    # Constant hazard rate: λ ≈ spread / LGD
    hazard = spread / lgd

    # Protection leg PV (simplified, flat hazard + flat discount)
    pv_protection = notional * lgd * (1.0 - math.exp(-(hazard + r) * T)) \
                    / (hazard + r) * hazard

    # CS01 = change in PV for +1bp spread
    spread_up = (spread_bps + 1) / 10_000
    hazard_up = spread_up / lgd
    pv_up = notional * lgd * (1.0 - math.exp(-(hazard_up + r) * T)) \
            / (hazard_up + r) * hazard_up
    cs01 = pv_up - pv_protection

    # DV01 (sensitivity to parallel shift in discount rate)
    r_up = r + 0.0001
    pv_r_up = notional * lgd * (1.0 - math.exp(-(hazard + r_up) * T)) \
              / (hazard + r_up) * hazard
    dv01 = abs(pv_r_up - pv_protection)

    return {
        "price": round(pv_protection, 4),
        "hazard_rate": round(hazard, 6),
        "cs01": round(cs01, 6),
        "dv01": round(dv01, 6),
    }


class CreditPricer(BaseAssetPricer):
    asset_class = "credit"

    def price(self, candidate, ctx: PricingContext) -> PriceResult:
        ticker = candidate.ticker.upper()
        spread_bps = ctx.credit_spreads.get(ticker) \
                     or _DEFAULT_SPREAD_BPS.get(ticker, 200.0)
        T = years_to_expiry(candidate.expiry_date) if candidate.expiry_date else 5.0
        r = ctx.risk_free_rate
        notional = 10_000.0             # standard 1 contract = $10k notional
        recovery = _RECOVERY.get("HY" if spread_bps > 200 else "IG", 0.40)

        result = _cds_price(notional, spread_bps, r, T, recovery)

        return PriceResult(
            price=result["price"],
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            lambda_leverage=0.0,
            extended={
                "cs01": result["cs01"],
                "dv01": result["dv01"],
                "hazard_rate": result["hazard_rate"],
                "spread_bps": spread_bps,
                "recovery": recovery,
            },
            model_used="Hazard_Rate",
        )
