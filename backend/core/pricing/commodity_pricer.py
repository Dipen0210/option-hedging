"""
Commodity option pricer — Black-76 (L6).

Commodity options are written on futures contracts, not spot.
Uses Black-76: replaces spot S with futures price F, no cost-of-carry.

Extended metrics stored:
  roll_yield_daily  — estimated daily carry cost from roll
"""
from __future__ import annotations

from backend.core.pricing.base_pricer import BaseAssetPricer, PriceResult, PricingContext
from backend.instruments.options.black76_pricer import black76_greeks
from backend.core.pricing.base_pricer import years_to_expiry

# Approximate annualised roll yield by commodity (negative = cost to carry)
_ROLL_YIELD_ANNUAL = {
    "GC":  -0.002,   # Gold
    "CL":  -0.025,   # Crude Oil
    "NG":  -0.060,   # Nat Gas (high contango)
    "SI":  -0.003,   # Silver
    "ZC":  -0.015,   # Corn
    "ZS":  -0.012,   # Soy
    "GLD": -0.002,   # Gold ETF proxy
    "USO": -0.025,   # Oil ETF proxy
}


class CommodityPricer(BaseAssetPricer):
    asset_class = "commodity"

    def price(self, candidate, ctx: PricingContext) -> PriceResult:
        ticker = candidate.ticker.upper()
        F = ctx.futures_for(ticker)        # futures price
        K = candidate.strike or F
        T = years_to_expiry(candidate.expiry_date) if candidate.expiry_date else 0.0
        r = ctx.risk_free_rate
        sigma = ctx.vol_for(ticker)
        opt_type = (candidate.option_type or "put").lower()

        if F <= 0 or K <= 0:
            return PriceResult(price=0.0, model_used="commodity_skip")

        g = black76_greeks(F, K, r, sigma, T, opt_type)

        roll_annual = _ROLL_YIELD_ANNUAL.get(ticker, -0.015)
        roll_daily = roll_annual / 252

        return PriceResult(
            price=g["price"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g["theta"],
            vega=g["vega"],
            rho=0.0,
            lambda_leverage=g["lambda_"],
            extended={"roll_yield_daily": roll_daily},
            model_used="Black76",
        )
