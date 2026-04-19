"""
Interest-rate instrument pricer (L6).

IR instruments (Treasuries, swaps, SOFR futures) are priced via
yield-curve discounting — NOT BSM.  There are no BSM Greeks.

Output uses extended dict for DV01-based metrics:
  dv01          — dollar value of a 1bp move
  duration      — Macaulay duration (years)
  mod_duration  — Modified duration
  convexity     — Convexity (years²)
  yield_level   — current yield of the instrument

For options on rate instruments, a simplified Hull-White approximation
is used to estimate premium (if option_type is set).
"""
from __future__ import annotations

import math
from backend.core.pricing.base_pricer import BaseAssetPricer, PriceResult, PricingContext

# Approximate duration (years) by instrument type
_DURATION_MAP = {
    "TLT": 17.0,    # 20yr Treasury ETF
    "IEF": 7.5,     # 7-10yr Treasury ETF
    "SHY": 1.8,     # 1-3yr Treasury ETF
    "LQD": 8.5,     # IG Corp Bond ETF
    "HYG": 4.0,     # HY Bond ETF
    "SOFR": 0.25,   # SOFR future ~3-month
    "ZB":  15.0,    # 30yr T-Bond future
    "ZN":  6.5,     # 10yr T-Note future
    "ZF":  4.0,     # 5yr T-Note future
    "ZT":  1.9,     # 2yr T-Note future
}

_CONVEXITY_MAP = {
    "TLT": 300.0,
    "IEF": 60.0,
    "SHY": 3.5,
    "LQD": 80.0,
    "HYG": 18.0,
    "ZB":  250.0,
    "ZN":  50.0,
    "ZF":  18.0,
    "ZT":  4.0,
}


def _yield_for(ticker: str, yield_curve: dict) -> float:
    """Pick the most relevant rate from yield_curve for an instrument."""
    dur = _DURATION_MAP.get(ticker, 5.0)
    # Map duration to nearest tenor key
    tenors = sorted(yield_curve.keys())
    if not tenors:
        return 0.05
    closest = min(tenors, key=lambda t: abs(float(t) - dur))
    return yield_curve.get(closest, 0.05)


class InterestRatePricer(BaseAssetPricer):
    asset_class = "bond"

    def price(self, candidate, ctx: PricingContext) -> PriceResult:
        ticker = candidate.ticker.upper()
        S = ctx.spot_for(ticker)           # price of the bond/ETF
        if S <= 0:
            S = 100.0                      # assume par if no price

        duration = _DURATION_MAP.get(ticker, 5.0)
        convexity = _CONVEXITY_MAP.get(ticker, 50.0)
        y = _yield_for(ticker, ctx.yield_curve) if ctx.yield_curve else ctx.risk_free_rate
        mod_duration = duration / (1 + y)

        # DV01 = Modified Duration × Price × 0.0001
        dv01 = mod_duration * S * 0.0001

        # If this is an option on a rate instrument, use a rough Black approx
        if candidate.option_type and candidate.strike and candidate.expiry_date:
            from backend.core.pricing.base_pricer import years_to_expiry
            from backend.instruments.options.bsm_pricer import bsm_price
            T = years_to_expiry(candidate.expiry_date)
            vol = ctx.vol_for(ticker)
            K = candidate.strike
            opt = candidate.option_type.lower()
            price = bsm_price(S, K, ctx.risk_free_rate, 0.0, vol, T, opt)
            model = "BSM_on_rate"
        else:
            # Forward price of the bond (no option)
            T_settle = 0.0
            if candidate.expiry_date:
                from backend.core.pricing.base_pricer import years_to_expiry
                T_settle = years_to_expiry(candidate.expiry_date)
            price = S * math.exp(-y * T_settle) if T_settle > 0 else S
            model = "yield_curve_discount"

        return PriceResult(
            price=round(price, 4),
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            lambda_leverage=0.0,
            extended={
                "dv01": round(dv01, 6),
                "duration": duration,
                "mod_duration": round(mod_duration, 4),
                "convexity": convexity,
                "yield_level": round(y, 6),
            },
            model_used=model,
        )
