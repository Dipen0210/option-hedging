"""
FX option pricer — Garman-Kohlhagen (L6).

GK is BSM extended with two rates:
  r_d = domestic risk-free rate  (replaces r in BSM)
  r_f = foreign risk-free rate   (replaces dividend yield q)

Delta is dV/dS_fx.
Two Rho values: rho_d (domestic) and rho_f (foreign).

Extended metrics:
  rho_domestic   — sensitivity to domestic rate
  rho_foreign    — sensitivity to foreign rate
  r_domestic     — rate used
  r_foreign      — rate used
"""
from __future__ import annotations

import math
from scipy.stats import norm

from backend.core.pricing.base_pricer import BaseAssetPricer, PriceResult, PricingContext
from backend.core.pricing.base_pricer import years_to_expiry

# Default foreign rates by currency pair (approximate)
_FOREIGN_RATE = {
    "EURUSD": 0.040, "GBPUSD": 0.052, "USDJPY": 0.001,
    "USDCHF": 0.015, "AUDUSD": 0.042, "USDCAD": 0.050,
    "USDMXN": 0.110, "USDBRL": 0.135,
}


def _gk_greeks(
    S: float, K: float, r_d: float, r_f: float,
    sigma: float, T: float, opt_type: str = "put"
) -> dict:
    """Garman-Kohlhagen full Greeks."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        price = max(K * math.exp(-r_d * T) - S * math.exp(-r_f * T), 0.0) \
                if opt_type == "put" else \
                max(S * math.exp(-r_f * T) - K * math.exp(-r_d * T), 0.0)
        return {"price": price, "delta": 0.0, "gamma": 0.0, "theta": 0.0,
                "vega": 0.0, "rho_d": 0.0, "rho_f": 0.0, "lambda_": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc_d = math.exp(-r_d * T)
    disc_f = math.exp(-r_f * T)
    n_d1 = norm.pdf(d1)

    if opt_type == "put":
        price = K * disc_d * norm.cdf(-d2) - S * disc_f * norm.cdf(-d1)
        delta = -disc_f * norm.cdf(-d1)
        rho_d = K * T * disc_d * norm.cdf(-d2)          # positive for puts
        rho_f = -S * T * disc_f * norm.cdf(-d1)         # negative
    else:
        price = S * disc_f * norm.cdf(d1) - K * disc_d * norm.cdf(d2)
        delta = disc_f * norm.cdf(d1)
        rho_d = -K * T * disc_d * norm.cdf(d2)
        rho_f = S * T * disc_f * norm.cdf(d1)

    price = max(float(price), 0.0)
    gamma = disc_f * n_d1 / (S * sigma * sqrt_T)
    theta = (
        -disc_f * S * n_d1 * sigma / (2 * sqrt_T)
        + r_f * S * disc_f * (norm.cdf(-d1) if opt_type == "put" else norm.cdf(d1))
        - r_d * K * disc_d * (norm.cdf(-d2) if opt_type == "put" else norm.cdf(d2))
    ) / 365
    vega = S * disc_f * sqrt_T * n_d1 / 100
    lambda_ = delta * S / price if price > 1e-6 else 0.0

    return {
        "price": round(price, 6),
        "delta": round(delta, 6),
        "gamma": round(gamma, 8),
        "theta": round(theta, 6),
        "vega": round(vega, 6),
        "rho_d": round(rho_d, 6),
        "rho_f": round(rho_f, 6),
        "lambda_": round(lambda_, 4),
    }


class FXPricer(BaseAssetPricer):
    asset_class = "fx"

    def price(self, candidate, ctx: PricingContext) -> PriceResult:
        ticker = candidate.ticker.upper()
        S = ctx.fx_rates.get(ticker) or ctx.spot_for(ticker)
        K = candidate.strike or S
        T = years_to_expiry(candidate.expiry_date) if candidate.expiry_date else 0.0
        r_d = ctx.risk_free_rate
        r_f = _FOREIGN_RATE.get(ticker, 0.03)
        sigma = ctx.vol_for(ticker)
        opt_type = (candidate.option_type or "put").lower()

        if S <= 0:
            return PriceResult(price=0.0, model_used="fx_skip")

        g = _gk_greeks(S, K, r_d, r_f, sigma, T, opt_type)

        return PriceResult(
            price=g["price"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g["theta"],
            vega=g["vega"],
            rho=g["rho_d"],               # primary rho = domestic
            lambda_leverage=g["lambda_"],
            extended={
                "rho_domestic": g["rho_d"],
                "rho_foreign": g["rho_f"],
                "r_domestic": r_d,
                "r_foreign": r_f,
            },
            model_used="Garman_Kohlhagen",
        )
