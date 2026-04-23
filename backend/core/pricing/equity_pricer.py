"""
Equity option pricer (L6).

Routing logic:
  - Broad-market ETF tickers (SPY, QQQ, IWM, …) → BSM (European)
  - Single-stock tickers                         → CRR Binomial (American)
  - Multi-leg strategies (spread, collar)        → net Greeks from both legs

Per-strike IV: for each leg, looks up the actual implied volatility at the
specific strike from the live options chain (volatility skew). Falls back to
the ATM IV from PricingContext if the chain lookup fails.
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.core.pricing.base_pricer import BaseAssetPricer, PriceResult, PricingContext
from backend.instruments.options.bsm_pricer import bsm_greeks
from backend.instruments.options.binomial_pricer import crr_greeks
from backend.instruments.options.greeks import net_greeks

logger = logging.getLogger(__name__)

# Tickers whose options trade as European-style → use BSM
_ETF_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "SLV", "TLT", "HYG",
    "XLF", "XLE", "XLK", "XLV", "EEM", "EFA", "VXX",
}


def _resolve_strike_iv(
    ticker: str,
    expiry_date: Optional[str],
    strike: float,
    opt_type: str,
    fallback: float,
) -> float:
    """
    Look up per-strike implied volatility from the real options chain.
    Uses linear interpolation between adjacent strikes (volatility smile).
    Returns fallback (ATM IV) if the chain is unavailable.
    """
    if not expiry_date:
        return fallback
    try:
        from backend.data.options_chain import get_strike_iv
        iv = get_strike_iv(ticker, expiry_date, strike, opt_type)
        if iv and 0.01 < iv < 5.0:
            logger.debug(
                "L6 per-strike IV: %s %s K=%.1f → %.3f (fallback=%.3f)",
                ticker, opt_type, strike, iv, fallback,
            )
            return iv
    except Exception:
        pass
    return fallback


class EquityOptionPricer(BaseAssetPricer):
    asset_class = "equity"

    def _price_leg(
        self,
        ticker: str,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        opt_type: str,
    ) -> dict:
        """Price a single option leg using BSM (ETFs) or CRR Binomial (stocks)."""
        if ticker in _ETF_TICKERS:
            return bsm_greeks(S, K, r, q, sigma, T, opt_type)
        return crr_greeks(S, K, r, q, sigma, T, opt_type, american=True)

    def price(self, candidate, ctx: PricingContext) -> PriceResult:
        ticker = candidate.ticker.upper()
        S = ctx.spot_for(ticker)
        K = candidate.strike or S          # fallback: ATM
        T = 0.0
        if candidate.expiry_date:
            from backend.core.pricing.base_pricer import years_to_expiry
            T = years_to_expiry(candidate.expiry_date)
        r = ctx.risk_free_rate
        q = ctx.div_yield_for(ticker)
        sigma_atm = ctx.vol_for(ticker)
        opt_type = (candidate.option_type or "put").lower()

        if S <= 0 or K <= 0:
            return PriceResult(price=0.0, model_used="equity_skip")

        model_suffix = "BSM" if ticker in _ETF_TICKERS else "CRR_Binomial"

        # ── Multi-leg strategy (spread / collar) ──────────────────────────────
        if candidate.short_strike:
            K_short = candidate.short_strike
            short_type = (candidate.short_option_type or opt_type).lower()

            # Resolve per-strike IV for each leg independently
            sigma_long  = _resolve_strike_iv(ticker, candidate.expiry_date, K,       opt_type,   sigma_atm)
            sigma_short = _resolve_strike_iv(ticker, candidate.expiry_date, K_short, short_type, sigma_atm)

            g_long  = self._price_leg(ticker, S, K,       r, q, sigma_long,  T, opt_type)
            g_short = self._price_leg(ticker, S, K_short, r, q, sigma_short, T, short_type)
            net = net_greeks([(g_long, +1), (g_short, -1)])

            net_price = max(net["price"], 0.0)   # spreads are always a debit or zero
            return PriceResult(
                price=net_price,
                delta=net["delta"],
                gamma=net["gamma"],
                theta=net["theta"],
                vega=net["vega"],
                rho=0.0,
                lambda_leverage=net["lambda_"],
                model_used=f"{model_suffix}_spread",
            )

        # ── Single-leg (protective put, OTM put, etc.) ────────────────────────
        sigma = _resolve_strike_iv(ticker, candidate.expiry_date, K, opt_type, sigma_atm)
        g = self._price_leg(ticker, S, K, r, q, sigma, T, opt_type)
        return PriceResult(
            price=g["price"],
            delta=g["delta"],
            gamma=g["gamma"],
            theta=g["theta"],
            vega=g["vega"],
            rho=0.0,
            lambda_leverage=g["lambda_"],
            model_used=model_suffix,
        )
