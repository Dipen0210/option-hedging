"""
Commodity position sizer — Minimum-variance hedge ratio (L7).

Formula (Johnson 1960):
    N = ρ × (σ_S / σ_F) × V_holding / (F × contract_size)

Where:
    ρ             = correlation between spot and futures
    σ_S           = spot annualised vol
    σ_F           = futures annualised vol
    V_holding     = dollar notional of commodity exposure
    F             = futures price
    contract_size = units per futures contract (default 100)

Roll cost is incorporated into total_cost from PriceResult.extended.
"""
from __future__ import annotations

import math
from backend.core.sizing.base_sizer import BaseSizer, SizingContext, SizingResult

# Contract sizes by commodity ticker
_CONTRACT_SIZE = {
    "GC": 100,    # Gold: 100 troy oz
    "SI": 5000,   # Silver: 5000 troy oz
    "CL": 1000,   # Crude: 1000 bbl
    "NG": 10000,  # Nat Gas: 10,000 MMBtu
    "ZC": 5000,   # Corn: 5000 bu
    "ZS": 5000,   # Soy: 5000 bu
    "GLD": 100,   # ETF proxy
    "USO": 100,
}
_DEFAULT_CONTRACT_SIZE = 100


class CommoditySizer(BaseSizer):
    asset_class = "commodity"

    def size(self, candidate, price_result, ctx: SizingContext) -> SizingResult:
        ticker = candidate.ticker.upper()
        F = ctx.extended.get("futures_price") or price_result.price
        if F <= 0:
            return SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)

        contract_size = _CONTRACT_SIZE.get(ticker, _DEFAULT_CONTRACT_SIZE)
        sigma_s = ctx.spot_vol if ctx.spot_vol > 0 else 0.20
        sigma_f = ctx.futures_vol if ctx.futures_vol > 0 else sigma_s
        rho = ctx.correlation

        # Minimum-variance hedge ratio
        h_star = rho * (sigma_s / sigma_f)
        n_raw = h_star * ctx.holding_notional * ctx.hedge_ratio \
                / (F * contract_size)
        n = max(1, round(n_raw))

        # Budget gate (roll cost included)
        roll_daily = price_result.extended.get("roll_yield_daily", 0.0)
        # Cost = option premium (if option) + roll cost over holding period
        days = ctx.extended.get("holding_days", 30)
        roll_cost = abs(roll_daily) * days * n * F * contract_size
        option_cost = price_result.price * contract_size * n if price_result.price > 0 else 0.0
        total_cost = option_cost + roll_cost

        notional_hedged = n * F * contract_size * h_star

        return SizingResult(
            n_contracts=n,
            total_cost=round(total_cost, 2),
            notional_hedged=round(notional_hedged, 2),
            hedge_effectiveness=min(notional_hedged / max(ctx.holding_notional, 1), 1.0),
            partial_hedge_options=self._partial_series(candidate, price_result, ctx, n),
        )
