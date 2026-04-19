"""
Equity position sizer — Delta-adjusted sizing (L7).

Formula:
    N = ceil( β × V_holding / (|Δ| × 100 × S) )

Where:
    β         = portfolio beta vs hedge instrument
    V_holding = dollar value of the holding being hedged
    Δ         = option delta (absolute value)
    100       = shares per contract
    S         = current spot price of the hedge instrument
"""
from __future__ import annotations

import math
from backend.core.sizing.base_sizer import BaseSizer, SizingContext, SizingResult


class EquitySizer(BaseSizer):
    asset_class = "equity"

    def size(self, candidate, price_result, ctx: SizingContext) -> SizingResult:
        delta_abs = abs(price_result.delta)
        S = ctx.extended.get("spot_price", 0.0)
        if S <= 0 or delta_abs < 1e-6:
            return SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)

        # Delta-adjusted contracts
        n_raw = (ctx.beta * ctx.holding_notional * ctx.hedge_ratio) \
                / (delta_abs * 100 * S)
        n = max(1, math.ceil(n_raw))

        # Budget gate: cap at max_premium_pct
        cost_per = price_result.price * 100
        max_contracts = math.floor(
            (ctx.holding_notional * ctx.max_premium_pct) / max(cost_per, 1e-6)
        )
        n = min(n, max(max_contracts, 1))

        total_cost = n * cost_per
        notional_hedged = n * delta_abs * 100 * S

        return SizingResult(
            n_contracts=n,
            total_cost=round(total_cost, 2),
            notional_hedged=round(notional_hedged, 2),
            hedge_effectiveness=min(notional_hedged / max(ctx.holding_notional, 1), 1.0),
            partial_hedge_options=self._partial_series(candidate, price_result, ctx, n),
        )
