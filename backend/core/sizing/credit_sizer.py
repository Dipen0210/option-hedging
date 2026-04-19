"""
Credit position sizer — CS01 matching (L7).

Formula:
    N = portfolio_CS01 / instrument_CS01

where instrument_CS01 comes from PriceResult.extended["cs01"].

CDS contracts have standard notional of $10M ($10k CS01 per 100bps).
"""
from __future__ import annotations

import math
from backend.core.sizing.base_sizer import BaseSizer, SizingContext, SizingResult

_STANDARD_NOTIONAL = 1_000_000   # $1M per CDS contract (simplified)


class CreditSizer(BaseSizer):
    asset_class = "credit"

    def size(self, candidate, price_result, ctx: SizingContext) -> SizingResult:
        instrument_cs01 = price_result.extended.get("cs01", 0.0)
        portfolio_cs01 = ctx.portfolio_cs01

        # If no CS01 budget provided, fall back to simple notional match
        if portfolio_cs01 <= 0 or instrument_cs01 <= 0:
            notional = ctx.holding_notional * ctx.hedge_ratio
            n = max(1, math.ceil(notional / _STANDARD_NOTIONAL))
            total_cost = n * price_result.price
            return SizingResult(
                n_contracts=n,
                total_cost=round(total_cost, 2),
                notional_hedged=round(n * _STANDARD_NOTIONAL, 2),
            )

        n_raw = portfolio_cs01 * ctx.hedge_ratio / instrument_cs01
        n = max(1, math.ceil(n_raw))

        total_cost = n * price_result.price
        notional_hedged = n * _STANDARD_NOTIONAL

        return SizingResult(
            n_contracts=n,
            total_cost=round(total_cost, 2),
            notional_hedged=round(notional_hedged, 2),
            hedge_effectiveness=min(
                (n * instrument_cs01) / max(portfolio_cs01, 1e-6), 1.0
            ),
            partial_hedge_options=self._partial_series(candidate, price_result, ctx, n),
        )
