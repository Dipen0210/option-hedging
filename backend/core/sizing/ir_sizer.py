"""
Interest-rate position sizer — DV01 matching (L7).

Formula:
    N = portfolio_DV01 / instrument_DV01

where instrument_DV01 comes from PriceResult.extended["dv01"].

No integer rounding for OTC instruments (swaps, bonds).
ETF/futures positions are rounded to whole contracts.
"""
from __future__ import annotations

import math
from backend.core.sizing.base_sizer import BaseSizer, SizingContext, SizingResult

_OTC_INSTRUMENTS = {"SWAP", "IRS", "SOFR_SWAP"}


class IRSizer(BaseSizer):
    asset_class = "bond"

    def size(self, candidate, price_result, ctx: SizingContext) -> SizingResult:
        instrument_dv01 = price_result.extended.get("dv01", 0.0)
        if instrument_dv01 <= 0 or ctx.portfolio_dv01 <= 0:
            return SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)

        n_raw = ctx.portfolio_dv01 * ctx.hedge_ratio / instrument_dv01

        ticker_upper = candidate.ticker.upper()
        if any(t in ticker_upper for t in _OTC_INSTRUMENTS):
            n = n_raw           # OTC: fractional notional OK
        else:
            n = max(1, math.ceil(n_raw))

        # Cost = price × n (bond price × n contracts × multiplier)
        # For bonds priced at par = 100, cost = price / 100 × notional
        multiplier = ctx.extended.get("contract_multiplier", 1000)  # $1000 face
        total_cost = price_result.price * int(math.ceil(n)) * multiplier / 100
        notional_hedged = int(math.ceil(n)) * multiplier

        return SizingResult(
            n_contracts=int(math.ceil(n)),
            total_cost=round(total_cost, 2),
            notional_hedged=round(notional_hedged, 2),
            hedge_effectiveness=min(
                (int(math.ceil(n)) * instrument_dv01) / max(ctx.portfolio_dv01, 1e-6),
                1.0,
            ),
            partial_hedge_options=self._partial_series(
                candidate, price_result, ctx, int(math.ceil(n))
            ),
        )
