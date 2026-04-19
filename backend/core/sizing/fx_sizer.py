"""
FX position sizer — Exact notional matching (L7).

FX forwards/options are OTC instruments priced in notional, not contracts.
No integer rounding — notional is continuous.

Formula:
    forward_notional = fx_exposure × hedge_ratio

Cost = option premium × notional  (for FX options)
     = 0                           (for forwards — no upfront premium)
"""
from __future__ import annotations

from backend.core.sizing.base_sizer import BaseSizer, SizingContext, SizingResult


class FXSizer(BaseSizer):
    asset_class = "fx"

    def size(self, candidate, price_result, ctx: SizingContext) -> SizingResult:
        fx_exposure = ctx.fx_exposure if ctx.fx_exposure > 0 else ctx.holding_notional
        notional = fx_exposure * ctx.hedge_ratio

        # FX options: cost = price × notional (price is in % of notional)
        # FX forwards: no premium
        is_option = candidate.option_type is not None
        if is_option and price_result.price > 0:
            total_cost = price_result.price * notional
        else:
            total_cost = 0.0    # forward: no upfront cost

        # "n_contracts" for FX = notional in base currency (continuous)
        # Store as int representation of thousands for display
        n_display = max(1, round(notional / 1000))

        return SizingResult(
            n_contracts=n_display,
            total_cost=round(total_cost, 2),
            notional_hedged=round(notional, 2),
            hedge_effectiveness=ctx.hedge_ratio,
            partial_hedge_options=[
                {
                    "hedge_pct": pct,
                    "notional": round(fx_exposure * pct, 2),
                    "total_cost": round(
                        price_result.price * fx_exposure * pct if is_option else 0.0, 2
                    ),
                }
                for pct in (0.25, 0.50, 0.75, 1.00)
            ],
        )
