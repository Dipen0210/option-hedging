"""
Equity position sizer — delta-adjusted sizing (L7).

Formula:
    N = floor( h x V_holding x target% / (|D| x 100 x S_hedge) )

Where:
    h         = hedge ratio — 1.0 for an option on the holding itself,
                h* (the min-variance ratio) for a proxy or macro hedge
    V_holding = market value of the holding being hedged
    target%   = the user's protection level (fraction of exposure to hedge)
    D         = option delta (absolute value)
    100       = shares per contract
    S_hedge   = spot price of the HEDGE instrument, not of the holding

The arithmetic itself lives in `backend.instruments.options.position_sizer`
and is shared with L4, which has to size a candidate before it can judge
whether the candidate is worth keeping. Two implementations of one formula is
how they drifted apart: this module previously re-derived the contract count
with `ceil` instead of `floor`, a 1e-6 delta floor instead of
`MIN_HEDGE_DELTA`, no minimum premium at all, `max(max_contracts, 1)` — which
let one contract through a budget that could not afford it — and no protection
level, so it sized every hedge as if the user had asked for 100% coverage.
Because L7 runs after L4, its numbers were the ones the user saw.
"""
from __future__ import annotations

from backend.core.sizing.base_sizer import BaseSizer, SizingContext, SizingResult
from backend.instruments.options.position_sizer import (
    MIN_HEDGE_DELTA,
    compute_position,
    size_by_delta,
)


class EquitySizer(BaseSizer):
    asset_class = "equity"

    @staticmethod
    def _hedge_spot(candidate, ctx: SizingContext) -> float:
        """
        Spot price of the instrument being bought.

        `ctx` is built per holding, so `ctx.extended["spot_price"]` is the
        holding's price. For a cross hedge — QQQ puts against an AAPL
        position — that is the wrong number on both sides of the division, and
        it silently mis-sized every proxy and macro hedge. L4 and L6 each
        record the spot they priced against; the holding's price is only a
        last resort, and then only when the candidate is on the holding
        itself.
        """
        spot = candidate.extended_metrics.get("hedge_spot")
        if spot and spot > 0:
            return float(spot)
        return float(ctx.extended.get("spot_price", 0.0))

    @staticmethod
    def _hedge_ratio(candidate, ctx: SizingContext) -> float:
        """
        How much hedge instrument stands for one unit of the holding.

        L4 computes this per candidate — 1.0 for a direct hedge, h* for a
        proxy — and records it, because L7's context is per holding and cannot
        tell the two apart. `ctx.beta` is the fallback and is only right for a
        hedge whose underlying is SPY.
        """
        ratio = candidate.extended_metrics.get("sizing_ratio")
        if ratio is not None:
            return float(ratio)
        return ctx.beta

    def size(self, candidate, price_result, ctx: SizingContext) -> SizingResult:
        S = self._hedge_spot(candidate, ctx)
        premium = price_result.price
        ratio = self._hedge_ratio(candidate, ctx)

        if S <= 0 or abs(price_result.delta) < MIN_HEDGE_DELTA:
            return SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)

        if premium < 0:
            # Net credit (a collar whose short call more than finances the
            # long put). The cost budget cannot bind on a trade that pays the
            # user, so size on the delta target alone and carry the credit
            # through as a negative cost rather than as zero.
            n = size_by_delta(
                ctx.holding_notional, ratio, price_result.delta,
                S, 100, ctx.hedge_ratio,
            )
            if n <= 0:
                return SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)
            total_cost = round(n * premium * 100, 2)
            partial_series = []
        else:
            sizing = compute_position(
                portfolio_notional=ctx.holding_notional,
                asset_beta=ratio,
                option_delta=price_result.delta,
                option_price=premium,
                spot_price=S,
                max_cost_pct=ctx.max_premium_pct,
                hedge_target_pct=ctx.hedge_ratio,
            )
            if not sizing["viable"]:
                # Nothing affordable, or nothing priceable. Reporting one
                # contract anyway would quote the user a trade that breaches
                # the budget they set.
                return SizingResult(n_contracts=0, total_cost=0.0, notional_hedged=0.0)
            n = sizing["gated"]["n_contracts_gated"]
            total_cost = sizing["gated"]["total_cost"]
            partial_series = sizing["partial_series"]

        notional_hedged = n * abs(price_result.delta) * 100 * S

        return SizingResult(
            n_contracts=n,
            total_cost=total_cost,
            notional_hedged=round(notional_hedged, 2),
            hedge_effectiveness=min(notional_hedged / max(ctx.holding_notional, 1), 1.0),
            partial_hedge_options=partial_series,
        )
