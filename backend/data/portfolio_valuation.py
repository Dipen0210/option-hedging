"""
Portfolio valuation — market value from live prices, kept distinct from cost basis.

Hedge sizing, VaR and stress losses must all scale off *current exposure*. The
client previously supplied `total_notional` computed from purchase price, which
is cost basis; on the default portfolio that understated exposure by 52% and
mis-scaled every dollar figure downstream.

This module recomputes market value server-side and reports both numbers plus
their divergence, so the user can see what their portfolio is actually worth.
"""

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

from backend.models.portfolio_models import (
    PortfolioInput, StockPosition, OptionsPosition,
)
from backend.api.security import UserFacingError
from backend.data import data_quality as dq

logger = logging.getLogger(__name__)

# Divergence between declared and computed value that is worth telling the user.
DIVERGENCE_WARN_PCT = 0.05


class PortfolioValuationError(UserFacingError, ValueError):
    """
    A valuation problem whose message is written for the end user.

    Distinct from a bare ValueError so the API can surface this verbatim while
    an incidental ValueError raised inside pandas/numpy still goes down the
    generic 500 path. Without the distinction, `detail=str(exc)` on a 422
    forwards library internals ("Length of values (3) does not match length of
    index (5)") to the browser as if it were guidance for the user.
    """


class HoldingValuation(BaseModel):
    key: str                     # pipeline key (ticker, or TICKER_P650 for options)
    ticker: str
    position_type: str
    quantity: float
    cost_price: float            # per share / per contract-share
    current_price: float         # live underlying price
    cost_basis: float
    market_value: float          # current exposure in dollars
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float                # share of portfolio market value
    price_source: str = "live"


class PortfolioValuation(BaseModel):
    market_value: float
    cost_basis: float
    declared_notional: Optional[float] = None
    declared_divergence_pct: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    holdings: List[HoldingValuation] = []
    warnings: List[str] = []

    def spot_prices(self) -> Dict[str, float]:
        return {h.ticker: h.current_price for h in self.holdings}


def _option_key(opt: OptionsPosition) -> str:
    return f"{opt.ticker}_{opt.option_type[0].upper()}{int(opt.strike)}"


def value_portfolio(
    portfolio: PortfolioInput,
    spot_prices: Optional[Dict[str, float]] = None,
) -> PortfolioValuation:
    """
    Compute current market value for a portfolio.

    `spot_prices` may be supplied when the caller already fetched them (the API
    route does, concurrently); otherwise prices are fetched here.

    Raises ValueError when no holding could be priced — analysis must not
    proceed on an unpriced portfolio.
    """
    from backend.data.market_data import get_current_price

    prices: Dict[str, float] = dict(spot_prices or {})
    rows: List[HoldingValuation] = []
    warnings: List[str] = []

    def _price(ticker: str) -> Optional[float]:
        if ticker in prices:
            return prices[ticker]
        try:
            prices[ticker] = get_current_price(ticker)
            return prices[ticker]
        except Exception as exc:
            logger.warning("Could not price %s: %s", ticker, exc)
            dq.record(f"spot:{ticker}", dq.SOURCE_UNAVAILABLE, detail=str(exc)[:80])
            return None

    for h in portfolio.holdings:
        spot = _price(h.ticker)
        if spot is None:
            warnings.append(
                f"{h.ticker}: live price unavailable — position excluded from "
                f"portfolio market value"
            )
            continue

        if isinstance(h, StockPosition):
            cost_basis = h.shares * h.purchase_price
            market_value = h.shares * spot
            rows.append(HoldingValuation(
                key=h.ticker, ticker=h.ticker, position_type="stock",
                quantity=h.shares, cost_price=h.purchase_price, current_price=spot,
                cost_basis=cost_basis, market_value=market_value,
                unrealized_pnl=market_value - cost_basis,
                unrealized_pnl_pct=(market_value / cost_basis - 1.0) if cost_basis else 0.0,
                weight=0.0,
            ))
        else:
            # Underlying notional controlled by the contracts. Delta-adjusted
            # exposure is computed in L3; this is the gross controlled notional.
            market_value = h.contracts * 100 * spot
            premium = h.premium_paid if h.premium_paid is not None else 0.0
            cost_basis = h.contracts * 100 * premium
            rows.append(HoldingValuation(
                key=_option_key(h), ticker=h.ticker, position_type="option",
                quantity=h.contracts, cost_price=premium, current_price=spot,
                cost_basis=cost_basis, market_value=market_value,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0, weight=0.0,
            ))

    if not rows:
        raise PortfolioValuationError(
            "Could not retrieve a live price for any holding — portfolio cannot be valued"
        )

    total_mv = sum(r.market_value for r in rows)
    total_cb = sum(r.cost_basis for r in rows)
    for r in rows:
        r.weight = (r.market_value / total_mv) if total_mv > 0 else 0.0

    declared = portfolio.total_notional
    divergence = None
    if declared and total_mv > 0:
        divergence = declared / total_mv - 1.0
        if abs(divergence) > DIVERGENCE_WARN_PCT:
            warnings.append(
                f"Submitted portfolio value ${declared:,.0f} differs from computed "
                f"market value ${total_mv:,.0f} by {divergence:+.1%}. "
                f"Analysis uses the computed market value."
            )

    if len(rows) < len(portfolio.holdings):
        dq.record("portfolio_valuation", dq.SOURCE_FALLBACK,
                  detail=f"{len(portfolio.holdings) - len(rows)} holding(s) unpriced")
    else:
        dq.record("portfolio_valuation", dq.SOURCE_LIVE)

    return PortfolioValuation(
        market_value=total_mv,
        cost_basis=total_cb,
        declared_notional=declared,
        declared_divergence_pct=divergence,
        unrealized_pnl=total_mv - total_cb,
        unrealized_pnl_pct=(total_mv / total_cb - 1.0) if total_cb else 0.0,
        holdings=rows,
        warnings=warnings,
    )
