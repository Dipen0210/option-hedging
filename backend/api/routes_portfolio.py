"""
Portfolio analysis API — runs the full 12-layer pipeline.

POST /portfolio/analyze
    Body: PortfolioInput
    Returns: HedgeOutput

Market data (spot prices, vols, risk-free rate) are fetched automatically
from yfinance for every ticker in the portfolio.
"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException

from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import HedgeOutput
from backend.engines.pipeline_orchestrator import HedgeOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


async def fetch_market_data(portfolio: PortfolioInput) -> dict:
    """
    Auto-fetch spot price and historical vol for every holding.
    Runs yfinance calls concurrently in a thread pool.
    """
    from backend.data.market_data import (
        get_current_price,
        get_historical_volatility,
        get_risk_free_rate,
    )

    async def _fetch_one(ticker: str):
        spot  = await asyncio.to_thread(get_current_price, ticker)
        sigma = await asyncio.to_thread(get_historical_volatility, ticker)
        return ticker, spot, sigma

    tickers = list({h.ticker for h in portfolio.holdings})
    rows = await asyncio.gather(*[_fetch_one(t) for t in tickers])

    market_data: dict = {}
    for ticker, spot, sigma in rows:
        market_data[ticker] = {"spot_price": spot, "sigma": sigma}

    rfr = await asyncio.to_thread(get_risk_free_rate)
    market_data["_global"] = {"risk_free_rate": rfr}
    return market_data


@router.post("/analyze", response_model=HedgeOutput)
async def analyze_portfolio(portfolio: PortfolioInput):
    """Full 12-layer hedge pipeline — all instrument types."""
    try:
        market_data = await fetch_market_data(portfolio)
        orchestrator = HedgeOrchestrator(run_l9_simulation=True, run_l11_llm=False)
        result: HedgeOutput = await asyncio.to_thread(
            orchestrator.run, portfolio, market_data
        )
        return result
    except Exception as exc:
        logger.exception("Portfolio analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
