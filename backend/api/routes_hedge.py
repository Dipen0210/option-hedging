"""
Per-instrument hedge APIs — each endpoint runs the pipeline restricted
to a single instrument type.

POST /instruments/options        — options only (BSM/CRR/Black-76)
POST /instruments/futures        — futures only
POST /instruments/inverse-etfs   — inverse ETFs only
POST /instruments/forwards       — forwards only
POST /instruments/swaps          — swaps only

All endpoints accept the same PortfolioInput body and return HedgeOutput.
Market data is auto-fetched from yfinance.
"""
import asyncio
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import HedgeOutput
from backend.engines.pipeline_orchestrator import HedgeOrchestrator
from backend.api.routes_portfolio import fetch_market_data

logger = logging.getLogger(__name__)
router = APIRouter()


async def _run(portfolio: PortfolioInput, instrument_types: List[str]) -> HedgeOutput:
    market_data = await fetch_market_data(portfolio)
    orchestrator = HedgeOrchestrator(
        run_l9_simulation=False,
        run_l11_llm=False,
        instrument_types=instrument_types,
    )
    return await asyncio.to_thread(orchestrator.run, portfolio, market_data)


@router.post("/options", response_model=HedgeOutput)
async def options_hedge(portfolio: PortfolioInput):
    """Hedge candidates using options only (puts, spreads, collars)."""
    try:
        return await _run(portfolio, ["options"])
    except Exception as exc:
        logger.exception("Options analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/futures", response_model=HedgeOutput)
async def futures_hedge(portfolio: PortfolioInput):
    """Hedge candidates using index / commodity / treasury futures."""
    try:
        return await _run(portfolio, ["futures"])
    except Exception as exc:
        logger.exception("Futures analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/inverse-etfs", response_model=HedgeOutput)
async def inverse_etfs_hedge(portfolio: PortfolioInput):
    """Hedge candidates using inverse and leveraged-inverse ETFs."""
    try:
        return await _run(portfolio, ["inverse_etfs"])
    except Exception as exc:
        logger.exception("Inverse ETF analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/forwards", response_model=HedgeOutput)
async def forwards_hedge(portfolio: PortfolioInput):
    """Hedge candidates using FX and commodity forwards."""
    try:
        return await _run(portfolio, ["forwards"])
    except Exception as exc:
        logger.exception("Forwards analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/swaps", response_model=HedgeOutput)
async def swaps_hedge(portfolio: PortfolioInput):
    """Hedge candidates using interest rate swaps, TRS, and CDS."""
    try:
        return await _run(portfolio, ["swaps"])
    except Exception as exc:
        logger.exception("Swaps analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
