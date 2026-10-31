"""
Per-instrument hedge APIs — each endpoint runs the pipeline restricted
to a single instrument type.

POST /instruments/options        — options only (BSM/CRR/Black-76)
POST /instruments/futures        — not implemented in this version
POST /instruments/inverse-etfs   — not implemented in this version
POST /instruments/forwards       — not implemented in this version
POST /instruments/swaps          — not implemented in this version

All endpoints accept the same PortfolioInput body and return HedgeOutput.
Market data is auto-fetched from yfinance.

Only the options selector is built. The other four selectors are declared
stubs: they evaluate nothing and return no candidates. Those endpoints are
kept routable so their contract is visible and their absence is stated — each
returns 200 with an empty recommendation set and a warning naming
non-implementation as the reason. They are documented that way here because
this docstring is the description a reader sees in /docs, and the previous
text promised hedges the endpoint cannot produce.

Error policy: internal exception text is logged against a correlation id and
never returned. Clients get a generic 500 carrying only that id.
"""
import asyncio
import logging
from typing import List

from fastapi import APIRouter

from backend.api.security import internal_error

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
        run_l11_llm=True,
        instrument_types=instrument_types,
    )
    return await asyncio.to_thread(orchestrator.run, portfolio, market_data)


@router.post("/options", response_model=HedgeOutput)
async def options_hedge(portfolio: PortfolioInput):
    """Hedge candidates using options only (puts, spreads, collars)."""
    try:
        return await _run(portfolio, ["options"])
    except Exception as exc:
        raise internal_error(exc, "options analysis")


@router.post("/futures", response_model=HedgeOutput)
async def futures_hedge(portfolio: PortfolioInput):
    """
    Not implemented: no futures hedge is evaluated.

    Returns 200 with no candidates and a warning saying so. Planned coverage
    is index, treasury and commodity futures; none of it is built, so nothing
    here consults a price, a budget or a contract specification.
    """
    try:
        return await _run(portfolio, ["futures"])
    except Exception as exc:
        raise internal_error(exc, "futures analysis")


@router.post("/inverse-etfs", response_model=HedgeOutput)
async def inverse_etfs_hedge(portfolio: PortfolioInput):
    """
    Not implemented: no inverse-ETF hedge is evaluated.

    Returns 200 with no candidates and a warning saying so.
    """
    try:
        return await _run(portfolio, ["inverse_etfs"])
    except Exception as exc:
        raise internal_error(exc, "inverse etf analysis")


@router.post("/forwards", response_model=HedgeOutput)
async def forwards_hedge(portfolio: PortfolioInput):
    """
    Not implemented: no forward hedge is evaluated.

    Returns 200 with no candidates and a warning saying so.
    """
    try:
        return await _run(portfolio, ["forwards"])
    except Exception as exc:
        raise internal_error(exc, "forwards analysis")


@router.post("/swaps", response_model=HedgeOutput)
async def swaps_hedge(portfolio: PortfolioInput):
    """
    Not implemented: no swap hedge is evaluated.

    Returns 200 with no candidates and a warning saying so.
    """
    try:
        return await _run(portfolio, ["swaps"])
    except Exception as exc:
        raise internal_error(exc, "swaps analysis")
