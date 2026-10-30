"""
Portfolio analysis API — runs the full 12-layer pipeline.

POST /portfolio/analyze
    Body: PortfolioInput
    Returns: HedgeOutput

Market data (spot prices, vols, risk-free rate) are fetched automatically
from market data providers for every ticker in the portfolio.

Error policy: internal exception text is logged, never returned. Clients get a
stable error code plus a message safe to show a user, so a stack-trace detail
(file paths, library internals, SQL) can never reach the browser.
"""
import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException

from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import HedgeOutput
from backend.engines.pipeline_orchestrator import HedgeOrchestrator
from backend.api.security import user_facing_error, UserFacingError
from backend.data import data_quality as dq
from backend.data.portfolio_valuation import PortfolioValuationError

logger = logging.getLogger(__name__)
router = APIRouter()

# Analysis fans out to many market-data calls plus GMM scoring and Monte Carlo.
# Bound it so a provider outage cannot hold a worker open indefinitely.
ANALYSIS_TIMEOUT_SECONDS = 120


class MarketDataUnavailable(UserFacingError, RuntimeError):
    """
    No usable market data for the requested portfolio.

    A `UserFacingError`: the message is composed here for the user, naming the
    tickers that could not be priced, so it is safe to return verbatim.
    """


async def fetch_market_data(portfolio: PortfolioInput) -> dict:
    """
    Auto-fetch spot price and volatility for every holding.

    Runs provider calls concurrently in a thread pool. A ticker that cannot be
    priced is omitted and recorded as unavailable — it is never backfilled with
    a plausible-looking default. If no ticker can be priced the request fails
    loudly rather than analysing an unpriced portfolio.

    `sigma` here is 20-day realized volatility of the *underlying*, used only
    where no contract-specific implied vol applies. Option pricing must use
    per-strike IV from backend.data.options_chain; see L6.
    """
    from backend.data.market_data import (
        get_current_price,
        get_historical_volatility,
        get_risk_free_rate,
    )

    async def _fetch_one(ticker: str):
        try:
            spot = await asyncio.to_thread(get_current_price, ticker)
        except Exception as exc:
            logger.warning("Spot price unavailable for %s: %s", ticker, exc)
            dq.record(f"spot:{ticker}", dq.SOURCE_UNAVAILABLE, detail=str(exc)[:80])
            return ticker, None, None

        try:
            sigma = await asyncio.to_thread(get_historical_volatility, ticker)
        except Exception as exc:
            # Spot is usable without realized vol; downstream layers that need
            # vol will see it missing rather than receiving an invented number.
            logger.warning("Realized vol unavailable for %s: %s", ticker, exc)
            dq.record(f"realized_vol:{ticker}", dq.SOURCE_UNAVAILABLE, detail=str(exc)[:80])
            sigma = None

        return ticker, spot, sigma

    tickers = sorted({h.ticker for h in portfolio.holdings})
    rows = await asyncio.gather(*[_fetch_one(t) for t in tickers])

    market_data: dict = {}
    unpriced = []
    for ticker, spot, sigma in rows:
        if spot is None:
            unpriced.append(ticker)
            continue
        entry = {"spot_price": spot}
        if sigma is not None:
            entry["sigma"] = sigma
        market_data[ticker] = entry

    if not market_data:
        raise MarketDataUnavailable(
            "Market data is currently unavailable for every holding in this "
            f"portfolio ({', '.join(tickers)}). Please try again shortly."
        )

    try:
        rfr = await asyncio.to_thread(get_risk_free_rate)
        market_data["_global"] = {"risk_free_rate": rfr}
    except Exception as exc:
        # Disclosed by the orchestrator's _risk_free_rate() fallback.
        logger.warning("Risk-free rate unavailable: %s", exc)
        market_data["_global"] = {}

    if unpriced:
        logger.warning("Holdings excluded for lack of a price: %s", unpriced)

    return market_data


@router.post("/analyze", response_model=HedgeOutput)
async def analyze_portfolio(portfolio: PortfolioInput):
    """Full 12-layer hedge pipeline — all instrument types."""
    error_id = uuid.uuid4().hex[:12]

    with dq.capture():
        try:
            async with asyncio.timeout(ANALYSIS_TIMEOUT_SECONDS):
                market_data = await fetch_market_data(portfolio)
                orchestrator = HedgeOrchestrator(run_l9_simulation=True, run_l11_llm=True)
                return await asyncio.to_thread(orchestrator.run, portfolio, market_data)

        except MarketDataUnavailable as exc:
            # Safe to surface verbatim — this message is authored for the user.
            logger.warning("[%s] market data unavailable", error_id)
            raise user_facing_error(exc, status_code=503)

        except TimeoutError:
            logger.error("[%s] analysis exceeded %ds", error_id, ANALYSIS_TIMEOUT_SECONDS)
            raise HTTPException(
                status_code=504,
                detail=(
                    "Analysis took too long to complete, usually because a market "
                    f"data provider is slow. Please try again. (ref {error_id})"
                ),
            )

        except PortfolioValuationError as exc:
            # Message is authored for the user, so it is safe verbatim. A bare
            # ValueError is not — it may carry library internals — and falls
            # through to the generic handler below.
            logger.warning("[%s] portfolio could not be valued", error_id)
            raise user_facing_error(exc, status_code=422)

        except Exception:
            logger.exception("[%s] Portfolio analysis failed", error_id)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Analysis could not be completed due to an internal error. "
                    f"No recommendations were produced. (ref {error_id})"
                ),
            )
