"""
OptionQ API — FastAPI application entry point.

Startup behaviour:
  - APScheduler starts in the background with two jobs:
      fast_retrain  every 24 hours
      main_retrain  1st of every month at 02:00 UTC
  - Both models are trained on first run if missing (handled inside predict()).

Routes:
  /portfolio/analyze        — full 12-layer pipeline
  /instruments/{type}       — single-instrument pipelines
  /admin/training/*         — training logs + manual retrain triggers
  /health                   — liveness probe
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.api.routes_portfolio import router as portfolio_router
from backend.api.routes_hedge     import router as instruments_router
from backend.api.routes_monitor   import router as admin_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Start background scheduler ────────────────────────────────────────
    from backend.ml.regime.scheduler import build_scheduler
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("APScheduler started — fast(24h) + main(monthly) retrains scheduled.")

    yield  # server is running

    # ── Shutdown ──────────────────────────────────────────────────────────
    scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped.")


app = FastAPI(
    title       = "OptionQ API",
    description = "AI-powered portfolio hedging platform",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    = [settings.frontend_url],
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)

app.include_router(portfolio_router,  prefix="/portfolio",   tags=["portfolio"])
app.include_router(instruments_router,prefix="/instruments", tags=["instruments"])
app.include_router(admin_router,      prefix="/admin",       tags=["admin"])


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "0.1.0"}
