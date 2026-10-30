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
  /admin/*                  — training logs + manual retrain triggers.
                              Requires `Authorization: Bearer $ADMIN_API_KEY`.
  /admin/cron/retrain-*     — scheduler-only retrain triggers.
                              Requires `Authorization: Bearer $CRON_SECRET`.
  /health                   — liveness probe (unauthenticated)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.api.security import require_admin, validate_environment
from backend.api.routes_portfolio import router as portfolio_router
from backend.api.routes_hedge     import router as instruments_router
from backend.api.routes_monitor   import router as admin_router
from backend.api.routes_monitor   import cron_router
from backend.api.validation_errors import validation_exception_handler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Security-relevant configuration is checked before the first request is
    # served. In production a gap raises and the process exits; in development
    # it is logged and startup continues, since the guarded routes refuse
    # themselves and nothing is left exposed.
    validate_environment()

    # The scheduler runs in-process. It needs a persistent process, which is
    # why the backend is a container rather than a serverless function: this
    # branch used to be skipped under Vercel, where APScheduler cannot run at
    # all, and retraining depended on an external cron hitting
    # /admin/cron/retrain-*. Those routes still exist and are still the right
    # trigger if the service is configured to scale to zero — a container that
    # is not running cannot fire a 24-hour timer.
    from backend.ml.regime.scheduler import build_scheduler
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("APScheduler started — fast(24h) + main(monthly) retrains scheduled.")

    yield  # server is running

    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")


app = FastAPI(
    title       = "OptionQ API",
    description = "AI-powered portfolio hedging platform",
    version     = "0.1.0",
    lifespan    = lifespan,
)

# CORS. `allow_methods`/`allow_headers` were "*", which with
# allow_credentials=True let any browser page on an allowed origin replay a
# session cookie against any verb. The API only ever serves GET and POST, and
# only needs these two request headers, so the policy says exactly that.
# Origins come from FRONTEND_URL (comma-separated) and are never wildcarded,
# because a wildcard origin plus credentials is rejected by browsers anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins    = settings.allowed_origins,
    allow_credentials= True,
    allow_methods    = ["GET", "POST", "OPTIONS"],
    allow_headers    = ["Authorization", "Content-Type"],
)

# Request validation failures are translated before they leave the service:
# FastAPI's default body is pydantic's own error objects, which name the
# library, expose the models' discriminator structure and echo the submitted
# portfolio back to the caller. See backend/api/validation_errors.py.
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(portfolio_router,  prefix="/portfolio",   tags=["portfolio"])
app.include_router(instruments_router,prefix="/instruments", tags=["instruments"])

# Admin is guarded at the mount point, so every current and future route on
# `admin_router` inherits authentication — a route added later cannot be left
# open by forgetting a decorator.
app.include_router(
    admin_router, prefix="/admin", tags=["admin"],
    dependencies=[Depends(require_admin)],
)

# The cron routes carry their own narrower credential (CRON_SECRET) and are
# mounted separately so that credential does not unlock the rest of /admin.
app.include_router(cron_router, prefix="/admin", tags=["admin"])


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "0.1.0"}
