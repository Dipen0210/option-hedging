"""
Admin / monitoring routes.

GET  /admin/training/log              — full training history (both models)
GET  /admin/training/log/{model}      — log for "main" or "fast"
GET  /admin/training/status           — last run + model ages + next scheduled run
POST /admin/retrain/main              — force retrain main model (blocking)
POST /admin/retrain/fast              — force retrain fast model (blocking)

Gap 4 — Hedge monitor:
POST /admin/hedge/check-triggers      — check roll/delta-drift/spot-move alerts
GET  /admin/earnings/{ticker}         — next earnings date for a ticker
"""

import asyncio
import logging
import os
import time
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Gap 4: Hedge monitor models ───────────────────────────────────────────────

class HedgeCheckRequest(BaseModel):
    """Active hedge position to check for re-hedge triggers."""
    hedge_ticker: str               # ticker of the option (e.g. "QQQ")
    option_type: str                # "put" or "call"
    strike: float
    expiry_date: str                # ISO date e.g. "2026-06-20"
    n_contracts: int = 1
    entry_spot: float               # spot at hedge inception
    initial_delta: float            # delta at inception (e.g. -0.35)
    underlying_ticker: Optional[str] = None   # if different from hedge_ticker
    # Threshold overrides (optional)
    roll_dte_threshold: int = 21
    delta_drift_threshold: float = 0.20
    spot_move_threshold: float = 0.05


# ── Log endpoints ─────────────────────────────────────────────────────────────

@router.get("/training/log")
async def training_log_all(limit: int = 100):
    """Return the last `limit` training log entries across both models."""
    from backend.ml.regime.training_log import TrainingLog
    return {"entries": TrainingLog.tail(limit)}


@router.get("/training/log/{model}")
async def training_log_by_model(model: Literal["main", "fast"], limit: int = 50):
    """Return the last `limit` entries for a specific model."""
    from backend.ml.regime.training_log import TrainingLog
    entries = TrainingLog.by_model(model)
    return {"model": model, "entries": entries[-limit:]}


@router.get("/training/status")
async def training_status():
    """
    Returns current model ages, last training timestamps,
    VIX thresholds, and next scheduled retrain times.
    """
    from backend.ml.regime.main_detector import MainRegimeDetector
    from backend.ml.regime.fast_detector import FastRegimeDetector
    from backend.ml.regime.training_log  import TrainingLog

    main_det = MainRegimeDetector()
    fast_det = FastRegimeDetector()

    def _status(det):
        age_h = det.model_age_hours()
        exists = det.model_exists()
        last_entries = TrainingLog.by_model(det.model_prefix)
        last = last_entries[-1] if last_entries else None
        return {
            "model_exists":        exists,
            "model_age_hours":     round(age_h, 2) if age_h != float("inf") else None,
            "is_stale":            det.is_stale(),
            "max_age_hours":       det.max_age_hours,
            "training_years":      det.training_years,
            "vix_thresholds":      det.vix_thresholds if exists and det.load() else None,
            "last_training":       last,
            "total_training_runs": len(last_entries),
        }

    return {
        "main": _status(main_det),
        "fast": _status(fast_det),
        "log_path": TrainingLog.log_path(),
    }


# ── Force-retrain endpoints ───────────────────────────────────────────────────

@router.post("/retrain/main")
async def retrain_main():
    """Force-retrain the main (10yr) model. Runs synchronously — may take 15-30s."""
    try:
        from backend.ml.regime.main_detector import MainRegimeDetector
        from backend.ml.regime.training_log  import TrainingLog
        det     = MainRegimeDetector()
        summary = await asyncio.to_thread(det.train, "manual")
        TrainingLog.append(summary)
        return {"status": "ok", "summary": summary}
    except Exception as exc:
        logger.exception("Main model retrain failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retrain/fast")
async def retrain_fast():
    """Force-retrain the fast (6mo) model. Runs synchronously — may take 5-10s."""
    try:
        from backend.ml.regime.fast_detector import FastRegimeDetector
        from backend.ml.regime.training_log  import TrainingLog
        det     = FastRegimeDetector()
        summary = await asyncio.to_thread(det.train, "manual")
        TrainingLog.append(summary)
        return {"status": "ok", "summary": summary}
    except Exception as exc:
        logger.exception("Fast model retrain failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Gap 4: Hedge monitor endpoints ───────────────────────────────────────────

@router.post("/hedge/check-triggers")
async def check_hedge_triggers(req: HedgeCheckRequest):
    """
    Evaluate re-hedge triggers for an active hedge position.

    Returns a list of alerts indicating whether the hedge needs to be:
      - Rolled (DTE < threshold)
      - Re-balanced (delta drift > threshold)
      - Re-evaluated (underlying spot moved > threshold)
      - Closed / adjusted (earnings IV crush risk)

    All checks run against live market data.

    Example request:
        {
            "hedge_ticker":    "QQQ",
            "option_type":     "put",
            "strike":          420.0,
            "expiry_date":     "2026-06-20",
            "n_contracts":     5,
            "entry_spot":      445.0,
            "initial_delta":   -0.35
        }
    """
    try:
        from backend.core.risk.hedge_monitor import HedgeMonitor

        monitor = HedgeMonitor()
        alerts  = await asyncio.to_thread(
            monitor.check_triggers,
            hedge_ticker            = req.hedge_ticker,
            option_type             = req.option_type,
            strike                  = req.strike,
            expiry_str              = req.expiry_date,
            n_contracts             = req.n_contracts,
            entry_spot              = req.entry_spot,
            initial_delta           = req.initial_delta,
            underlying_ticker       = req.underlying_ticker,
            roll_dte_threshold      = req.roll_dte_threshold,
            delta_drift_threshold   = req.delta_drift_threshold,
            spot_move_threshold     = req.spot_move_threshold,
        )

        return {
            "hedge_ticker": req.hedge_ticker,
            "strike":       req.strike,
            "expiry_date":  req.expiry_date,
            "checked_at":   date.today().isoformat(),
            "n_alerts":     len(alerts),
            "alerts":       alerts,
            "status":       "action_required" if any(a["severity"] in ("medium", "high") for a in alerts)
                            else "ok",
        }
    except Exception as exc:
        logger.exception("Hedge trigger check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/earnings/{ticker}")
async def get_earnings_date(ticker: str):
    """
    Return the next known earnings date for a ticker.

    Useful for front-end to flag positions before recommending a hedge expiry.

    Returns:
        { "ticker": "AAPL", "next_earnings": "2026-07-31" }  or null if unknown.
    """
    try:
        from backend.data.earnings_calendar import get_next_earnings_date
        ed = await asyncio.to_thread(get_next_earnings_date, ticker.upper())
        return {"ticker": ticker.upper(), "next_earnings": ed}
    except Exception as exc:
        logger.exception("Earnings date lookup failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))
