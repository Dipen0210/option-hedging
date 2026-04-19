"""
APScheduler job definitions for regime model retraining.

Jobs:
  fast_retrain  — runs every 24 hours (daily)
  main_retrain  — runs on the 1st of every month

Both jobs:
  - Run in a ThreadPoolExecutor (training is CPU/IO-bound, not async-safe)
  - Write to TrainingLog on success
  - Log errors without crashing the server

FastAPI lifespan integration (call from main.py):

    from backend.ml.regime.scheduler import build_scheduler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = build_scheduler()
        scheduler.start()
        yield
        scheduler.shutdown(wait=False)
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


def _job_retrain_fast() -> None:
    try:
        from backend.ml.regime.fast_detector import FastRegimeDetector
        from backend.ml.regime.training_log import TrainingLog
        logger.info("[Scheduler] Fast model retrain starting...")
        det     = FastRegimeDetector()
        summary = det.train(trigger="scheduled")
        TrainingLog.append(summary)
        logger.info("[Scheduler] Fast model retrain complete.")
    except Exception:
        logger.exception("[Scheduler] Fast model retrain FAILED")


def _job_retrain_main() -> None:
    try:
        from backend.ml.regime.main_detector import MainRegimeDetector
        from backend.ml.regime.training_log import TrainingLog
        logger.info("[Scheduler] Main model retrain starting...")
        det     = MainRegimeDetector()
        summary = det.train(trigger="scheduled")
        TrainingLog.append(summary)
        logger.info("[Scheduler] Main model retrain complete.")
    except Exception:
        logger.exception("[Scheduler] Main model retrain FAILED")


def build_scheduler() -> BackgroundScheduler:
    """
    Builds and returns a configured BackgroundScheduler (not yet started).
    Call .start() in the FastAPI lifespan, .shutdown() on teardown.
    """
    executors = {"default": ThreadPoolExecutor(max_workers=2)}
    scheduler = BackgroundScheduler(executors=executors, timezone="UTC")

    # Fast model — every 24 hours
    scheduler.add_job(
        _job_retrain_fast,
        trigger   = IntervalTrigger(hours=24),
        id        = "fast_retrain",
        name      = "Fast regime model — daily retrain",
        replace_existing = True,
    )

    # Main model — 1st of every month at 02:00 UTC
    scheduler.add_job(
        _job_retrain_main,
        trigger   = CronTrigger(day=1, hour=2, minute=0, timezone="UTC"),
        id        = "main_retrain",
        name      = "Main regime model — monthly retrain",
        replace_existing = True,
    )

    logger.info("[Scheduler] Jobs registered: fast(24h) + main(monthly day=1 02:00 UTC)")
    return scheduler
