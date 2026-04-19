"""
Training log — append-only JSONL file that records every training run.

Each line is a JSON object:
{
  "ts":             "2026-04-16T10:30:00Z",   # ISO-8601 UTC
  "model_prefix":   "main" | "fast",
  "trigger":        "startup" | "scheduled" | "manual",
  "training_years": 10.0,
  "n_samples":      2520,
  "n_hdbscan_clusters": 3,
  "noise_pct":      4.1,
  "vix_thresholds": {"p40": 16.9, "p75": 21.9, "p90": 27.0},
  "duration_s":     18.3
}

Usage:
    TrainingLog.append(summary_dict)   # called automatically by detectors
    TrainingLog.tail(n=20)             # last N entries
    TrainingLog.all()                  # full history
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models", "logs")
_LOG_PATH = os.path.join(_LOG_DIR, "training_log.jsonl")


class TrainingLog:

    @staticmethod
    def _ensure_dir() -> None:
        os.makedirs(_LOG_DIR, exist_ok=True)

    @staticmethod
    def append(summary: Dict[str, Any]) -> None:
        """Append one training run to the JSONL log."""
        TrainingLog._ensure_dir()
        entry = {
            "ts":  datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        # Coerce numpy types to plain Python so json.dumps doesn't choke
        entry = json.loads(json.dumps(entry, default=lambda x: float(x) if hasattr(x, "item") else str(x)))
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("[TrainingLog] appended entry for model=%s trigger=%s",
                    entry.get("model_prefix"), entry.get("trigger"))

    @staticmethod
    def all() -> List[Dict[str, Any]]:
        """Return full log as a list of dicts, oldest first."""
        TrainingLog._ensure_dir()
        if not os.path.exists(_LOG_PATH):
            return []
        entries = []
        with open(_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    @staticmethod
    def tail(n: int = 20) -> List[Dict[str, Any]]:
        """Return the last N log entries."""
        return TrainingLog.all()[-n:]

    @staticmethod
    def by_model(model_prefix: str) -> List[Dict[str, Any]]:
        """Return all entries for a specific model."""
        return [e for e in TrainingLog.all() if e.get("model_prefix") == model_prefix]

    @staticmethod
    def log_path() -> str:
        return _LOG_PATH
