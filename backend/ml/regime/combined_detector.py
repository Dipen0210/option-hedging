"""
CombinedRegimeDetector — weighted fusion of main (10yr) + fast (6mo) outputs.

Fusion strategy
───────────────
Both models produce:
  - regime_label   : "low_vol" | "mid_vol" | "high_vol" | "anomaly"
  - anomaly_score  : 0–1 blended (VIX percentile + GMM uncertainty)
  - confidence     : 1 - anomaly_score

Weights:
  MAIN_WEIGHT = 0.65  — long-term 10yr baseline, anchors regime in full cycle
  FAST_WEIGHT = 0.35  — 6mo window, more sensitive to recent regime shifts

Regime label (weighted vote):
  Each model maps its label to a numeric regime ordinal
  (low_vol=0, mid_vol=1, high_vol=2, anomaly=3) weighted by its
  confidence. The weighted average is snapped back to the nearest label.

Anomaly score (weighted average):
  combined_anomaly_score = MAIN_WEIGHT * main.anomaly_score
                         + FAST_WEIGHT * fast.anomaly_score

  is_anomaly = combined_anomaly_score >= ANOMALY_THRESHOLD (0.60)
             OR either model individually exceeds 0.85 (hard override)
"""

import logging
from dataclasses import dataclass

from backend.ml.regime.gmm_detector import RegimeResult
from backend.ml.regime.main_detector import MainRegimeDetector
from backend.ml.regime.fast_detector import FastRegimeDetector

logger = logging.getLogger(__name__)

MAIN_WEIGHT = 0.65
FAST_WEIGHT = 0.35

ANOMALY_THRESHOLD      = 0.60
HARD_ANOMALY_THRESHOLD = 0.85

_LABEL_TO_ORD: dict[str, float] = {
    "low_vol":  0.0,
    "mid_vol":  1.0,
    "high_vol": 2.0,
    "anomaly":  3.0,
}
_ORD_TO_LABEL: list[str] = ["low_vol", "mid_vol", "high_vol", "anomaly"]


def _snap_label(ordinal: float) -> str:
    idx = max(0, min(len(_ORD_TO_LABEL) - 1, round(ordinal)))
    return _ORD_TO_LABEL[idx]


@dataclass
class CombinedRegimeResult:
    """Full picture from both models with weighted fusion."""

    # ── Final fused values ────────────────────────────────────────────────
    regime_label: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float

    # ── Per-model raw results ─────────────────────────────────────────────
    main: RegimeResult
    fast: RegimeResult

    # ── Weights actually used ─────────────────────────────────────────────
    main_weight: float
    fast_weight: float

    # ── Passthrough ───────────────────────────────────────────────────────
    vix_level: float
    realized_vol_20d: float
    vol_forecast_garch: float

    # ── Diagnostic: per-model anomaly signal breakdown ────────────────────
    @property
    def main_vix_score(self) -> float:
        return self.main.vix_anomaly_score

    @property
    def fast_vix_score(self) -> float:
        return self.fast.vix_anomaly_score

    @property
    def main_gmm_score(self) -> float:
        return self.main.gmm_anomaly_score

    @property
    def fast_gmm_score(self) -> float:
        return self.fast.gmm_anomaly_score

    @property
    def fast_only_anomaly(self) -> bool:
        return self.fast.is_anomaly and not self.main.is_anomaly

    @property
    def regime_diverged(self) -> bool:
        return self.main.regime_label != self.fast.regime_label

    def summary(self) -> str:
        return (
            f"regime={self.regime_label}  anomaly={self.is_anomaly}  "
            f"score={self.anomaly_score:.3f}  conf={self.confidence:.3f}  "
            f"(main={self.main.regime_label}@{MAIN_WEIGHT} "
            f"vix={self.main.vix_anomaly_score:.3f} gmm={self.main.gmm_anomaly_score:.3f} "
            f"blended={self.main.anomaly_score:.3f})  "
            f"(fast={self.fast.regime_label}@{FAST_WEIGHT} "
            f"vix={self.fast.vix_anomaly_score:.3f} gmm={self.fast.gmm_anomaly_score:.3f} "
            f"blended={self.fast.anomaly_score:.3f})"
        )


class CombinedRegimeDetector:
    """
    Runs MainRegimeDetector + FastRegimeDetector and produces a
    weighted-ratio fused RegimeResult.
    """

    def __init__(self):
        self._main = MainRegimeDetector()
        self._fast = FastRegimeDetector()

    def predict(self, garch_vol_forecast: float = 0.0) -> CombinedRegimeResult:
        logger.info("[Combined] Running main model (weight=%.0f%%)...", MAIN_WEIGHT * 100)
        main_r: RegimeResult = self._main.predict(garch_vol_forecast)

        logger.info("[Combined] Running fast model (weight=%.0f%%)...", FAST_WEIGHT * 100)
        fast_r: RegimeResult = self._fast.predict(garch_vol_forecast)

        result = self._fuse(main_r, fast_r, garch_vol_forecast)
        logger.info("[Combined] %s", result.summary())
        return result

    def force_retrain_main(self) -> dict:
        from backend.ml.regime.training_log import TrainingLog
        summary = self._main.train(trigger="manual")
        TrainingLog.append(summary)
        return summary

    def force_retrain_fast(self) -> dict:
        from backend.ml.regime.training_log import TrainingLog
        summary = self._fast.train(trigger="manual")
        TrainingLog.append(summary)
        return summary

    @staticmethod
    def _fuse(
        main_r: RegimeResult,
        fast_r: RegimeResult,
        garch_vol: float,
    ) -> "CombinedRegimeResult":
        w_anomaly_score = round(
            MAIN_WEIGHT * main_r.anomaly_score + FAST_WEIGHT * fast_r.anomaly_score, 4
        )
        w_confidence = round(
            MAIN_WEIGHT * main_r.confidence + FAST_WEIGHT * fast_r.confidence, 4
        )

        is_anomaly = (
            w_anomaly_score >= ANOMALY_THRESHOLD
            or main_r.anomaly_score >= HARD_ANOMALY_THRESHOLD
            or fast_r.anomaly_score >= HARD_ANOMALY_THRESHOLD
        )

        main_ord = _LABEL_TO_ORD.get(main_r.regime_label, 1.0)
        fast_ord = _LABEL_TO_ORD.get(fast_r.regime_label, 1.0)

        main_eff  = MAIN_WEIGHT * main_r.confidence
        fast_eff  = FAST_WEIGHT * fast_r.confidence
        total_eff = main_eff + fast_eff

        if total_eff > 0:
            weighted_ord = (main_eff * main_ord + fast_eff * fast_ord) / total_eff
        else:
            weighted_ord = main_ord

        regime_label = "anomaly" if is_anomaly else _snap_label(weighted_ord)

        logger.debug(
            "[Combined/fuse] main=(%s, conf=%.2f) fast=(%s, conf=%.2f) "
            "→ weighted_ord=%.2f anomaly_score=%.3f is_anomaly=%s → %s",
            main_r.regime_label, main_r.confidence,
            fast_r.regime_label, fast_r.confidence,
            weighted_ord, w_anomaly_score, is_anomaly, regime_label,
        )

        return CombinedRegimeResult(
            regime_label       = regime_label,
            is_anomaly         = is_anomaly,
            anomaly_score      = w_anomaly_score,
            confidence         = w_confidence,
            main               = main_r,
            fast               = fast_r,
            main_weight        = MAIN_WEIGHT,
            fast_weight        = FAST_WEIGHT,
            vix_level          = main_r.vix_level,
            realized_vol_20d   = main_r.realized_vol_20d,
            vol_forecast_garch = garch_vol,
        )
