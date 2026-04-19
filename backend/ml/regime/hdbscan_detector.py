"""
HDBSCAN-based market regime detector — base class.

Architecture (hybrid):
  Regime label  → VIX percentile thresholds computed from training data.
                  Robust across different data windows; thresholds update
                  on each retrain so they reflect the current period.
  Anomaly flag  → HDBSCAN outlier score ≥ ANOMALY_SCORE_THRESHOLD.
                  Catches never-seen-before market conditions.
  Soft membership → HDBSCAN approximate_predict.

Subclasses (MainRegimeDetector, FastRegimeDetector) configure:
  - model_prefix  : file prefix for persisted artefacts
  - training_years: rolling data window
  - max_age_hours : how stale the model can be before retraining

7 features:
    vix_level, vix_5d_change, realized_vol_20d, spy_return_5d,
    vix_term_slope, tlt_return_5d, hyg_return_5d
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
import hdbscan
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass
class RegimeResult:
    regime_label: str        # "low_vol" | "mid_vol" | "high_vol" | "anomaly"
    regime_id: int           # raw HDBSCAN cluster id (-1 = noise)
    is_anomaly: bool

    # Two independent anomaly signals — combined_detector chooses which to trust
    vix_anomaly_score: float    # 0–1 from VIX percentile position (always reliable)
    hdbscan_anomaly_score: float # 0–1 from 1-HDBSCAN_strength (reliable only when clusters are dense)
    anomaly_score: float         # final blended score used by this model

    soft_membership: list
    n_discovered_regimes: int
    vix_level: float
    vol_forecast: float
    realized_vol_20d: float
    confidence: float        # 1 - anomaly_score
    model_type: str = "base" # "main" | "fast"


class HDBSCANRegimeDetector:
    """
    Parameterised HDBSCAN regime detector.
    Instantiate via MainRegimeDetector or FastRegimeDetector.
    """

    VIX_PERCENTILES        = (40, 75, 90)   # p40 = low/mid, p75 = mid/high, p90 = high/anomaly
    ANOMALY_SCORE_THRESHOLD = 0.70

    def __init__(
        self,
        model_prefix: str  = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: int      = 2,
        training_years: float = 5.0,
        max_age_hours: int    = 24,
        vix_signal_weight: float = 0.75,
    ):
        self.model_prefix      = model_prefix
        self.min_cluster_size  = min_cluster_size
        self.min_samples       = min_samples
        self.training_years    = training_years
        self.max_age_hours     = max_age_hours
        self.vix_signal_weight = vix_signal_weight  # weight given to VIX-based vs HDBSCAN anomaly signal

        # Derived paths
        self._clusterer_path  = os.path.join(MODEL_DIR, f"{model_prefix}_clusterer.joblib")
        self._scaler_path     = os.path.join(MODEL_DIR, f"{model_prefix}_scaler.joblib")
        self._thresholds_path = os.path.join(MODEL_DIR, f"{model_prefix}_vix_thresholds.joblib")
        self._trained_at_path = os.path.join(MODEL_DIR, f"{model_prefix}_trained_at.txt")

        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.scaler: Optional[RobustScaler]        = None
        self.vix_thresholds: Dict[str, float]      = {}

    # ── Model age ─────────────────────────────────────────────────────────────

    def model_age_hours(self) -> float:
        if not os.path.exists(self._trained_at_path):
            return float("inf")
        with open(self._trained_at_path) as f:
            return (time.time() - float(f.read().strip())) / 3600

    def is_stale(self) -> bool:
        return self.model_age_hours() >= self.max_age_hours

    def model_exists(self) -> bool:
        return all(os.path.exists(p) for p in [
            self._clusterer_path, self._scaler_path, self._thresholds_path
        ])

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _dl_close(ticker: str, period: str) -> pd.Series:
        import yfinance as yf
        raw   = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.squeeze()

    def _yf_period(self) -> str:
        """Convert training_years (float) to a valid yfinance period string."""
        y = self.training_years
        if y <= 0.25:  return "3mo"
        if y <= 0.5:   return "6mo"
        if y <= 1.0:   return "1y"
        if y <= 2.0:   return "2y"
        if y <= 5.0:   return "5y"
        return "10y"

    def _fetch_training_data(self) -> pd.DataFrame:
        period = self._yf_period()
        spy   = self._dl_close("SPY",   period)
        vix   = self._dl_close("^VIX",  period)
        tlt   = self._dl_close("TLT",   period)
        hyg   = self._dl_close("HYG",   period)
        vix3m = self._dl_close("^VIX3M", period)
        return pd.DataFrame({
            "spy_close": spy, "vix_level": vix,
            "tlt_close": tlt, "hyg_close": hyg, "vix3m": vix3m,
        }).dropna()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        feat["vix_level"]        = df["vix_level"]
        feat["vix_5d_change"]    = df["vix_level"].diff(5)
        spy_ret = np.log(df["spy_close"] / df["spy_close"].shift(1))
        feat["realized_vol_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
        feat["spy_return_5d"]    = df["spy_close"].pct_change(5)
        feat["vix_term_slope"]   = df["vix3m"] - df["vix_level"]
        feat["tlt_return_5d"]    = df["tlt_close"].pct_change(5)
        feat["hyg_return_5d"]    = df["hyg_close"].pct_change(5)
        return feat.dropna()

    # ── VIX → regime label ────────────────────────────────────────────────────

    def _vix_to_regime(self, vix: float) -> str:
        t = self.vix_thresholds
        if not t:                         return "mid_vol"
        if vix >= t["p90"]:               return "high_vol"
        if vix >= t["p75"]:               return "high_vol"
        if vix >= t["p40"]:               return "mid_vol"
        return "low_vol"

    def _vix_to_anomaly_score(self, vix: float) -> float:
        """
        Map raw VIX level to a 0–1 anomaly score using the training-data
        percentile thresholds.  Always reliable — no dependence on cluster density.

        Piecewise linear:
          vix < p40          → 0.00            (clearly normal)
          p40 ≤ vix < p75    → 0.00 – 0.30    (elevated but not alarming)
          p75 ≤ vix < p90    → 0.30 – 0.70    (high-vol territory)
          vix ≥ p90          → 0.70 – 1.00    (extreme; caps at 1.0)
        """
        t = self.vix_thresholds
        if not t:
            return 0.3  # no thresholds yet — conservative neutral
        p40, p75, p90 = t["p40"], t["p75"], t["p90"]
        if vix < p40:
            return 0.0
        if vix < p75:
            return 0.30 * (vix - p40) / max(p75 - p40, 1e-6)
        if vix < p90:
            return 0.30 + 0.40 * (vix - p75) / max(p90 - p75, 1e-6)
        # Above p90 — scale from 0.70 toward 1.0 over another p90-wide band
        return min(1.0, 0.70 + 0.30 * (vix - p90) / max(p90 * 0.30, 1e-6))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, trigger: str = "manual") -> Dict:
        """
        Full training cycle. Returns a summary dict (also passed to TrainingLog).
        `trigger`: "scheduled" | "manual" | "startup"
        """
        t_start = time.perf_counter()
        logger.info("[%s] Training started (trigger=%s, window=%.1fy)",
                    self.model_prefix, trigger, self.training_years)

        raw_df  = self._fetch_training_data()
        feat_df = self._build_features(raw_df)
        X       = feat_df.values

        # VIX percentile thresholds
        vix_series = feat_df["vix_level"].values
        p40, p75, p90 = np.percentile(vix_series, list(self.VIX_PERCENTILES))
        self.vix_thresholds = {"p40": float(p40), "p75": float(p75), "p90": float(p90)}

        # Scale
        self.scaler = RobustScaler()
        X_scaled    = self.scaler.fit_transform(X)

        # HDBSCAN
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=0.3,
            prediction_data=True,
        )
        self.clusterer.fit(X_scaled)

        labels    = self.clusterer.labels_
        n_regimes = len(set(labels[labels != -1]))
        noise_pct = round(100 * (labels == -1).mean(), 2)

        # Persist
        joblib.dump(self.clusterer,      self._clusterer_path)
        joblib.dump(self.scaler,         self._scaler_path)
        joblib.dump(self.vix_thresholds, self._thresholds_path)
        with open(self._trained_at_path, "w") as f:
            f.write(str(time.time()))

        duration = round(time.perf_counter() - t_start, 2)
        summary = {
            "model_prefix":      self.model_prefix,
            "trigger":           trigger,
            "training_years":    self.training_years,
            "n_samples":         len(X),
            "n_hdbscan_clusters": n_regimes,
            "noise_pct":         noise_pct,
            "vix_thresholds":    self.vix_thresholds,
            "duration_s":        duration,
        }
        logger.info(
            "[%s] Training done in %.1fs — %d samples, thresholds: low<%.1f mid<%.1f high<%.1f",
            self.model_prefix, duration, len(X), p40, p75, p90,
        )
        return summary

    def load(self) -> bool:
        if not self.model_exists():
            return False
        self.clusterer      = joblib.load(self._clusterer_path)
        self.scaler         = joblib.load(self._scaler_path)
        self.vix_thresholds = joblib.load(self._thresholds_path)
        return True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, garch_vol_forecast: float = 0.0,
                trigger: str = "on_demand") -> RegimeResult:
        """
        Returns a RegimeResult. Retrains if model is missing or stale.
        """
        from backend.ml.regime.training_log import TrainingLog

        if not self.model_exists() or self.is_stale():
            summary = self.train(trigger="startup" if not self.model_exists() else "scheduled")
            TrainingLog.append(summary)
        else:
            logger.info("[%s] Loading cached model (age=%.1fh)",
                        self.model_prefix, self.model_age_hours())
            self.load()

        raw_df  = self._fetch_training_data()
        feat_df = self._build_features(raw_df)
        vix_level    = float(feat_df["vix_level"].iloc[-1])
        realized_vol = float(feat_df["realized_vol_20d"].iloc[-1])
        today_scaled = self.scaler.transform(feat_df.iloc[[-1]].values)

        regime_label = self._vix_to_regime(vix_level)

        try:
            labels, strengths = hdbscan.approximate_predict(self.clusterer, today_scaled)
            cluster_id = int(labels[0])
            strength   = float(strengths[0])
        except Exception:
            cluster_id, strength = -1, 0.5

        # ── Two independent anomaly signals ──────────────────────────────────
        # Signal 1: VIX percentile position (always reliable — no cluster dependency)
        vix_anomaly_score = round(self._vix_to_anomaly_score(vix_level), 4)

        # Signal 2: HDBSCAN outlier score (only reliable when clusters are dense;
        #   main 10yr model often shows strength=0 for 80%+ of points → unreliable alone)
        hdbscan_anomaly_score = round(1.0 - strength, 4)

        # Blended final score — vix_signal_weight controls how much to trust VIX vs HDBSCAN.
        #   Main model (sparse 10yr clusters): vix_signal_weight=0.80 → trust VIX heavily
        #   Fast model (dense 6mo clusters):  vix_signal_weight=0.50 → blend both equally
        w = self.vix_signal_weight
        anomaly_score = round(w * vix_anomaly_score + (1.0 - w) * hdbscan_anomaly_score, 4)

        # Anomaly flag: blended score exceeds threshold, OR VIX is in extreme territory
        vix_p90   = self.vix_thresholds.get("p90", 27.0)
        vix_p75   = self.vix_thresholds.get("p75", 21.5)
        is_anomaly = (
            vix_level >= vix_p90
            or (anomaly_score >= self.ANOMALY_SCORE_THRESHOLD and vix_level >= vix_p75)
        )
        if is_anomaly:
            regime_label = "anomaly"

        try:
            soft = hdbscan.membership_vector(self.clusterer, today_scaled)
            soft_list = soft[0].tolist() if len(soft) > 0 else []
        except Exception:
            soft_list = []

        n_clusters = len(set(self.clusterer.labels_[self.clusterer.labels_ != -1]))

        logger.info(
            "[%s] regime=%s vix=%.1f vix_score=%.3f hdbscan_score=%.3f "
            "blended=%.3f (vix_w=%.0f%%) is_anomaly=%s",
            self.model_prefix, regime_label, vix_level,
            vix_anomaly_score, hdbscan_anomaly_score,
            anomaly_score, self.vix_signal_weight * 100, is_anomaly,
        )
        return RegimeResult(
            regime_label          = regime_label,
            regime_id             = cluster_id,
            is_anomaly            = is_anomaly,
            vix_anomaly_score     = vix_anomaly_score,
            hdbscan_anomaly_score = hdbscan_anomaly_score,
            anomaly_score         = anomaly_score,
            soft_membership       = soft_list,
            n_discovered_regimes  = n_clusters,
            vix_level             = vix_level,
            vol_forecast          = garch_vol_forecast,
            realized_vol_20d      = realized_vol,
            confidence            = round(1.0 - anomaly_score, 4),
            model_type            = self.model_prefix,
        )
