"""
GMM-based market regime detector — base class.

Architecture (hybrid):
  Regime label    → VIX percentile thresholds computed from training data.
                    Robust across different data windows; thresholds update
                    on each retrain so they reflect the current period.
  Anomaly signal  → GMM uncertainty: 1 - max(component probabilities).
                    High when the current state doesn't fit any known regime.
  Soft membership → GMM predict_proba — probability of belonging to each regime.

Validated approach: Two Sigma applies GMM to multi-asset factor returns for
regime classification in production (Botte & Bao, 2021).

Subclasses (MainRegimeDetector, FastRegimeDetector) configure:
  - model_prefix  : file prefix for persisted artefacts
  - n_components  : number of GMM regimes (3 = low / mid / high vol)
  - training_years: rolling data window
  - max_age_hours : how stale the model can be before retraining

10 features:
    vix_level, vix_1d_change, vix_5d_change, realized_vol_20d, spy_return_5d,
    vix_term_slope, tlt_return_5d, hyg_return_5d, iv_rv_ratio, spy_tlt_corr_20d
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass
class RegimeResult:
    regime_label: str        # "low_vol" | "mid_vol" | "high_vol" | "anomaly"
    regime_id: int           # dominant GMM component index
    is_anomaly: bool

    # Two independent anomaly signals
    vix_anomaly_score: float   # 0–1 from VIX percentile position (always reliable)
    gmm_anomaly_score: float   # 0–1 = 1 - max(GMM component probabilities)
    anomaly_score: float       # final blended score used by this model

    soft_membership: list      # per-component probabilities from GMM
    n_discovered_regimes: int  # always = n_components for GMM
    vix_level: float
    vol_forecast: float
    realized_vol_20d: float
    confidence: float          # 1 - anomaly_score
    model_type: str = "base"   # "main" | "fast"


class GMMRegimeDetector:
    """
    Parameterised GMM regime detector.
    Instantiate via MainRegimeDetector or FastRegimeDetector.
    """

    VIX_PERCENTILES         = (40, 75, 90)
    ANOMALY_SCORE_THRESHOLD = 0.70

    def __init__(
        self,
        model_prefix: str     = "gmm",
        n_components: int     = 3,
        training_years: float = 5.0,
        max_age_hours: int    = 24,
        vix_signal_weight: float = 0.75,
    ):
        self.model_prefix      = model_prefix
        self.n_components      = n_components
        self.training_years    = training_years
        self.max_age_hours     = max_age_hours
        self.vix_signal_weight = vix_signal_weight

        self._gmm_path        = os.path.join(MODEL_DIR, f"{model_prefix}_gmm.joblib")
        self._scaler_path     = os.path.join(MODEL_DIR, f"{model_prefix}_scaler.joblib")
        self._thresholds_path = os.path.join(MODEL_DIR, f"{model_prefix}_vix_thresholds.joblib")
        self._trained_at_path = os.path.join(MODEL_DIR, f"{model_prefix}_trained_at.txt")

        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[RobustScaler] = None
        self.vix_thresholds: Dict[str, float] = {}

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
            self._gmm_path, self._scaler_path, self._thresholds_path
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
        y = self.training_years
        if y <= 0.25: return "3mo"
        if y <= 0.5:  return "6mo"
        if y <= 1.0:  return "1y"
        if y <= 2.0:  return "2y"
        if y <= 5.0:  return "5y"
        return "10y"

    def _fetch_training_data(self) -> pd.DataFrame:
        period = self._yf_period()
        spy   = self._dl_close("SPY",    period)
        vix   = self._dl_close("^VIX",   period)
        tlt   = self._dl_close("TLT",    period)
        hyg   = self._dl_close("HYG",    period)
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
        tlt_ret = np.log(df["tlt_close"] / df["tlt_close"].shift(1))
        feat["realized_vol_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
        feat["spy_return_5d"]    = df["spy_close"].pct_change(5)
        feat["vix_term_slope"]   = df["vix3m"] - df["vix_level"]
        feat["tlt_return_5d"]    = df["tlt_close"].pct_change(5)
        feat["hyg_return_5d"]    = df["hyg_close"].pct_change(5)
        feat["vix_1d_change"]    = df["vix_level"].diff(1)
        rv = feat["realized_vol_20d"].replace(0, np.nan)
        feat["iv_rv_ratio"]      = (df["vix_level"] / 100) / rv
        feat["spy_tlt_corr_20d"] = spy_ret.rolling(20).corr(tlt_ret)

        return feat.dropna()

    # ── VIX → regime label ────────────────────────────────────────────────────

    def _vix_to_regime(self, vix: float) -> str:
        t = self.vix_thresholds
        if not t:           return "mid_vol"
        if vix >= t["p75"]: return "high_vol"
        if vix >= t["p40"]: return "mid_vol"
        return "low_vol"

    def _vix_to_anomaly_score(self, vix: float) -> float:
        """
        Piecewise linear VIX → anomaly score:
          vix < p40       → 0.00
          p40–p75         → 0.00–0.30
          p75–p90         → 0.30–0.70
          vix ≥ p90       → 0.70–1.00
        """
        t = self.vix_thresholds
        if not t:
            return 0.3
        p40, p75, p90 = t["p40"], t["p75"], t["p90"]
        if vix < p40:
            return 0.0
        if vix < p75:
            return 0.30 * (vix - p40) / max(p75 - p40, 1e-6)
        if vix < p90:
            return 0.30 + 0.40 * (vix - p75) / max(p90 - p75, 1e-6)
        return min(1.0, 0.70 + 0.30 * (vix - p90) / max(p90 * 0.30, 1e-6))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, trigger: str = "manual") -> Dict:
        t_start = time.perf_counter()
        logger.info(
            "[%s] Training started (trigger=%s, window=%.1fy, n_components=%d)",
            self.model_prefix, trigger, self.training_years, self.n_components,
        )

        raw_df  = self._fetch_training_data()
        feat_df = self._build_features(raw_df)
        X       = feat_df.values

        vix_series = feat_df["vix_level"].values
        p40, p75, p90 = np.percentile(vix_series, list(self.VIX_PERCENTILES))
        self.vix_thresholds = {"p40": float(p40), "p75": float(p75), "p90": float(p90)}

        self.scaler = RobustScaler()
        X_scaled    = self.scaler.fit_transform(X)

        self.gmm = GaussianMixture(
            n_components    = self.n_components,
            covariance_type = "full",   # full covariance captures feature correlations
            n_init          = 5,        # multiple restarts to avoid local optima
            random_state    = 42,
            max_iter        = 200,
        )
        self.gmm.fit(X_scaled)

        proba           = self.gmm.predict_proba(X_scaled)
        avg_uncertainty = round(float(1.0 - proba.max(axis=1).mean()), 4)

        joblib.dump(self.gmm,            self._gmm_path)
        joblib.dump(self.scaler,         self._scaler_path)
        joblib.dump(self.vix_thresholds, self._thresholds_path)
        with open(self._trained_at_path, "w") as f:
            f.write(str(time.time()))

        duration = round(time.perf_counter() - t_start, 2)
        summary = {
            "model_prefix":    self.model_prefix,
            "trigger":         trigger,
            "training_years":  self.training_years,
            "n_samples":       len(X),
            "n_components":    self.n_components,
            "avg_uncertainty": avg_uncertainty,
            "vix_thresholds":  self.vix_thresholds,
            "duration_s":      duration,
        }
        logger.info(
            "[%s] Training done in %.1fs — %d samples, avg_uncertainty=%.3f, "
            "thresholds: low<%.1f mid<%.1f high<%.1f",
            self.model_prefix, duration, len(X), avg_uncertainty, p40, p75, p90,
        )
        return summary

    def load(self) -> bool:
        if not self.model_exists():
            return False
        self.gmm            = joblib.load(self._gmm_path)
        self.scaler         = joblib.load(self._scaler_path)
        self.vix_thresholds = joblib.load(self._thresholds_path)
        return True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, garch_vol_forecast: float = 0.0,
                trigger: str = "on_demand") -> RegimeResult:
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

        proba     = self.gmm.predict_proba(today_scaled)[0]  # shape: (n_components,)
        regime_id = int(np.argmax(proba))
        max_proba = float(proba.max())

        vix_anomaly_score = round(self._vix_to_anomaly_score(vix_level), 4)
        gmm_anomaly_score = round(1.0 - max_proba, 4)

        w = self.vix_signal_weight
        anomaly_score = round(w * vix_anomaly_score + (1.0 - w) * gmm_anomaly_score, 4)

        vix_p90 = self.vix_thresholds.get("p90", 27.0)
        vix_p75 = self.vix_thresholds.get("p75", 21.5)
        is_anomaly = (
            vix_level >= vix_p90
            or (anomaly_score >= self.ANOMALY_SCORE_THRESHOLD and vix_level >= vix_p75)
        )
        if is_anomaly:
            regime_label = "anomaly"

        logger.info(
            "[%s] regime=%s vix=%.1f vix_score=%.3f gmm_score=%.3f "
            "blended=%.3f (vix_w=%.0f%%) is_anomaly=%s",
            self.model_prefix, regime_label, vix_level,
            vix_anomaly_score, gmm_anomaly_score,
            anomaly_score, self.vix_signal_weight * 100, is_anomaly,
        )
        return RegimeResult(
            regime_label      = regime_label,
            regime_id         = regime_id,
            is_anomaly        = is_anomaly,
            vix_anomaly_score = vix_anomaly_score,
            gmm_anomaly_score = gmm_anomaly_score,
            anomaly_score     = anomaly_score,
            soft_membership   = proba.tolist(),
            n_discovered_regimes = self.n_components,
            vix_level         = vix_level,
            vol_forecast      = garch_vol_forecast,
            realized_vol_20d  = realized_vol,
            confidence        = round(1.0 - anomaly_score, 4),
            model_type        = self.model_prefix,
        )
