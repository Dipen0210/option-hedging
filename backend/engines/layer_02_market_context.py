"""
Layer 2 — Market Context Engine

Uses CombinedRegimeDetector (main + fast GMM models) to produce a single
RegimeState for the entire portfolio.

Flow:
  1. GARCH on SPY → vol forecast
  2. CombinedRegimeDetector.predict() → fused regime result
  3. Enrich with live VIX level from vix_data module
  4. Return RegimeState

Degradation: regime detection is contextual, not a pricing input, so a model
failure must not fail the whole analysis. When the GMM pipeline is unavailable
this layer falls back to a VIX-threshold label and marks the RegimeState
degraded, with the reason, so the API can tell the user what it lost.
"""

import logging

from backend.models.risk_models import RegimeState
from backend.models.portfolio_models import PortfolioInput
from backend.data import data_quality as dq

logger = logging.getLogger(__name__)


class MarketContextEngine:

    def get_regime(self, portfolio: PortfolioInput) -> RegimeState:
        garch_vol = self._get_garch_vol()
        vix_data  = self._get_vix_data()

        try:
            combined = self._run_combined(garch_vol)
        except Exception as exc:
            logger.warning(
                "Regime model unavailable (%s) — falling back to VIX thresholds", exc
            )
            dq.record("regime", dq.SOURCE_FALLBACK,
                      detail=f"GMM unavailable: {str(exc)[:100]}")
            return self._vix_fallback_regime(vix_data, garch_vol, str(exc))

        dq.record("regime", dq.SOURCE_LIVE)
        # Prefer live VIX fetch over the value embedded in the model result
        vix_level = vix_data.get("level") or combined.vix_level

        return RegimeState(
            regime_label       = combined.regime_label,
            regime_id          = combined.main.regime_id,
            is_anomaly         = combined.is_anomaly,
            anomaly_score      = combined.anomaly_score,
            soft_membership    = combined.main.soft_membership,
            n_discovered_regimes = combined.main.n_discovered_regimes,
            vix_level          = vix_level,
            vol_forecast_garch = garch_vol,
            realized_vol_20d   = combined.realized_vol_20d,
            sentiment_score    = 0.0,
            regime_source      = "gmm_combined",
            degraded           = False,
        )

    # ── Degraded path ─────────────────────────────────────────────────────────

    def _vix_fallback_regime(
        self, vix_data: dict, garch_vol: float, reason: str
    ) -> RegimeState:
        """
        Label the regime from the live VIX level alone.

        This is a real, documented methodology (static VIX bands) rather than an
        invented value — but it loses the trained percentile thresholds and the
        GMM anomaly signal, so the result is explicitly marked degraded.
        """
        from backend.data.vix_data import classify_vix_regime

        vix_level = vix_data.get("level")
        if vix_level is None:
            raise RuntimeError(
                "Market regime unavailable: both the GMM models and the live VIX "
                f"quote could not be retrieved ({reason})"
            )

        label = classify_vix_regime(vix_level)
        realized = 0.0
        try:
            from backend.data.market_data import get_historical_volatility
            realized = get_historical_volatility("SPY")
        except Exception:
            pass

        return RegimeState(
            regime_label       = label,
            regime_id          = -1,
            is_anomaly         = vix_level >= 30.0,
            anomaly_score      = 0.0,
            soft_membership    = [],
            n_discovered_regimes = 0,
            vix_level          = vix_level,
            vol_forecast_garch = garch_vol,
            realized_vol_20d   = realized,
            sentiment_score    = 0.0,
            regime_source      = "vix_fallback",
            degraded           = True,
            degradation_reason = (
                "Regime model unavailable; label derived from static VIX bands "
                f"({reason[:120]})"
            ),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_garch_vol(self) -> float:
        try:
            from backend.ml.volatility.garch_model import get_garch_vol
            vol = get_garch_vol("SPY")
            logger.info("GARCH SPY vol forecast: %.2f%%", vol * 100)
            dq.record("garch:SPY", dq.SOURCE_LIVE)
            return vol
        except Exception as e:
            logger.warning("GARCH failed (%s) — using realized vol fallback", e)
            try:
                from backend.data.market_data import get_historical_volatility
                vol = get_historical_volatility("SPY")
                dq.record("garch:SPY", dq.SOURCE_FALLBACK,
                          detail=f"realized vol substituted ({str(e)[:60]})")
                return vol
            except Exception as e2:
                dq.record("garch:SPY", dq.SOURCE_UNAVAILABLE, detail=str(e2)[:80])
                return 0.0

    def _run_combined(self, garch_vol: float):
        from backend.ml.regime.combined_detector import CombinedRegimeDetector
        detector = CombinedRegimeDetector()
        result   = detector.predict(garch_vol_forecast=garch_vol)
        logger.info(
            "Regime: %s  (main=%s fast=%s  anomaly=%s  score=%.3f  confidence=%.3f)",
            result.regime_label,
            result.main.regime_label,
            result.fast.regime_label,
            result.is_anomaly,
            result.anomaly_score,
            result.confidence,
        )
        return result

    def _get_vix_data(self) -> dict:
        try:
            from backend.data.vix_data import get_vix
            return get_vix()
        except Exception as e:
            logger.warning("VIX fetch failed: %s", e)
            dq.record("vix", dq.SOURCE_UNAVAILABLE, detail=str(e)[:80])
            return {}
