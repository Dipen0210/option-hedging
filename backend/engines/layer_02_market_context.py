"""
Layer 2 — Market Context Engine

Uses CombinedRegimeDetector (main 10yr + fast 6mo) to produce a single
RegimeState for the entire portfolio.

Flow:
  1. GARCH on SPY → vol forecast
  2. CombinedRegimeDetector.predict() → fused regime result
  3. Enrich with live VIX level from vix_data module
  4. Return RegimeState
"""

import logging
from backend.models.risk_models import RegimeState
from backend.models.portfolio_models import PortfolioInput

logger = logging.getLogger(__name__)


class MarketContextEngine:

    def get_regime(self, portfolio: PortfolioInput) -> RegimeState:
        garch_vol    = self._get_garch_vol()
        combined     = self._run_combined(garch_vol)
        vix_data     = self._get_vix_data()

        # Prefer live VIX fetch over the value embedded in the model result
        vix_level = vix_data.get("level", combined.vix_level)

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
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_garch_vol(self) -> float:
        try:
            from backend.ml.volatility.garch_model import get_garch_vol
            vol = get_garch_vol("SPY")
            logger.info("GARCH SPY vol forecast: %.2f%%", vol * 100)
            return vol
        except Exception as e:
            logger.warning("GARCH failed (%s) — using historical vol fallback", e)
            from backend.data.market_data import get_historical_volatility
            return get_historical_volatility("SPY")

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
            return {}
