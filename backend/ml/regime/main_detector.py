"""
Main regime detector — long-horizon, slow-moving model.

  Data window : 10 years of daily macro data
  Retrain     : monthly (every 30 days)
  Role        : establishes stable, long-term VIX regime boundaries;
                thresholds are anchored in a full market cycle
                (bull + bear + recovery).
"""

from backend.ml.regime.gmm_detector import GMMRegimeDetector


class MainRegimeDetector(GMMRegimeDetector):
    """10-year rolling GMM detector, retrained once per month."""

    def __init__(self):
        super().__init__(
            model_prefix      = "main",
            n_components      = 4,
            training_years    = 10.0,
            max_age_hours     = 24 * 30,  # ~1 month
            vix_signal_weight = 0.65,     # GMM reliable with 4 components on 10yr data
        )
