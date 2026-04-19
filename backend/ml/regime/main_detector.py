"""
Main regime detector — long-horizon, slow-moving model.

  Data window : 10 years of daily macro data
  Retrain     : monthly (every 30 days)
  Role        : establishes stable, long-term VIX regime boundaries;
                thresholds are anchored in a full market cycle
                (bull + bear + recovery).
"""

from backend.ml.regime.hdbscan_detector import HDBSCANRegimeDetector


class MainRegimeDetector(HDBSCANRegimeDetector):
    """10-year rolling HDBSCAN detector, retrained once per month."""

    def __init__(self):
        super().__init__(
            model_prefix       = "main",
            min_cluster_size   = 15,   # larger clusters needed for 10yr dataset
            min_samples        = 5,
            training_years     = 10.0,
            max_age_hours      = 24 * 30,  # ~1 month
            vix_signal_weight  = 0.80,  # 10yr data → sparse clusters → trust VIX heavily
        )
