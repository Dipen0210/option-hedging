"""
Fast regime detector — short-horizon, high-frequency model.

  Data window : 6 months (~125 trading days)
  Retrain     : every 24 hours
  Role        : catches rapid regime shifts that the main model (10yr)
                is slow to reflect — e.g. sudden VIX spikes, credit events,
                earnings surprises.  Flags anomalies relative to the
                recent environment, not the 10-year baseline.
"""

from backend.ml.regime.gmm_detector import GMMRegimeDetector


class FastRegimeDetector(GMMRegimeDetector):
    """6-month rolling GMM detector, retrained every 24 hours."""

    def __init__(self):
        super().__init__(
            model_prefix      = "fast",
            n_components      = 3,
            training_years    = 0.5,   # ~125 trading days
            max_age_hours     = 24,    # daily retrain
            vix_signal_weight = 0.50,  # 6mo data → tighter clusters → blend equally
        )
