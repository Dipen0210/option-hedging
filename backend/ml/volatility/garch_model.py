"""
GARCH(1,1) volatility forecaster.
σ²t = ω + α·ε²(t-1) + β·σ²(t-1)

Used to get forward-looking volatility estimate for BSM pricing (Phase 4).
Also passed into RegimeResult as vol_forecast_garch.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional


class GARCHVolModel:

    def __init__(self):
        self.result = None
        self._ticker: Optional[str] = None

    def fit(self, returns: pd.Series) -> None:
        """
        Fit GARCH(1,1) on a returns series.
        Input: daily log returns (not percentage).
        """
        ret_pct = returns * 100   # scale for numerical stability
        model = arch_model(ret_pct, vol="Garch", p=1, q=1, dist="normal")
        self.result = model.fit(disp="off", show_warning=False)

    def forecast_vol(self, horizon: int = 1) -> float:
        """
        Returns annualized volatility forecast for next `horizon` days.
        """
        if self.result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        forecast = self.result.forecast(horizon=horizon, reindex=False)
        var_forecast = float(forecast.variance.values[-1, -1])
        daily_vol = np.sqrt(var_forecast) / 100.0
        return round(daily_vol * np.sqrt(252), 6)

    def current_conditional_vol(self) -> float:
        """Today's conditional volatility (annualized)."""
        if self.result is None:
            raise RuntimeError("Model not fitted.")
        daily_vol = float(np.sqrt(self.result.conditional_volatility.iloc[-1])) / 100.0
        return round(daily_vol * np.sqrt(252), 6)

    def fit_and_forecast(self, returns: pd.Series, horizon: int = 1) -> float:
        """Convenience: fit then return forecast in one call."""
        self.fit(returns)
        return self.forecast_vol(horizon)


def get_garch_vol(ticker: str) -> float:
    """
    Fetches 2Y returns for ticker, fits GARCH, returns annualized vol forecast.
    Convenience function used by Layer 2 (market context).
    """
    from backend.data.market_data import get_returns
    returns = get_returns(ticker, period="2y")
    model = GARCHVolModel()
    return model.fit_and_forecast(returns)
