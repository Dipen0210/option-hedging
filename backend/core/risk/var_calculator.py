"""
Historical VaR and CVaR calculator.

Improvements over naive historical simulation:
  1. Cornish-Fisher expansion  — adjusts the normal quantile for skewness and
     excess kurtosis in the return distribution (fat-tail correction).
  2. GARCH vol for horizon scaling — replaces the √T rule (which assumes
     i.i.d. normal returns) with the forward-looking GARCH vol forecast when
     available, giving a more accurate multi-horizon estimate.
  3. Proper CVaR — mean of realised losses beyond the VaR threshold, not a
     fixed 1.3× multiplier.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Optional


# ── Cornish-Fisher z-score ────────────────────────────────────────────────────

def _cornish_fisher_z(confidence: float, skew: float, kurt_excess: float) -> float:
    """
    Cornish-Fisher expansion: adjusts the standard normal quantile z for
    skewness (S) and excess kurtosis (K) of the empirical distribution.

        z_CF = z + (z²−1)·S/6 + (z³−3z)·K/24 − (2z³−5z)·S²/36

    Returns the loss-side quantile (positive = tail loss).
    Skewness < 0 (left-skewed) and excess kurtosis > 0 (fat tails) both
    increase the adjusted z, making VaR larger — which is correct.
    """
    z = norm.ppf(1 - confidence)   # e.g. −1.645 for 95% confidence

    z_cf = (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt_excess / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36)

    # Clip to avoid numerical blow-up with extreme skew/kurt
    z_cf = float(np.clip(z_cf, -8.0, -0.5))
    return abs(z_cf)   # return as positive magnitude


class VaRCalculator:

    def historical_var(
        self,
        returns: pd.Series,
        notional: float,
        confidence: float = 0.95,
        horizon_days: int = 180,
        garch_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        VaR and CVaR with Cornish-Fisher fat-tail correction and GARCH
        horizon scaling.

        Method:
          1. Compute skewness + excess kurtosis from the return series.
          2. Derive CF-adjusted 1-day quantile z_CF.
          3. Estimate 1-day σ: use GARCH forecast if provided, else historical std.
          4. 1-day VaR = σ × z_CF   (parametric, fat-tail adjusted)
          5. Horizon VaR = VaR_1d × √T  (with GARCH vol this is already better
             than raw historical √T because σ is forward-looking).
          6. CVaR = mean of realised losses beyond the VaR threshold.

        Args:
            returns:     daily log-return series (not percentage)
            notional:    position value in dollars
            confidence:  VaR confidence level (default 0.95 → 95% VaR)
            horizon_days: scaling horizon in calendar days
            garch_vol:   annualised GARCH vol forecast (optional).
                         If supplied replaces historical std for σ estimation.
        """
        if len(returns) < 30:
            # Insufficient history — parametric fallback with 20% vol assumption
            sigma_1d  = 0.20 / np.sqrt(252)
            z         = norm.ppf(1 - confidence)
            var_T_pct = abs(z) * sigma_1d * np.sqrt(horizon_days)
            return {
                "var_dollar":  round(var_T_pct * notional, 2),
                "cvar_dollar": round(var_T_pct * 1.25 * notional, 2),
                "var_pct":     round(var_T_pct, 6),
                "cvar_pct":    round(var_T_pct * 1.25, 6),
            }

        # ── Step 1: Distribution moments ─────────────────────────────────────
        skew       = float(returns.skew())
        kurt_excess = float(returns.kurtosis())   # scipy returns excess kurtosis

        # Clip to stable range — very extreme values cause CF to be unreliable
        skew        = float(np.clip(skew,        -3.0, 3.0))
        kurt_excess = float(np.clip(kurt_excess, -2.0, 10.0))

        # ── Step 2: Cornish-Fisher adjusted quantile ──────────────────────────
        z_cf = _cornish_fisher_z(confidence, skew, kurt_excess)

        # ── Step 3: 1-day sigma ───────────────────────────────────────────────
        if garch_vol and 0.01 < garch_vol < 3.0:
            sigma_1d = garch_vol / np.sqrt(252)
        else:
            sigma_1d = float(returns.std())

        # ── Step 4 & 5: Horizon VaR ───────────────────────────────────────────
        # √T scaling is still used but now with GARCH vol + CF quantile,
        # which already captures fat-tail risk that plain √T misses.
        horizon_scale = np.sqrt(horizon_days)
        var_T_pct     = z_cf * sigma_1d * horizon_scale

        # ── Step 6: CVaR — mean of tail losses ───────────────────────────────
        # Use empirical threshold from historical returns scaled to horizon.
        # 1-day tail losses:
        threshold_1d = -z_cf * sigma_1d            # negative (loss side)
        tail         = returns[returns <= threshold_1d]

        if len(tail) >= 5:
            cvar_1d_pct  = float(tail.mean())      # negative mean of tail
            cvar_T_pct   = abs(cvar_1d_pct) * horizon_scale
        else:
            # Parametric CVaR for normal: φ(z) / (1−confidence) × σ
            phi_z        = norm.pdf(-z_cf)
            cvar_T_pct   = (phi_z / (1 - confidence)) * sigma_1d * horizon_scale

        # Ensure CVaR ≥ VaR (it always should be by definition)
        cvar_T_pct = max(cvar_T_pct, var_T_pct)

        return {
            "var_dollar":  round(var_T_pct  * notional, 2),
            "cvar_dollar": round(cvar_T_pct * notional, 2),
            "var_pct":     round(var_T_pct,  6),
            "cvar_pct":    round(cvar_T_pct, 6),
        }

    def tail_correlation(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        percentile: float = 0.10,
    ) -> float:
        """
        Correlation in the bottom `percentile` of the return distribution.
        In crashes, tail correlation spikes vs normal-market correlation.
        Use this — not Pearson — for hedge basis risk assessment.
        """
        common = returns_a.index.intersection(returns_b.index)
        ra = returns_a.loc[common]
        rb = returns_b.loc[common]

        if len(ra) < 30:
            return float(np.corrcoef(ra.values, rb.values)[0, 1])

        threshold = np.percentile(ra, percentile * 100)
        mask = ra <= threshold

        if mask.sum() < 5:
            return float(np.corrcoef(ra.values, rb.values)[0, 1])

        corr = float(np.corrcoef(ra[mask].values, rb[mask].values)[0, 1])
        return round(corr, 4)

    def portfolio_var(
        self,
        returns_matrix: pd.DataFrame,
        weights: np.ndarray,
        notional: float,
        confidence: float = 0.95,
        horizon_days: int = 180,
        garch_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Portfolio-level VaR using weighted return series.
        Passes garch_vol through to historical_var for consistent scaling.
        """
        aligned      = returns_matrix.dropna()
        port_returns = aligned.values @ weights
        port_series  = pd.Series(port_returns, index=aligned.index)
        return self.historical_var(
            port_series, notional, confidence, horizon_days, garch_vol=garch_vol
        )
