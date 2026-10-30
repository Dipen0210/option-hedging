"""
VaR and CVaR calculator — parametric Cornish-Fisher with an empirical tail.

NAMING: the primary estimator is *parametric*, not historical simulation. It
takes σ and the higher moments from the sample, then evaluates a closed-form
adjusted quantile; it does not read a loss off the empirical return
distribution. The method was previously called `historical_var`, which
misdescribed it to anyone reading a risk number — it is now
`parametric_cf_var`, with `historical_var` retained as a deprecated alias so
existing callers keep working.

Method:
  1. Cornish-Fisher expansion  — adjusts the normal quantile for skewness and
     excess kurtosis in the return distribution (fat-tail correction).
  2. GARCH vol for horizon scaling — replaces the sample std with the
     forward-looking GARCH vol forecast when available.
  3. CVaR — mean of realised losses beyond the VaR threshold where enough tail
     observations exist, else the closed-form normal ES, not a fixed multiplier.

HORIZON CONVENTION: callers pass a hedge horizon in **calendar** days (that is
what a user picks: "protect me for 180 days"). σ is a **trading-day** quantity,
so √T scaling must use trading days. Scaling a trading-day σ by √(calendar
days) overstated 180-day VaR by ~20%. `_trading_days()` does the conversion.
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Optional

from backend.data import data_quality as dq

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25

# Below this many observations the sample moments (especially kurtosis) are not
# estimable, so Cornish-Fisher cannot be applied.
MIN_OBS_FOR_MOMENTS = 30


def _trading_days(calendar_days: float) -> float:
    """
    Convert a calendar-day horizon to trading days.

    σ is measured per trading day, so √T scaling must count trading days.
    Always returns at least 1 — a sub-one-day horizon still carries one day of
    risk, and returning <1 would shrink VaR below its single-day floor.
    """
    return max(1.0, calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)


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

    def parametric_cf_var(
        self,
        returns: pd.Series,
        notional: float,
        confidence: float = 0.95,
        horizon_days: int = 180,
        garch_vol: Optional[float] = None,
        label: str = "",
    ) -> Dict[str, float]:
        """
        VaR and CVaR with Cornish-Fisher fat-tail correction and GARCH
        horizon scaling.

        Method:
          1. Compute skewness + excess kurtosis from the return series.
          2. Derive CF-adjusted 1-day quantile z_CF.
          3. Estimate 1-day σ: use GARCH forecast if provided, else sample std.
          4. 1-day VaR = σ × z_CF   (parametric, fat-tail adjusted)
          5. Horizon VaR = VaR_1d × √(trading days in the horizon).
          6. CVaR = mean of realised losses beyond the VaR threshold, else the
             closed-form normal ES.

        Args:
            returns:      daily log-return series (not percentage)
            notional:     position value in dollars
            confidence:   VaR confidence level (default 0.95 → 95% VaR)
            horizon_days: hedge horizon in CALENDAR days (converted internally)
            garch_vol:    annualised GARCH vol forecast (optional).
                          If supplied, replaces sample std for σ estimation.
            label:        identifier used in data-quality disclosures.

        Returns a dict with var/cvar in dollars and as a fraction of notional,
        plus `method`, `sigma_source` and `horizon_trading_days` so a caller
        can tell a fully-estimated number from a fallback one.
        """
        horizon_td    = _trading_days(horizon_days)
        horizon_scale = np.sqrt(horizon_td)
        who           = f"var:{label}" if label else "var"

        if len(returns) < MIN_OBS_FOR_MOMENTS:
            # Too few observations to estimate skew/kurtosis. The assumed vol
            # below is NOT a measurement — it is disclosed as a fallback so the
            # API can tell the user this VaR is an assumption, not an estimate.
            assumed_annual_vol = 0.20
            sigma_1d  = assumed_annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
            z         = abs(norm.ppf(1 - confidence))
            var_T_pct = z * sigma_1d * horizon_scale
            # Normal-distribution ES/VaR ratio at this confidence — consistent
            # with the assumed-normal σ above, instead of an arbitrary 1.25×.
            es_ratio  = float(norm.pdf(z) / (1 - confidence) / z)
            logger.warning(
                "VaR for %s: only %d observations (need %d) — using assumed "
                "%.0f%% annual vol; this is an assumption, not an estimate",
                label or "position", len(returns), MIN_OBS_FOR_MOMENTS,
                assumed_annual_vol * 100,
            )
            dq.record(
                who, dq.SOURCE_FALLBACK,
                detail=(
                    f"only {len(returns)} return observations (need "
                    f"{MIN_OBS_FOR_MOMENTS}); VaR assumes "
                    f"{assumed_annual_vol:.0%} annual vol"
                ),
            )
            return {
                "var_dollar":  round(var_T_pct * notional, 2),
                "cvar_dollar": round(var_T_pct * es_ratio * notional, 2),
                "var_pct":     round(var_T_pct, 6),
                "cvar_pct":    round(var_T_pct * es_ratio, 6),
                "method":      "assumed_vol_fallback",
                "sigma_source": f"assumed {assumed_annual_vol:.0%} annual",
                "horizon_trading_days": round(horizon_td, 2),
                "n_observations": int(len(returns)),
            }

        # ── Step 1: Distribution moments ─────────────────────────────────────
        skew       = float(returns.skew())
        kurt_excess = float(returns.kurtosis())   # pandas: Fisher (excess) kurtosis

        # Clip to stable range — very extreme values cause CF to be unreliable
        skew        = float(np.clip(skew,        -3.0, 3.0))
        kurt_excess = float(np.clip(kurt_excess, -2.0, 10.0))

        # ── Step 2: Cornish-Fisher adjusted quantile ──────────────────────────
        z_cf = _cornish_fisher_z(confidence, skew, kurt_excess)

        # ── Step 3: 1-day sigma ───────────────────────────────────────────────
        if garch_vol and 0.01 < garch_vol < 3.0:
            sigma_1d = garch_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
            sigma_source = "garch_forecast"
        else:
            sigma_1d = float(returns.std())
            sigma_source = "sample_std"
            if garch_vol:
                dq.record(who, dq.SOURCE_FALLBACK,
                          detail=(f"GARCH vol {garch_vol:.3f} outside plausible "
                                  f"range; used sample std"))

        if not np.isfinite(sigma_1d) or sigma_1d <= 0:
            raise ValueError(
                f"Cannot compute VaR for {label or 'position'}: daily volatility "
                f"estimate is {sigma_1d!r}"
            )

        # ── Step 4 & 5: Horizon VaR ───────────────────────────────────────────
        # σ is per trading day, so √T counts TRADING days, not calendar days.
        var_T_pct = z_cf * sigma_1d * horizon_scale

        # ── Step 6: CVaR — mean of tail losses ───────────────────────────────
        # Use empirical threshold from historical returns scaled to horizon.
        # 1-day tail losses:
        threshold_1d = -z_cf * sigma_1d            # negative (loss side)
        tail         = returns[returns <= threshold_1d]

        if len(tail) >= 5:
            cvar_1d_pct  = float(tail.mean())      # negative mean of tail
            cvar_T_pct   = abs(cvar_1d_pct) * horizon_scale
            cvar_source  = f"empirical_tail_n{len(tail)}"
        else:
            # Closed-form CVaR for normal: φ(z) / (1−confidence) × σ
            phi_z        = norm.pdf(-z_cf)
            cvar_T_pct   = (phi_z / (1 - confidence)) * sigma_1d * horizon_scale
            cvar_source  = "normal_closed_form"

        # Ensure CVaR ≥ VaR (it always should be by definition)
        cvar_T_pct = max(cvar_T_pct, var_T_pct)

        return {
            "var_dollar":  round(var_T_pct  * notional, 2),
            "cvar_dollar": round(cvar_T_pct * notional, 2),
            "var_pct":     round(var_T_pct,  6),
            "cvar_pct":    round(cvar_T_pct, 6),
            "method":      "parametric_cornish_fisher",
            "sigma_source": sigma_source,
            "cvar_source": cvar_source,
            "horizon_trading_days": round(horizon_td, 2),
            "n_observations": int(len(returns)),
        }

    # Deprecated alias. The estimator is parametric, not historical simulation;
    # kept so existing callers and tests keep working.
    def historical_var(self, *args, **kwargs) -> Dict[str, float]:
        return self.parametric_cf_var(*args, **kwargs)

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
        Passes garch_vol through to parametric_cf_var for consistent scaling.
        """
        aligned      = returns_matrix.dropna()
        port_returns = aligned.values @ weights
        port_series  = pd.Series(port_returns, index=aligned.index)
        return self.parametric_cf_var(
            port_series, notional, confidence, horizon_days,
            garch_vol=garch_vol, label="portfolio",
        )
