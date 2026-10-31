from pydantic import BaseModel
from typing import List, Dict, Optional


class RegimeState(BaseModel):
    regime_label: str           # "low_vol" | "mid_vol" | "high_vol" | "anomaly"
    regime_id: int              # dominant GMM component index
    is_anomaly: bool
    anomaly_score: float        # blended VIX-percentile + GMM-uncertainty score
    soft_membership: List[float]  # GMM probability per component
    n_discovered_regimes: int   # number of GMM components
    vix_level: float
    vol_forecast_garch: float
    realized_vol_20d: float
    sentiment_score: float = 0.0

    # Provenance: "gmm_combined" when both models scored, "vix_fallback" when
    # the GMM pipeline was unavailable and the label came from VIX thresholds
    # alone. Surfaced to the user so a degraded regime is never mistaken for a
    # full model read.
    regime_source: str = "gmm_combined"
    degraded: bool = False
    degradation_reason: str = ""


class RiskProfile(BaseModel):
    ticker: str
    asset_class: str
    notional_value: float

    # ── Measurability ─────────────────────────────────────────────────────────
    # False when risk analysis failed for this holding and the profile is a
    # placeholder rather than a measurement. `L3._fallback_profile` used to
    # invent a complete profile — 15% VaR, 20% CVaR, 0.7 tail correlation,
    # beta 1.0 — and return it indistinguishable from a real one, so the
    # pipeline went on to build tradeable hedge recommendations out of numbers
    # nobody computed. Consumers must refuse to recommend on an unavailable
    # profile rather than treating these figures as risk.
    data_available: bool = True
    unavailable_reason: str = ""

    # ── Beta / factor exposures ───────────────────────────────────────────────
    # Optional, and None means "could not be measured" — not zero. These
    # defaulted to 0.0, which made an unmeasurable rate sensitivity
    # indistinguishable from a holding genuinely insensitive to rates, and
    # `beta_vs_spy` was substituted with 1.0 by the decomposer on the same
    # path. Both are claims about the position that feed hedge sizing, so a
    # failed measurement now reads as absent rather than as a number.
    # `beta_vs_spy` stays a float because it is consumed arithmetically
    # throughout sizing; `beta_available` is what distinguishes a measured
    # beta from the 1.0 the decomposer used to return silently. Any consumer
    # that would size a position off beta must check the flag.
    beta_vs_spy: float = 1.0
    beta_available: bool = True
    small_cap_beta: Optional[float] = None
    value_beta: Optional[float] = None
    rate_sensitivity: Optional[float] = None
    credit_spread_sensitivity: Optional[float] = None
    usd_sensitivity: Optional[float] = None
    factor_contributions: Dict[str, float] = {}

    # What the SPY regression rests on, so a beta from 120 observations is not
    # presented with the same authority as one from 1,000.
    beta_r2: Optional[float] = None
    beta_stderr: Optional[float] = None
    beta_n_obs: Optional[int] = None
    # Exposure name → why it could not be measured.
    factors_unavailable: Dict[str, str] = {}

    # VaR
    var_5pct: float             # 95% VaR in dollars
    cvar_5pct: float            # 95% CVaR (Expected Shortfall) in dollars
    var_pct: float              # VaR as % of notional
    cvar_pct: float             # CVaR as % of notional
    notional_at_risk: float

    # The horizon the VaR and CVaR figures above cover, carried with them.
    # They are scaled to the portfolio's hedge horizon — not to one day and
    # not to one month — and the dollar amount means nothing without it: the
    # same position reads about $4.6K over one day and $42.2K over a 120-day
    # horizon. Both consumers used to assert a horizon of their own instead of
    # reading one: the L11 narrative called it "One-day 95% VaR" and the UI
    # labelled it "1mo VaR (95%)", so a four-month tail loss was presented as
    # an overnight one and understated by roughly 9x. Label the figure from
    # these fields; 0 means the horizon is unknown and no horizon may be named.
    var_horizon_calendar_days: int = 0
    var_horizon_trading_days: float = 0.0

    # How the VaR above was produced, and what the volatility input was.
    # `parametric_cornish_fisher` is an estimate from this holding's own return
    # distribution; `assumed_vol_fallback` means there were too few
    # observations to estimate moments and a flat annual vol was assumed, so
    # the figure is an assumption about the position rather than a measurement
    # of it. The calculator has always distinguished the two and logged it, but
    # L3 dropped both fields on the way into this model, so the response
    # presented an assumed number and an estimated one identically.
    var_method: str = ""            # "parametric_cornish_fisher" | "assumed_vol_fallback"
    var_sigma_source: str = ""      # e.g. "garch_forecast" | "realized" | "assumed"
    var_n_obs: Optional[int] = None  # return observations behind the estimate

    # Hedge ratios
    optimal_hedge_ratio: float      # h* = ρ * (σ_S / σ_F)
    regression_hedge_ratio: float   # OLS slope
    tail_correlation: float         # bottom-10% correlation
    # Provenance of the OLS hedge ratio / basis-risk R².
    hedge_ratio_r2: Optional[float] = None
    hedge_ratio_stderr: Optional[float] = None
    hedge_ratio_n_obs: Optional[int] = None

    # Direction: +1 = net long (positive delta), -1 = net short (negative delta)
    # Stocks are always +1; options depend on direction × option_type
    effective_delta_sign: int = 1


class PortfolioGreeks(BaseModel):
    """
    Dollar-denominated aggregate Greeks for the entire portfolio.

    These capture the portfolio's net sensitivity to market moves — not the
    hedge candidates, but the user's *existing* positions.  They answer:

      dollar_delta : how much the portfolio gains/loses per 1% move in the market
      dollar_gamma : how dollar_delta changes per 1% move (convexity)
      dollar_vega  : P&L change per 1 vol-point (1%) shift in implied vol
      dollar_theta : daily time decay in dollars (options positions only)

    Stocks contribute delta = shares × spot × 1.0.
    Options contribute delta = |Δ| × 100 × spot × contracts (signed by direction).

    Used by the UI and the hedge optimizer to select the minimum-cost hedge
    set that neutralises portfolio-level exposure — not just per-position.
    """
    dollar_delta: float = 0.0       # $ P&L per 1% upward market move
    dollar_gamma: float = 0.0       # $ change in delta per 1% move (acceleration)
    dollar_vega:  float = 0.0       # $ P&L per 1 vol-point IV shift
    dollar_theta: float = 0.0       # $ daily time decay (negative = cost to hold)
    net_delta_pct: float = 0.0      # portfolio delta as % of total notional
    dominant_risk: str = "delta"    # "delta" | "vega" | "gamma" — largest exposure


class PortfolioRiskSummary(BaseModel):
    total_notional: float
    portfolio_beta: float
    portfolio_var_5pct: float
    portfolio_cvar_5pct: float
    # Horizon of the two figures above. See RiskProfile.var_horizon_*.
    var_horizon_calendar_days: int = 0
    var_horizon_trading_days: float = 0.0
    concentration_top1: float       # weight of largest holding
    risk_profiles: List[RiskProfile]
    correlation_matrix: Optional[List[List[float]]] = None
    tickers: Optional[List[str]] = None
    portfolio_greeks: Optional[PortfolioGreeks] = None   # Gap 2: aggregated Greeks
