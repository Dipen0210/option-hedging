from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class RegimeState(BaseModel):
    regime_label: str           # "low_vol" | "mid_vol" | "high_vol" | "unknown"
    regime_id: int              # HDBSCAN cluster id (-1 = anomaly)
    is_anomaly: bool            # cluster_id == -1
    anomaly_score: float        # HDBSCAN outlier score
    soft_membership: List[float]  # probability per discovered cluster
    n_discovered_regimes: int   # total clusters found by HDBSCAN (excl. noise)
    vix_level: float
    vol_forecast_garch: float
    realized_vol_20d: float
    sentiment_score: float = 0.0


class RiskProfile(BaseModel):
    ticker: str
    asset_class: str
    notional_value: float

    # Beta / factor
    beta_vs_spy: float
    small_cap_beta: float = 0.0
    value_beta: float = 0.0
    rate_sensitivity: float = 0.0
    credit_spread_sensitivity: float = 0.0
    usd_sensitivity: float = 0.0
    factor_contributions: Dict[str, float] = {}

    # VaR
    var_5pct: float             # 95% VaR in dollars
    cvar_5pct: float            # 95% CVaR (Expected Shortfall) in dollars
    var_pct: float              # VaR as % of notional
    cvar_pct: float             # CVaR as % of notional
    notional_at_risk: float

    # Hedge ratios
    optimal_hedge_ratio: float      # h* = ρ * (σ_S / σ_F)
    regression_hedge_ratio: float   # OLS slope
    tail_correlation: float         # bottom-10% correlation

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
    concentration_top1: float       # weight of largest holding
    risk_profiles: List[RiskProfile]
    correlation_matrix: Optional[List[List[float]]] = None
    tickers: Optional[List[str]] = None
    portfolio_greeks: Optional[PortfolioGreeks] = None   # Gap 2: aggregated Greeks
