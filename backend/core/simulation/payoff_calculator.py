"""
Payoff calculator — computes hedge P&L across simulated price paths.

For each MC path, computes:
  - Option payoff at expiry
  - Net P&L after premium paid
  - Protected loss (what the hedge saved vs unhedged)

Works for all asset classes:
  Equity    — BSM put/call payoff: max(K-S_T, 0) × n × 100
  Commodity — Black-76 payoff on futures paths
  FX        — GK put/call payoff on FX rate paths
  IR        — DV01-based linear P&L (rate shock → DV01 gain)
  Credit    — CS01-based P&L (spread widening → protection gain)
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from backend.core.simulation.monte_carlo import MCResult


@dataclass
class PayoffResult:
    """Hedge P&L summary across all simulated paths."""
    mean_payoff: float          # mean option payoff (gross, before premium)
    mean_net_pnl: float         # mean net P&L = payoff - premium_paid
    pct_profitable: float       # % of paths where hedge generates positive payoff
    expected_protection: float  # E[max(loss_saved, 0)] in dollar terms
    payoff_at_10pct_drop: float # expected hedge payoff conditional on asset -10%
    payoff_at_20pct_drop: float # expected hedge payoff conditional on asset -20%
    var_reduction_pct: float    # % reduction in portfolio VaR from hedge
    hedge_ratio_realized: float # avg realized hedge ratio = payoff / asset_loss


def option_payoffs(
    terminal_prices: np.ndarray,
    K: float,
    option_type: str,
    n_contracts: int,
    multiplier: int = 100,
) -> np.ndarray:
    """
    Intrinsic payoff at expiry for each path.
    Returns array of gross payoffs (before premium).
    """
    if option_type.lower() == "put":
        intrinsic = np.maximum(K - terminal_prices, 0.0)
    else:
        intrinsic = np.maximum(terminal_prices - K, 0.0)
    return intrinsic * n_contracts * multiplier


def compute_payoff(
    asset_mc: MCResult,
    hedge_mc: MCResult,
    K: float,
    option_type: str,
    n_contracts: int,
    premium_paid: float,
    asset_notional: float,
    multiplier: int = 100,
    asset_class: str = "equity",
    extended: Optional[dict] = None,
) -> PayoffResult:
    """
    Full P&L profile for one hedge candidate.

    asset_mc:       MC result for the asset being hedged
    hedge_mc:       MC result for the hedge instrument
    K:              strike (or 0 for non-option instruments)
    n_contracts:    position size from L7
    premium_paid:   total upfront cost from L7 (dollars)
    asset_notional: dollar value of position being hedged
    """
    ext = extended or {}
    s0_asset = asset_mc.s0

    if asset_class in ("equity", "commodity", "fx") and K > 0:
        # Option payoff uses HEDGE instrument terminal prices
        gross = option_payoffs(hedge_mc.terminal_prices, K, option_type, n_contracts, multiplier)
    elif asset_class == "bond":
        # IR: linear DV01 payoff — rate shock → DV01 gain
        dv01 = ext.get("dv01", 0.0) * n_contracts
        # Rate change approximation: assume rate moves proportional to asset price change
        rate_change_paths = -(hedge_mc.pct_returns * 0.01)  # 1% price → ~1bp rate
        gross = dv01 * rate_change_paths * 10000             # bps
    elif asset_class == "credit":
        # Credit: CS01 payoff — spread widening → protection gain
        cs01 = ext.get("cs01", 0.0) * n_contracts
        # Assume spread widens proportionally when asset falls
        spread_change_bps = np.where(
            asset_mc.pct_returns < 0,
            np.abs(asset_mc.pct_returns) * 500,  # rough: -1% asset → +5bps spread
            0.0,
        )
        gross = cs01 * spread_change_bps
    else:
        gross = np.zeros(len(asset_mc.terminal_prices))

    net_pnl = gross - premium_paid

    # Asset losses (negative = asset fell)
    asset_pnl = asset_mc.pct_returns * asset_notional
    asset_loss = np.where(asset_pnl < 0, -asset_pnl, 0.0)  # positive = loss amount

    protected = np.minimum(gross, asset_loss)  # hedge can't protect more than actual loss

    # Conditional payoffs at specific drop levels
    mask_10 = asset_mc.pct_returns <= -0.10
    mask_20 = asset_mc.pct_returns <= -0.20

    payoff_10 = float(gross[mask_10].mean()) if mask_10.sum() > 10 else 0.0
    payoff_20 = float(gross[mask_20].mean()) if mask_20.sum() > 5 else 0.0

    # VaR reduction
    unhedged_var = float(np.percentile(asset_pnl, 5))
    hedged_pnl = asset_pnl + net_pnl
    hedged_var = float(np.percentile(hedged_pnl, 5))
    var_reduction = (hedged_var - unhedged_var) / max(abs(unhedged_var), 1.0)

    # Realized hedge ratio
    avg_asset_loss = float(asset_loss.mean())
    avg_gross = float(gross.mean())
    realized_ratio = avg_gross / avg_asset_loss if avg_asset_loss > 1e-6 else 0.0

    return PayoffResult(
        mean_payoff=round(float(gross.mean()), 2),
        mean_net_pnl=round(float(net_pnl.mean()), 2),
        pct_profitable=round(float(np.mean(net_pnl > 0)), 4),
        expected_protection=round(float(protected.mean()), 2),
        payoff_at_10pct_drop=round(payoff_10, 2),
        payoff_at_20pct_drop=round(payoff_20, 2),
        var_reduction_pct=round(min(max(var_reduction, 0.0), 1.0), 4),
        hedge_ratio_realized=round(min(realized_ratio, 2.0), 4),
    )
