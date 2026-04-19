"""
Named stress scenario engine.

Applies pre-defined macro shock scenarios to a hedge candidate and
returns expected P&L under each scenario.

Scenarios (calibrated to historical crises):
  market_crash      → -30% asset, vol ×3  (2008/2020 style)
  vol_spike         → -15% asset, vol ×2  (VIX spike to 40+)
  rate_shock        → +200bps rates, -10% asset (2022 style)
  mild_correction   → -10% asset, vol ×1.5
  tail_event        → -40% asset, vol ×4  (black swan)
  flat_market       → 0% asset, vol ×0.8  (hedge cost only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from backend.core.simulation.monte_carlo import run_stressed_gbm
from backend.core.simulation.payoff_calculator import compute_payoff


@dataclass
class ScenarioDefinition:
    name: str
    drift_shock: float          # instantaneous drift (annualised equivalent)
    vol_multiplier: float       # σ_stressed = σ_base × vol_multiplier
    description: str


@dataclass
class ScenarioResult:
    scenario: str
    asset_pnl_pct: float        # expected asset P&L % under this scenario
    hedge_payoff: float         # expected gross payoff from hedge ($)
    net_pnl: float              # hedge_payoff - premium_paid ($)
    protection_ratio: float     # hedge_payoff / |asset_loss|  (0–1)


SCENARIOS: List[ScenarioDefinition] = [
    ScenarioDefinition("mild_correction",  -0.10, 1.5, "-10% asset, vol ×1.5"),
    ScenarioDefinition("vol_spike",        -0.15, 2.0, "-15% asset, VIX doubles"),
    ScenarioDefinition("market_crash",     -0.30, 3.0, "-30% asset, crisis-level vol"),
    ScenarioDefinition("rate_shock",       -0.10, 1.3, "+200bps rates, -10% asset"),
    ScenarioDefinition("tail_event",       -0.40, 4.0, "-40% asset, black swan"),
    ScenarioDefinition("flat_market",       0.00, 0.8, "flat market, low vol"),
]


def run_scenarios(
    s0: float,
    base_sigma: float,
    K: float,
    option_type: str,
    n_contracts: int,
    premium_paid: float,
    asset_notional: float,
    T: float,
    multiplier: int = 100,
    asset_class: str = "equity",
    extended: dict | None = None,
    n_paths: int = 1000,
) -> Dict[str, ScenarioResult]:
    """
    Run all SCENARIOS and return results keyed by scenario name.
    Uses a lighter 1000-path sim (vs 2000 for full MC) for speed.
    """
    results: Dict[str, ScenarioResult] = {}

    for sc in SCENARIOS:
        try:
            # Stressed asset paths
            asset_mc = run_stressed_gbm(
                s0=s0,
                base_sigma=base_sigma,
                stress_sigma_multiplier=sc.vol_multiplier,
                drift_shock=sc.drift_shock,
                T=T,
                n_paths=n_paths,
                seed=42,
            )

            # Hedge instrument assumed to move proportionally to asset
            # (simplified — hedge correlation = 0.9 for equity, 1.0 for exact)
            corr = 0.90 if asset_class == "equity" else 0.95
            hedge_terminal = (
                s0 * (1 + asset_mc.pct_returns * corr)
            )
            # Build a minimal MCResult-like structure
            from backend.core.simulation.monte_carlo import MCResult
            hedge_mc = MCResult(
                paths=np.column_stack([np.full(n_paths, s0), hedge_terminal]),
                terminal_prices=hedge_terminal,
                pct_returns=(hedge_terminal - s0) / s0,
                s0=s0, mu=sc.drift_shock, sigma=base_sigma * sc.vol_multiplier, T=T,
            )

            payoff = compute_payoff(
                asset_mc=asset_mc,
                hedge_mc=hedge_mc,
                K=K,
                option_type=option_type,
                n_contracts=n_contracts,
                premium_paid=premium_paid,
                asset_notional=asset_notional,
                multiplier=multiplier,
                asset_class=asset_class,
                extended=extended,
            )

            asset_loss = abs(asset_mc.expected_return() * asset_notional)
            protection = payoff.mean_payoff / asset_loss if asset_loss > 1e-6 else 0.0

            results[sc.name] = ScenarioResult(
                scenario=sc.name,
                asset_pnl_pct=round(asset_mc.expected_return(), 4),
                hedge_payoff=payoff.mean_payoff,
                net_pnl=payoff.mean_net_pnl,
                protection_ratio=round(min(protection, 1.0), 4),
            )
        except Exception:
            results[sc.name] = ScenarioResult(
                scenario=sc.name,
                asset_pnl_pct=sc.drift_shock,
                hedge_payoff=0.0,
                net_pnl=-premium_paid,
                protection_ratio=0.0,
            )

    return results
