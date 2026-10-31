"""
Layer 9 — Simulation Engine

Runs Monte Carlo + named stress scenarios for every candidate concurrently.
Requires L6 (pricing), L7 (sizing) to be complete — needs price, n_contracts.

Per candidate:
  1. Run GBM (2000 paths) on the hedge instrument
  2. Run GBM on the asset being hedged (for correlation)
  3. Compute PayoffResult — mean payoff, VaR reduction, protection ratios
  4. Run 6 named stress scenarios
  5. Store all results in candidate.extended_metrics

Output fields added to extended_metrics:
  sim_mean_net_pnl        mean net P&L across paths
  sim_pct_profitable      % of paths where hedge profits
  sim_payoff_10pct_drop   expected payoff if asset falls 10%
  sim_payoff_20pct_drop   expected payoff if asset falls 20%
  sim_var_reduction_pct   % VaR reduction from hedge
  sim_hedge_ratio_realized actual realized hedge ratio
  scenario_{name}_*       per-scenario results
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from backend.models.hedge_models import HedgeOutput, InstrumentCandidate
from backend.core.simulation.monte_carlo import run_gbm
from backend.core.simulation.payoff_calculator import compute_payoff
from backend.core.simulation.scenario_engine import run_scenarios

logger = logging.getLogger(__name__)

_MAX_WORKERS = 16
_N_PATHS = 2000
_N_STEPS = 50


def _simulate_one(
    candidate: InstrumentCandidate,
    asset_spot: float,
    asset_sigma: float,
    asset_notional: float,
    risk_free_rate: float,
    T: float,
) -> InstrumentCandidate:
    try:
        hedge_spot = asset_spot      # hedge instrument proxied to asset for sim
        hedge_sigma = candidate.extended_metrics.get("vol", asset_sigma)

        # GBM for both asset and hedge instrument
        asset_mc = run_gbm(
            s0=asset_spot,
            mu=risk_free_rate,
            sigma=asset_sigma,
            T=T,
            n_paths=_N_PATHS,
            n_steps=_N_STEPS,
            seed=42,
        )
        hedge_mc = run_gbm(
            s0=hedge_spot,
            mu=risk_free_rate,
            sigma=hedge_sigma,
            T=T,
            n_paths=_N_PATHS,
            n_steps=_N_STEPS,
            seed=43,
        )

        K = candidate.strike or hedge_spot
        opt_type = candidate.option_type or "put"
        n = max(candidate.n_contracts, 1)
        premium = candidate.total_cost
        multiplier = 100 if candidate.asset_class == "equity" else 100
        ext = dict(candidate.extended_metrics)

        # Full MC payoff
        payoff = compute_payoff(
            asset_mc=asset_mc,
            hedge_mc=hedge_mc,
            K=K,
            option_type=opt_type,
            n_contracts=n,
            premium_paid=premium,
            asset_notional=asset_notional,
            multiplier=multiplier,
            asset_class=candidate.asset_class,
            extended=ext,
        )

        # Stress scenarios
        scenarios = run_scenarios(
            s0=hedge_spot,
            base_sigma=hedge_sigma,
            K=K,
            option_type=opt_type,
            n_contracts=n,
            premium_paid=premium,
            asset_notional=asset_notional,
            T=T,
            multiplier=multiplier,
            asset_class=candidate.asset_class,
            extended=ext,
            n_paths=500,
        )

        # Write payoff metrics
        candidate.extended_metrics.update({
            "sim_mean_net_pnl":        payoff.mean_net_pnl,
            "sim_pct_profitable":      payoff.pct_profitable,
            "sim_payoff_10pct_drop":   payoff.payoff_at_10pct_drop,
            "sim_payoff_20pct_drop":   payoff.payoff_at_20pct_drop,
            "sim_var_reduction_pct":   payoff.var_reduction_pct,
            "sim_hedge_ratio_realized": payoff.hedge_ratio_realized,
            "sim_expected_protection": payoff.expected_protection,
        })

        # Write scenario results
        for sc_name, sc_result in scenarios.items():
            candidate.extended_metrics[f"scenario_{sc_name}_payoff"]     = sc_result.hedge_payoff
            candidate.extended_metrics[f"scenario_{sc_name}_net_pnl"]    = sc_result.net_pnl
            candidate.extended_metrics[f"scenario_{sc_name}_protection"]  = sc_result.protection_ratio

    except Exception as exc:
        logger.error("Simulation failed for %s: %s", candidate.ticker, exc, exc_info=True)

    return candidate


class SimulationEngine:
    """
    L9 Simulation Engine.

    run() — concurrently simulate all candidates, enriching extended_metrics.

    market_data: dict with per-ticker spot prices, vols, notionals
    """

    def run(
        self,
        hedge_output: HedgeOutput,
        market_data: Dict[str, Dict],
        holding_notionals: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.053,
        hedge_horizon_days: int = 180,
    ) -> HedgeOutput:
        t0 = time.perf_counter()
        T = hedge_horizon_days / 365.25

        tasks: List[tuple] = []
        unsimulated: List[str] = []
        for rec in hedge_output.recommendations:
            asset_ticker = rec.asset_ticker
            md = market_data.get(asset_ticker, {})

            # Every input to a Monte Carlo path has to be measured. These
            # three lines used to read:
            #
            #     asset_spot     = md.get("spot_price", 100.0)
            #     asset_sigma    = md.get("sigma", 0.20)
            #     asset_notional = md.get("notional", hedge_output.portfolio_notional)
            #
            # A $100 starting price, a 20% volatility, and — because nothing
            # ever writes `md["notional"]` — the whole portfolio's exposure as
            # the holding's, on *every* run. The simulation's outputs are not
            # internal: `sim_var_reduction_pct`, `sim_expected_protection` and
            # every `scenario_*_protection` ratio are divided by that notional
            # and shipped to the user, so each was scaled by the ratio of the
            # portfolio to the holding.
            asset_spot = md.get("spot_price")
            asset_sigma = md.get("sigma")
            # The same map L10 scores against — L3's `current_price x shares`,
            # keyed by holding. `rec.unhedged_risk` is not an option here: L12
            # populates it, three layers after this one.
            asset_notional = (holding_notionals or {}).get(asset_ticker)

            if asset_spot is None or asset_sigma is None or not asset_notional:
                if rec.candidates:
                    missing = [
                        name for name, value in (
                            ("spot price", asset_spot),
                            ("volatility", asset_sigma),
                            ("measured notional", asset_notional),
                        ) if not value
                    ]
                    logger.warning(
                        "L9: not simulating %d candidate(s) for %s — missing %s",
                        len(rec.candidates), asset_ticker, ", ".join(missing),
                    )
                    unsimulated.append(asset_ticker)
                continue

            for cand in rec.candidates:
                tasks.append((cand, asset_spot, asset_sigma, asset_notional))

        if unsimulated:
            hedge_output.warnings.append(
                "Scenario and simulation figures are unavailable for "
                f"{', '.join(sorted(set(unsimulated)))}: the inputs a "
                "simulation needs were not measured for "
                f"{'them' if len(set(unsimulated)) > 1 else 'it'}. The hedge "
                "figures shown are computed, not simulated."
            )

        logger.info("L9 SimulationEngine: simulating %d candidates", len(tasks))

        futures_map = {}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            for cand, spot, sigma, notional in tasks:
                fut = pool.submit(
                    _simulate_one, cand, spot, sigma, notional, risk_free_rate, T
                )
                futures_map[fut] = cand.ticker

            for fut in as_completed(futures_map):
                ticker = futures_map[fut]
                try:
                    fut.result()
                    logger.debug("Simulation done: %s", ticker)
                except Exception as exc:
                    logger.error("Sim future failed for %s: %s", ticker, exc)

        elapsed = time.perf_counter() - t0
        logger.info("L9 SimulationEngine done in %.3fs", elapsed)
        return hedge_output
