"""
Options position sizing engine.

Layers (applied in order):
  1. Delta-adjusted contracts  — how many contracts to hedge the target beta exposure
  2. Budget gate               — hard cap: max_cost_pct × portfolio_notional
  3. Round-lot constraint      — round down to nearest integer ≥ 1
  4. Partial hedge series      — present cost/protection tradeoff to the user

All functions are pure math — no network calls.
"""
import math
from typing import Dict, List, Optional


# Smallest option delta that can carry a hedge. Below this the contract has
# effectively no exposure to the underlying, so N = target / |Δ| explodes: a
# SPY put priced off a junk 1.56% implied vol came back with |Δ| ≈ 1e-5 and the
# sizer asked for 1,500,000 contracts. The old guard was 1e-6, which is a
# division-by-zero check rather than a statement about what can hedge.
MIN_HEDGE_DELTA = 0.01

# Smallest per-share premium that can be treated as a real price. A contract
# quoted below a cent is not protection the user can buy; it signals a bad
# pricing input, and dividing a budget by it bypasses the budget entirely.
MIN_PREMIUM_PER_SHARE = 0.01


# ── Core sizing ───────────────────────────────────────────────────────────────

def size_by_delta(
    portfolio_notional: float,
    asset_beta: float,
    option_delta: float,
    spot_price: float,
    contract_size: int = 100,
    hedge_target_pct: float = 1.0,
) -> int:
    """
    N = (β × V × target%) / (|Δ| × contract_size × S)

    Args:
        portfolio_notional: total $ value being hedged (position notional)
        asset_beta:         beta of position vs hedge instrument
        option_delta:       option delta (|Δ|; sign handled internally)
        spot_price:         current price of the underlying of the option
        contract_size:      shares per contract (default 100)
        hedge_target_pct:   fraction to hedge (1.0 = full hedge)

    Returns:
        Integer number of contracts (minimum 1 if > 0).
    """
    abs_delta = abs(option_delta)
    if abs_delta < MIN_HEDGE_DELTA or spot_price <= 0:
        # Not a usable hedge: a contract with no meaningful delta cannot offset
        # the position, and sizing against it produces an absurd contract count
        # rather than a small one.
        return 0

    numerator   = abs(asset_beta) * portfolio_notional * hedge_target_pct
    denominator = abs_delta * contract_size * spot_price
    n = numerator / denominator
    return max(1, math.floor(n))   # floor = conservative (don't over-hedge)


# ── Budget gate ───────────────────────────────────────────────────────────────

def apply_budget_gate(
    n_contracts: int,
    option_price: float,
    portfolio_notional: float,
    max_cost_pct: float,
    contract_size: int = 100,
) -> Dict[str, object]:
    """
    Hard cap on total premium spend.

    Args:
        n_contracts:       raw contract count from sizing
        option_price:      premium per share
        portfolio_notional: total portfolio $ value
        max_cost_pct:      user's max hedge cost as % of portfolio (e.g. 0.02 = 2%)
        contract_size:     shares per contract (default 100)

    Returns:
        dict with:
            n_contracts_gated:  contracts after budget cap (may be < input)
            total_cost:         actual premium spend
            budget_cap:         max allowed spend
            budget_used_pct:    total_cost / portfolio_notional
            capped:             True if budget was binding
    """
    budget_cap   = portfolio_notional * max_cost_pct
    cost_per_lot = option_price * contract_size

    if option_price < MIN_PREMIUM_PER_SHARE:
        # The gate previously fell back to `max_contracts = n_contracts` when
        # cost_per_lot was 0, which let an unpriced contract through the budget
        # cap untouched and sized it as if protection were free. An unpriceable
        # contract is rejected instead.
        return {
            "n_contracts_gated": 0,
            "total_cost":        0.0,
            "budget_cap":        round(budget_cap, 2),
            "budget_used_pct":   0.0,
            "capped":            True,
        }

    max_contracts = int(budget_cap / cost_per_lot)

    n_gated   = min(n_contracts, max_contracts)
    n_gated   = max(n_gated, 0)
    total_cost = n_gated * cost_per_lot

    return {
        "n_contracts_gated": n_gated,
        "total_cost":        round(total_cost, 2),
        "budget_cap":        round(budget_cap, 2),
        "budget_used_pct":   round(total_cost / portfolio_notional, 6) if portfolio_notional > 0 else 0.0,
        "capped":            n_gated < n_contracts,
    }


# ── Partial hedge series ──────────────────────────────────────────────────────

def partial_hedge_series(
    full_contracts: int,
    option_price: float,
    contract_size: int = 100,
    steps: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Generate cost/protection tradeoff table for the UI.

    Args:
        full_contracts: contracts for 100% hedge (after budget gate)
        option_price:   premium per share
        contract_size:  shares per contract
        steps:          hedge coverage fractions to show (default: 50/60/70/80/100%)

    Returns:
        List of dicts: {hedge_pct, n_contracts, estimated_cost, cost_per_pct_protection}
    """
    if steps is None:
        steps = [0.50, 0.60, 0.70, 0.80, 1.00]

    results = []
    for pct in steps:
        n = max(1, round(full_contracts * pct))
        cost = n * option_price * contract_size
        results.append({
            "hedge_pct":              round(pct, 2),
            "n_contracts":            n,
            "estimated_cost":         round(cost, 2),
            "cost_per_pct_protection": round(cost / (pct * 100), 2) if pct > 0 else 0.0,
        })
    return results


# ── Rounding diagnostics ──────────────────────────────────────────────────────

def rounding_error_pct(exact: float, rounded: int) -> float:
    """
    How much do we over/under hedge due to integer rounding?
    Returns error as % of exact contract count.
    """
    if exact == 0:
        return 0.0
    return round(abs(exact - rounded) / exact * 100, 2)


# ── All-in-one sizing pipeline ────────────────────────────────────────────────

def compute_position(
    portfolio_notional: float,
    asset_beta: float,
    option_delta: float,
    option_price: float,
    spot_price: float,
    max_cost_pct: float,
    hedge_target_pct: float = 1.0,
    contract_size: int = 100,
    partial_steps: Optional[List[float]] = None,
) -> Dict:
    """
    Full sizing pipeline: delta-size → budget gate → partial series.

    Returns:
        full_contracts:   contracts for requested hedge target (pre-budget)
        gated:            budget gate result dict
        rounding_error:   % error from integer rounding
        partial_series:   cost/protection tradeoff table
        viable:           False if gated to 0 contracts (too expensive)
    """
    n_full = size_by_delta(
        portfolio_notional, asset_beta, option_delta,
        spot_price, contract_size, hedge_target_pct,
    )

    gate = apply_budget_gate(
        n_full, option_price, portfolio_notional, max_cost_pct, contract_size
    )

    n_gated = gate["n_contracts_gated"]
    exact   = (abs(asset_beta) * portfolio_notional * hedge_target_pct
               / (abs(option_delta) * contract_size * spot_price)) if abs(option_delta) > 1e-6 and spot_price > 0 else 0.0

    return {
        "full_contracts":  n_full,
        "gated":           gate,
        "rounding_error":  rounding_error_pct(exact, n_gated),
        "partial_series":  partial_hedge_series(n_gated, option_price, contract_size, partial_steps),
        "viable":          n_gated > 0,
    }
