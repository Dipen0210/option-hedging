"""
Greeks aggregation for single-leg and multi-leg option strategies.

Single-leg:
    Use bsm_greeks / crr_greeks / black76_greeks directly.

Multi-leg net Greeks:
    collar    = long put + short call  (Δ_net = Δ_put + Δ_call)
    spread    = long put + short lower put
    straddle  = long call + long put
    Any arbitrary list of (greeks_dict, signed_quantity) pairs.

Lambda interpretation:
    Lambda = Δ × (S / V_option)  — how many $ the option moves per $1 move in
    the underlying, expressed as a multiple of premium paid. Use this for
    capital efficiency ranking in the selector.
"""
from typing import Dict, List, Tuple


GreeksDict = Dict[str, float]  # keys: delta, gamma, theta, vega, lambda_, price


def net_greeks(
    legs: List[Tuple[GreeksDict, int]],
) -> GreeksDict:
    """
    Aggregate Greeks across a multi-leg position.

    Args:
        legs: list of (greeks_dict, signed_quantity) tuples.
              quantity > 0 = long, quantity < 0 = short.
              Each greeks_dict has keys: delta, gamma, theta, vega, lambda_, price.

    Returns:
        Net Greeks dict. `price` = total premium (positive = net debit).
        `lambda_` is recomputed on net_delta / net_price.

    Example — collar:
        put_greeks  = bsm_greeks(S, K_put,  r, q, σ, T, "put")
        call_greeks = bsm_greeks(S, K_call, r, q, σ, T, "call")
        # Long 1 put, short 1 call:
        net = net_greeks([(put_greeks, +1), (call_greeks, -1)])
    """
    totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "price": 0.0}

    for g, qty in legs:
        for key in totals:
            totals[key] += g.get(key, 0.0) * qty

    # Recompute lambda on net position
    net_price = totals["price"]
    net_delta = totals["delta"]
    totals["lambda_"] = round(net_delta / net_price, 4) if abs(net_price) > 1e-6 else 0.0

    return {k: round(v, 6) for k, v in totals.items()}


# ── Convenience builders ──────────────────────────────────────────────────────

def collar_greeks(
    put_greeks: GreeksDict,
    call_greeks: GreeksDict,
    n_puts: int = 1,
    n_calls: int = 1,
) -> GreeksDict:
    """
    Long n_puts puts + short n_calls calls.
    Collar reduces delta exposure, lowers net premium.
    """
    return net_greeks([
        (put_greeks,  +n_puts),
        (call_greeks, -n_calls),
    ])


def bull_put_spread_greeks(
    long_put_greeks: GreeksDict,
    short_put_greeks: GreeksDict,
    n_contracts: int = 1,
) -> GreeksDict:
    """
    Short higher-strike put + long lower-strike put.
    Cheaper than naked put; capped protection.
    """
    return net_greeks([
        (short_put_greeks, -n_contracts),
        (long_put_greeks,  +n_contracts),
    ])


def bear_put_spread_greeks(
    long_put_greeks: GreeksDict,
    short_put_greeks: GreeksDict,
    n_contracts: int = 1,
) -> GreeksDict:
    """
    Long higher-strike put + short lower-strike put.
    Classic bearish hedge with defined risk/reward.
    """
    return net_greeks([
        (long_put_greeks,  +n_contracts),
        (short_put_greeks, -n_contracts),
    ])


def protective_put_greeks(
    stock_delta: float,
    put_greeks: GreeksDict,
    n_contracts: int = 1,
    shares_per_contract: int = 100,
) -> GreeksDict:
    """
    Stock position + long puts.
    stock_delta = 1.0 × number_of_shares (in delta units, so just shares).
    Net delta = shares + (put_delta × n_contracts × 100).
    """
    stock_leg: GreeksDict = {
        "delta":   stock_delta,
        "gamma":   0.0,
        "theta":   0.0,
        "vega":    0.0,
        "lambda_": 0.0,
        "price":   0.0,
    }
    # 1 option contract covers `shares_per_contract` shares
    effective_qty = n_contracts * shares_per_contract
    return net_greeks([
        (stock_leg,  1),
        (put_greeks, effective_qty),
    ])


# ── Greeks diagnostics ────────────────────────────────────────────────────────

def hedge_effectiveness(
    portfolio_delta: float,
    option_net_delta: float,
    portfolio_notional: float,
    option_notional: float,
) -> Dict[str, float]:
    """
    How much of the portfolio delta does this option position offset?

    Returns:
        delta_offset_pct:   % of portfolio delta hedged
        notional_ratio:     option cost as % of portfolio notional
    """
    if portfolio_delta == 0:
        return {"delta_offset_pct": 0.0, "notional_ratio": 0.0}

    delta_offset_pct = abs(option_net_delta / portfolio_delta) * 100
    notional_ratio   = (option_notional / portfolio_notional) * 100 if portfolio_notional > 0 else 0.0

    return {
        "delta_offset_pct": round(delta_offset_pct, 2),
        "notional_ratio":   round(notional_ratio,   2),
    }
