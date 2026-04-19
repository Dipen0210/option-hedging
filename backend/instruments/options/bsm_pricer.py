"""
Black-Scholes-Merton pricer for European options.

Handles puts and calls. Used for liquid index ETF options (SPY, QQQ, IWM, TLT, GLD).
"""
import math
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """d1 in BSM formula."""
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    """d2 = d1 - σ√T"""
    return d1 - sigma * math.sqrt(T)


def bsm_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "put",
) -> float:
    """
    BSM option price.

    Args:
        S: current spot price
        K: strike price
        r: risk-free rate (annualised, continuous)
        q: dividend yield (annualised, continuous)
        sigma: implied / GARCH vol (annualised)
        T: time to expiry in years
        option_type: "put" or "call"

    Returns:
        Option price in dollars per share (multiply by 100 for per-contract value).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "put":
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)

    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)

    if option_type == "put":
        price = (K * math.exp(-r * T) * norm.cdf(-d2)
                 - S * math.exp(-q * T) * norm.cdf(-d1))
    else:
        price = (S * math.exp(-q * T) * norm.cdf(d1)
                 - K * math.exp(-r * T) * norm.cdf(d2))
    return max(price, 0.0)


def bsm_greeks(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "put",
) -> Dict[str, float]:
    """
    Full first-order Greeks (Delta, Gamma, Theta, Vega) + Lambda.

    Returns:
        delta:   dV/dS  (negative for puts)
        gamma:   d²V/dS²  (always positive)
        theta:   dV/dT  (daily decay, always negative for long options)
        vega:    dV/dσ  (in $ per 1-vol-point, i.e. per 0.01 change in σ)
        lambda_: leverage ratio = Δ × (S / V)  — capital efficiency
        price:   option price (recomputed for consistency)
    """
    price = bsm_price(S, K, r, q, sigma, T, option_type)

    if T <= 0 or sigma <= 0:
        delta = -1.0 if (option_type == "put" and S < K) else 0.0
        return {
            "delta": delta, "gamma": 0.0, "theta": 0.0,
            "vega": 0.0, "lambda_": 0.0, "price": price,
        }

    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    n_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    # Delta
    if option_type == "put":
        delta = math.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        delta = math.exp(-q * T) * norm.cdf(d1)

    # Gamma (same for put & call by put-call parity)
    gamma = math.exp(-q * T) * n_d1 / (S * sigma * sqrt_T)

    # Theta (annualised → divide by 365 for daily)
    common = (- S * math.exp(-q * T) * n_d1 * sigma / (2 * sqrt_T))
    if option_type == "put":
        theta = (common
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)
                 - q * S * math.exp(-q * T) * norm.cdf(-d1)) / 365
    else:
        theta = (common
                 - r * K * math.exp(-r * T) * norm.cdf(d2)
                 + q * S * math.exp(-q * T) * norm.cdf(d1)) / 365

    # Vega (per 1% change in vol → divide annualised vega by 100)
    vega = S * math.exp(-q * T) * sqrt_T * n_d1 / 100

    # Lambda (leverage / capital efficiency)
    lambda_ = delta * S / price if price > 1e-6 else 0.0

    return {
        "delta":   round(delta,   6),
        "gamma":   round(gamma,   8),
        "theta":   round(theta,   6),
        "vega":    round(vega,    6),
        "lambda_": round(lambda_, 4),
        "price":   round(price,   4),
    }


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    option_type: str = "put",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Solve for implied volatility using Brent's method (scipy brentq).

    Returns None if no solution found (deep ITM/OTM or bad price).
    """
    from scipy.optimize import brentq

    intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
    if market_price <= intrinsic + 1e-6:
        return None

    def objective(sigma):
        return bsm_price(S, K, r, q, sigma, T, option_type) - market_price

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=tol, maxiter=max_iter)
        return round(float(iv), 6)
    except (ValueError, RuntimeError):
        return None


def put_call_parity_check(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    tolerance: float = 0.05,
) -> bool:
    """
    Validate: C - P ≈ S·e^(-qT) - K·e^(-rT)
    Uses q=0 simplified form. Returns True if parity holds within tolerance.
    """
    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    return abs(lhs - rhs) / max(abs(rhs), 1.0) < tolerance
