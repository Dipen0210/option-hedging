"""
Black-76 model for options on futures / commodity forwards.

Used when the underlying is a futures contract (GC, CL, NG, etc.)
rather than a spot asset. The key difference from BSM: no cost-of-carry
term — futures price F is used directly.

Reference: Fischer Black (1976) "The Pricing of Commodity Contracts"
"""
import math
from scipy.stats import norm
from typing import Dict, Optional


def _d1_76(F: float, K: float, sigma: float, T: float) -> float:
    return (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))


def black76_price(
    F: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
) -> float:
    """
    Black-76 option price on a futures contract.

    Args:
        F:           futures price (not spot)
        K:           strike
        r:           risk-free rate (annualised, continuous) — used for discounting
        sigma:       vol (annualised)
        T:           time to expiry in years
        option_type: "put" or "call"

    Returns:
        Option price per unit of underlying.
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        if option_type == "put":
            return max((K - F) * math.exp(-r * T), 0.0)
        return max((F - K) * math.exp(-r * T), 0.0)

    d1 = _d1_76(F, K, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    if option_type == "put":
        price = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))

    return max(float(price), 0.0)


def black76_greeks(
    F: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
) -> Dict[str, float]:
    """
    Full Greeks for Black-76 option.

    Note: Delta here is dV/dF (not dV/dS like BSM).
    Lambda = Delta × (F / V) — capital efficiency relative to futures notional.
    """
    price = black76_price(F, K, r, sigma, T, option_type)

    if T <= 0 or sigma <= 0:
        return {
            "delta": -1.0 if (option_type == "put" and F < K) else 0.0,
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "lambda_": 0.0,
            "price": price,
        }

    d1 = _d1_76(F, K, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    n_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    # Delta (dV/dF)
    if option_type == "put":
        delta = -disc * norm.cdf(-d1)
    else:
        delta = disc * norm.cdf(d1)

    # Gamma
    gamma = disc * n_d1 / (F * sigma * sqrt_T)

    # Theta (daily)
    common = -disc * F * n_d1 * sigma / (2 * sqrt_T)
    if option_type == "put":
        theta = (common + r * disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))) / 365
    else:
        theta = (common - r * disc * (F * norm.cdf(d1) - K * norm.cdf(d2))) / 365

    # Vega (per 1% vol change)
    vega = disc * F * sqrt_T * n_d1 / 100

    # Lambda
    lambda_ = delta * F / price if price > 1e-6 else 0.0

    return {
        "delta":   round(delta,   6),
        "gamma":   round(gamma,   8),
        "theta":   round(theta,   6),
        "vega":    round(vega,    6),
        "lambda_": round(lambda_, 4),
        "price":   round(price,   4),
    }


def implied_vol_76(
    market_price: float,
    F: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "put",
    tol: float = 1e-6,
) -> Optional[float]:
    """Solve for implied vol using Brent's method."""
    from scipy.optimize import brentq

    intrinsic = max(K - F, 0) if option_type == "put" else max(F - K, 0)
    intrinsic *= math.exp(-r * T)
    if market_price <= intrinsic + 1e-6:
        return None

    def objective(sigma):
        return black76_price(F, K, r, sigma, T, option_type) - market_price

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=tol, maxiter=100)
        return round(float(iv), 6)
    except (ValueError, RuntimeError):
        return None
