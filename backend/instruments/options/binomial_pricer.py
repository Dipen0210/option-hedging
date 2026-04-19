"""
Cox-Ross-Rubinstein (CRR) Binomial Tree pricer.

Use for American-style options where early exercise premium matters:
  - Long-dated equity puts on individual stocks (dividend-paying)
  - American puts on ETFs with discrete dividends

Falls back to BSM for speed when early-exercise premium is negligible.
"""
import math
import numpy as np
from typing import Dict


def crr_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    american: bool = True,
    steps: int = 200,
) -> float:
    """
    CRR Binomial Tree option price.

    Args:
        S:           current spot price
        K:           strike price
        r:           risk-free rate (annualised, continuous)
        q:           dividend yield (annualised, continuous)
        sigma:       volatility (annualised)
        T:           time to expiry in years
        option_type: "put" or "call"
        american:    True = American (early exercise), False = European
        steps:       number of tree steps (default 200 is accurate to ~1 cent)

    Returns:
        Option price per share.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "put":
            return max(K - S, 0.0)
        return max(S - K, 0.0)

    dt = T / steps
    u  = math.exp(sigma * math.sqrt(dt))
    d  = 1.0 / u
    disc = math.exp(-r * dt)
    p  = (math.exp((r - q) * dt) - d) / (u - d)
    q_ = 1.0 - p                              # risk-neutral down-probability

    # Terminal stock prices
    stock = S * d ** np.arange(steps, -1, -1) * u ** np.arange(0, steps + 1, 1)

    # Terminal payoffs
    if option_type == "put":
        values = np.maximum(K - stock, 0.0)
    else:
        values = np.maximum(stock - K, 0.0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        stock = stock[:-1] * u          # roll back: S*d^(N-j)*u^j → S*d^(N-1-j)*u^j
        values = disc * (p * values[1:] + q_ * values[:-1])
        if american:
            if option_type == "put":
                intrinsic = np.maximum(K - stock, 0.0)
            else:
                intrinsic = np.maximum(stock - K, 0.0)
            values = np.maximum(values, intrinsic)

    return float(max(values[0], 0.0))


def crr_greeks(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    american: bool = True,
    steps: int = 200,
) -> Dict[str, float]:
    """
    Finite-difference Greeks from the binomial tree.

    Uses central differences:
        Delta = (V(S+ε) - V(S-ε)) / (2ε)
        Gamma = (V(S+ε) - 2V(S) + V(S-ε)) / ε²
        Theta = (V(T-dt) - V(T)) / dt     (daily)
        Vega  = (V(σ+0.01) - V(σ-0.01)) / 0.02 / 100

    Returns same dict shape as bsm_greeks for interchangeability.
    """
    # Bump must exceed tree node spacing (~S*(u-1) ≈ S*σ*√(T/N))
    # Use 2% to stay safely above it while remaining numerically stable
    eps_s = S * 0.02           # 2% bump for delta/gamma
    eps_v = 0.01               # 1 vol-point for vega
    dt_bump = 1 / 365          # 1-day theta bump

    def price(s=S, sig=sigma, t=T):
        return crr_price(s, K, r, q, sig, t, option_type, american, steps)

    v0  = price()
    v_up   = price(s=S + eps_s)
    v_down = price(s=S - eps_s)
    delta  = (v_up - v_down) / (2 * eps_s)
    gamma  = (v_up - 2 * v0 + v_down) / (eps_s ** 2)

    # Theta: value with 1 less day
    T_bump = max(T - dt_bump, 1e-6)
    theta  = (price(t=T_bump) - v0) / dt_bump / 365

    # Vega: per 1% change in vol
    sig_hi = sigma + eps_v
    sig_lo = max(sigma - eps_v, 1e-4)
    vega   = (price(sig=sig_hi) - price(sig=sig_lo)) / (sig_hi - sig_lo) / 100

    lambda_ = delta * S / v0 if v0 > 1e-6 else 0.0

    return {
        "delta":   round(delta,   6),
        "gamma":   round(gamma,   8),
        "theta":   round(theta,   6),
        "vega":    round(vega,    6),
        "lambda_": round(lambda_, 4),
        "price":   round(v0,      4),
    }


def early_exercise_premium(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    steps: int = 200,
) -> float:
    """
    Early exercise premium = American price - European price.
    Use to decide whether CRR is necessary vs BSM (skip CRR if EEP < $0.02).
    """
    american_price = crr_price(S, K, r, q, sigma, T, option_type, american=True,  steps=steps)
    european_price = crr_price(S, K, r, q, sigma, T, option_type, american=False, steps=steps)
    return round(american_price - european_price, 4)
