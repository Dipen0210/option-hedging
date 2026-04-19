"""
Geometric Brownian Motion Monte Carlo engine.

Generates N price paths for an underlying asset using GBM:
    dS = μ·S·dt + σ·S·dW

Used by L9 SimulationEngine to simulate P&L distributions for
each hedge candidate under random market paths.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCResult:
    """Output of a single Monte Carlo run."""
    paths: np.ndarray               # shape (n_paths, n_steps+1) — price paths
    terminal_prices: np.ndarray     # shape (n_paths,) — S_T
    pct_returns: np.ndarray         # shape (n_paths,) — (S_T - S_0) / S_0
    s0: float
    mu: float
    sigma: float
    T: float

    def var(self, confidence: float = 0.95) -> float:
        """Historical VaR of terminal return at given confidence."""
        return float(np.percentile(self.pct_returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """CVaR (Expected Shortfall) of terminal return."""
        threshold = self.var(confidence)
        tail = self.pct_returns[self.pct_returns <= threshold]
        return float(tail.mean()) if len(tail) > 0 else threshold

    def prob_below(self, drop_pct: float) -> float:
        """Probability terminal price is ≥ drop_pct below S_0 (e.g. drop_pct=0.10 → -10%)."""
        threshold = self.s0 * (1 - drop_pct)
        return float(np.mean(self.terminal_prices < threshold))

    def expected_return(self) -> float:
        return float(self.pct_returns.mean())


def run_gbm(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int = 2000,
    n_steps: int = 50,
    seed: Optional[int] = None,
) -> MCResult:
    """
    Run GBM Monte Carlo simulation.

    Args:
        s0:      initial spot price
        mu:      annualised drift (risk-neutral: use risk_free_rate - div_yield)
        sigma:   annualised volatility
        T:       time horizon in years
        n_paths: number of simulated paths (2000 is fast + accurate enough for scoring)
        n_steps: time steps per path (50 = weekly for 1yr)
        seed:    RNG seed for reproducibility

    Returns:
        MCResult with full path matrix and summary stats.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # GBM increments: log(S_{t+1}/S_t) ~ N((μ - σ²/2)dt, σ²dt)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * sqrt_dt

    # Shape: (n_paths, n_steps)
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * Z

    # Cumulative path: shape (n_paths, n_steps+1)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)],
        axis=1,
    )
    paths = s0 * np.exp(log_paths)

    terminal = paths[:, -1]
    pct_returns = (terminal - s0) / s0

    return MCResult(
        paths=paths,
        terminal_prices=terminal,
        pct_returns=pct_returns,
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
    )


def run_stressed_gbm(
    s0: float,
    base_sigma: float,
    stress_sigma_multiplier: float,
    drift_shock: float,
    T: float,
    n_paths: int = 2000,
    n_steps: int = 50,
    seed: Optional[int] = None,
) -> MCResult:
    """
    Stressed GBM — for scenario analysis.

    stress_sigma_multiplier: e.g. 2.0 = vol doubles (VIX spike)
    drift_shock:             e.g. -0.20 = -20% instantaneous drift shock (crash)
    """
    stressed_sigma = base_sigma * stress_sigma_multiplier
    stressed_mu = drift_shock / T    # spread the shock over T years
    return run_gbm(s0, stressed_mu, stressed_sigma, T, n_paths, n_steps, seed)
