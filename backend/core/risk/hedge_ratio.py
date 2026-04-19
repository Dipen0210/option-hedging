"""
Hedge ratio and position sizing math.
Pure quant math — no ML.
"""
import math
from typing import List, Dict


class HedgeRatioCalculator:

    def delta_adjusted_contracts(
        self,
        portfolio_notional: float,
        asset_beta: float,
        option_delta: float,
        contract_size: int = 100,
        spot_price: float = None,
        hedge_target_pct: float = 1.0,
    ) -> int:
        """
        N = (β × V × target%) / (|Δ| × contract_size × S)
        For equity option hedges (puts on SPY/IWM).
        Rounds to nearest integer; minimum 1 contract.
        """
        if abs(option_delta) < 1e-6 or spot_price is None or spot_price <= 0:
            return 0

        numerator   = abs(asset_beta) * portfolio_notional * hedge_target_pct
        denominator = abs(option_delta) * contract_size * spot_price
        n = numerator / denominator
        return max(1, round(n))

    def beta_weighted_portfolio(
        self,
        portfolio_beta: float,
        portfolio_value: float,
        futures_contract_value: float,
        hedge_target_pct: float = 1.0,
    ) -> int:
        """
        N* = βp × (Portfolio Value / Contract Value) × target%
        Use for portfolio-level index futures hedge (ES, NQ, RTY).
        """
        if futures_contract_value <= 0:
            return 0
        n = portfolio_beta * (portfolio_value / futures_contract_value) * hedge_target_pct
        return max(1, round(abs(n)))

    def min_variance_futures(
        self,
        hedge_ratio_h: float,
        notional: float,
        contract_size: float,
        futures_price: float,
    ) -> int:
        """
        N* = h* × V / (contract_size × F)
        Use for commodity proxy hedges (PDBC → CL futures).
        """
        if contract_size <= 0 or futures_price <= 0:
            return 0
        n = abs(hedge_ratio_h) * notional / (contract_size * futures_price)
        return max(1, round(n))

    def dv01_match(
        self,
        portfolio_dv01: float,
        futures_dv01_per_contract: float,
    ) -> int:
        """
        N = portfolio_DV01 / DV01_per_contract
        Use for Treasury futures to hedge bond duration.
        """
        if futures_dv01_per_contract <= 0:
            return 0
        return max(1, round(portfolio_dv01 / futures_dv01_per_contract))

    def partial_hedge_series(
        self,
        full_contracts: int,
        steps: List[float] = [0.5, 0.6, 0.7, 0.8, 1.0],
        premium_per_contract: float = 0.0,
    ) -> List[Dict]:
        """
        Returns a series of (hedge_pct, n_contracts, estimated_cost)
        to show cost/protection tradeoff to user.
        """
        results = []
        for pct in steps:
            n = max(1, round(full_contracts * pct))
            results.append({
                "hedge_pct": pct,
                "n_contracts": n,
                "estimated_cost": round(n * premium_per_contract, 2),
            })
        return results

    def rounding_error_pct(
        self,
        exact_contracts: float,
        rounded_contracts: int,
    ) -> float:
        """Returns rounding error as % of exact number."""
        if exact_contracts == 0:
            return 0.0
        return round(abs(exact_contracts - rounded_contracts) / exact_contracts * 100, 2)
