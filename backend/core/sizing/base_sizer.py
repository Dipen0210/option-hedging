"""
Base sizer ABC — shared by all 5 asset-class position sizers (L7).

SizingContext  — portfolio + risk inputs
SizingResult   — n_contracts, notional, total_cost, partial_hedge_options
BaseSizer      — ABC each asset class implements
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from backend.models.hedge_models import InstrumentCandidate
    from backend.core.pricing.base_pricer import PriceResult


@dataclass
class SizingContext:
    """Risk budget and portfolio data for position sizing."""
    portfolio_notional: float           # total portfolio value ($)
    holding_notional: float             # notional of the specific holding being hedged
    hedge_ratio: float = 1.0            # target fraction to hedge (0–1)
    beta: float = 1.0                   # portfolio beta vs hedge instrument
    max_premium_pct: float = 0.02       # max cost as % of holding_notional
    correlation: float = 0.85           # ρ for commodity min-variance
    spot_vol: float = 0.20              # σ of the spot asset (for commodity)
    futures_vol: float = 0.22           # σ of futures (for commodity)
    portfolio_dv01: float = 0.0         # total portfolio DV01 (for IR)
    portfolio_cs01: float = 0.0         # total portfolio CS01 (for credit)
    fx_exposure: float = 0.0            # exact FX exposure in base currency
    extended: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SizingResult:
    """Output from any asset-class sizer."""
    n_contracts: int
    total_cost: float                   # premium outlay in dollars
    notional_hedged: float              # dollar notional covered
    hedge_effectiveness: float = 1.0    # % of exposure hedged
    partial_hedge_options: List[Dict[str, Any]] = field(default_factory=list)


class BaseSizer(ABC):
    """Abstract sizer.  Dispatch via SIZER_REGISTRY[asset_class]."""
    asset_class: str = ""

    def can_size(self, candidate: "InstrumentCandidate") -> bool:
        return candidate.asset_class == self.asset_class

    @abstractmethod
    def size(
        self,
        candidate: "InstrumentCandidate",
        price_result: "PriceResult",
        ctx: SizingContext,
    ) -> SizingResult:
        """Return a SizingResult for the given candidate + price."""
        ...

    def _partial_series(
        self,
        candidate: "InstrumentCandidate",
        price_result: "PriceResult",
        ctx: SizingContext,
        full_n: int,
    ) -> List[Dict[str, Any]]:
        """Generate a tradeoff table: 25/50/75/100% of full hedge."""
        rows = []
        for pct in (0.25, 0.50, 0.75, 1.00):
            n = max(1, round(full_n * pct))
            cost = n * price_result.price * 100
            rows.append({
                "hedge_pct": pct,
                "n_contracts": n,
                "total_cost": round(cost, 2),
                "cost_pct_portfolio": round(cost / max(ctx.holding_notional, 1), 4),
            })
        return rows
