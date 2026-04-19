"""
Base Greeks engine ABC — shared by all 5 asset-class Greeks engines (L8).

GreeksResult     — standardised output merged back into InstrumentCandidate
BaseGreeksEngine — ABC each asset class implements
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from backend.models.hedge_models import InstrumentCandidate
    from backend.core.pricing.base_pricer import PriceResult, PricingContext


@dataclass
class GreeksResult:
    """
    Portfolio-level Greeks (scaled by n_contracts × multiplier).
    Fields not applicable to an asset class remain 0.0.
    extended holds asset-class-specific risk metrics.
    """
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    lambda_leverage: float = 0.0
    extended: Dict[str, Any] = field(default_factory=dict)


class BaseGreeksEngine(ABC):
    """Abstract Greeks engine. Dispatch via GREEKS_REGISTRY[asset_class]."""
    asset_class: str = ""

    def can_compute(self, candidate: "InstrumentCandidate") -> bool:
        return candidate.asset_class == self.asset_class

    @abstractmethod
    def compute(
        self,
        candidate: "InstrumentCandidate",
        price_result: "PriceResult",
        ctx: "PricingContext",
        n_contracts: int,
    ) -> GreeksResult:
        """
        Scale per-contract Greeks from price_result by n_contracts.
        Return portfolio-level GreeksResult.
        """
        ...
