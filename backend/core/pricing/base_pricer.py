"""
Base classes shared by all asset-class pricers (L6).

PricingContext  — market data snapshot passed into every pricer
PriceResult     — standardised output with price + Greeks + extended dict
BaseAssetPricer — ABC that each per-asset pricer implements
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from backend.models.hedge_models import InstrumentCandidate


def years_to_expiry(expiry_str: str) -> float:
    """Convert ISO-8601 expiry string to fractional years from today."""
    try:
        expiry = date.fromisoformat(expiry_str)
        today = date.today()
        return max((expiry - today).days / 365.25, 1e-6)
    except (ValueError, TypeError):
        return 0.0


@dataclass
class PricingContext:
    """Market data snapshot for a single pricing run."""
    spot_prices: Dict[str, float]                          # ticker → spot price
    risk_free_rate: float = 0.053                          # annualised continuous
    regime_vol: float = 0.20                               # fallback vol if no override
    vol_overrides: Dict[str, float] = field(default_factory=dict)   # ticker → IV
    dividend_yields: Dict[str, float] = field(default_factory=dict) # ticker → q
    vix_level: float = 20.0
    yield_curve: Dict[str, float] = field(default_factory=dict)     # tenor → rate
    fx_rates: Dict[str, float] = field(default_factory=dict)        # pair → rate
    futures_prices: Dict[str, float] = field(default_factory=dict)  # ticker → F
    credit_spreads: Dict[str, float] = field(default_factory=dict)  # ticker → bps

    def vol_for(self, ticker: str) -> float:
        return self.vol_overrides.get(ticker, self.regime_vol)

    def spot_for(self, ticker: str) -> float:
        return self.spot_prices.get(ticker, 0.0)

    def div_yield_for(self, ticker: str) -> float:
        return self.dividend_yields.get(ticker, 0.0)

    def futures_for(self, ticker: str) -> float:
        """Return futures price, fallback to spot if not available."""
        return self.futures_prices.get(ticker, self.spot_prices.get(ticker, 0.0))


@dataclass
class PriceResult:
    """Standardised output from any asset-class pricer."""
    price: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    lambda_leverage: float = 0.0
    extended: Dict[str, float] = field(default_factory=dict)
    model_used: str = ""

    def is_valid(self) -> bool:
        return math.isfinite(self.price) and self.price >= 0.0


class BaseAssetPricer(ABC):
    """
    Abstract pricer.  Each asset class subclasses this and implements price().
    The engine dispatches via PRICER_REGISTRY[asset_class].
    """
    asset_class: str = ""

    def can_price(self, candidate: "InstrumentCandidate") -> bool:
        return candidate.asset_class == self.asset_class

    @abstractmethod
    def price(
        self,
        candidate: "InstrumentCandidate",
        ctx: PricingContext,
    ) -> PriceResult:
        """Return a PriceResult for the given candidate."""
        ...
