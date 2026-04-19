"""
Pricing registry — maps asset_class string → pricer instance.

Usage:
    from backend.core.pricing import PRICER_REGISTRY
    result = PRICER_REGISTRY["equity"].price(candidate, ctx)
"""
from backend.core.pricing.equity_pricer import EquityOptionPricer
from backend.core.pricing.commodity_pricer import CommodityPricer
from backend.core.pricing.ir_pricer import InterestRatePricer
from backend.core.pricing.fx_pricer import FXPricer
from backend.core.pricing.credit_pricer import CreditPricer

PRICER_REGISTRY = {
    "equity":    EquityOptionPricer(),
    "commodity": CommodityPricer(),
    "bond":      InterestRatePricer(),
    "fx":        FXPricer(),
    "credit":    CreditPricer(),
}

__all__ = ["PRICER_REGISTRY"]
