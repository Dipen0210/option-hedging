"""
Sizer registry — maps asset_class string → sizer instance.

Usage:
    from backend.core.sizing import SIZER_REGISTRY
    result = SIZER_REGISTRY["equity"].size(candidate, price_result, ctx)
"""
from backend.core.sizing.equity_sizer import EquitySizer
from backend.core.sizing.commodity_sizer import CommoditySizer
from backend.core.sizing.ir_sizer import IRSizer
from backend.core.sizing.fx_sizer import FXSizer
from backend.core.sizing.credit_sizer import CreditSizer

SIZER_REGISTRY = {
    "equity":    EquitySizer(),
    "commodity": CommoditySizer(),
    "bond":      IRSizer(),
    "fx":        FXSizer(),
    "credit":    CreditSizer(),
}

__all__ = ["SIZER_REGISTRY"]
