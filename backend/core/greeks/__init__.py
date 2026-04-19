"""
Greeks registry — maps asset_class string → greeks engine instance.

Usage:
    from backend.core.greeks import GREEKS_REGISTRY
    result = GREEKS_REGISTRY["equity"].compute(candidate, price_result, ctx, n)
"""
from backend.core.greeks.equity_greeks import EquityGreeksEngine
from backend.core.greeks.commodity_greeks import CommodityGreeksEngine
from backend.core.greeks.ir_greeks import IRGreeksEngine
from backend.core.greeks.fx_greeks import FXGreeksEngine
from backend.core.greeks.credit_greeks import CreditGreeksEngine

GREEKS_REGISTRY = {
    "equity":    EquityGreeksEngine(),
    "commodity": CommodityGreeksEngine(),
    "bond":      IRGreeksEngine(),
    "fx":        FXGreeksEngine(),
    "credit":    CreditGreeksEngine(),
}

__all__ = ["GREEKS_REGISTRY"]
