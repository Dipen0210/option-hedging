"""
Instrument selection layer — asset-class routing.

Active (Phase 1):
    options/    — BSM, Binomial CRR, Black-76, Greeks, position sizing

Stubs (Phase 2):
    futures/    — index futures, treasury futures, commodity futures
    forwards/   — FX forwards, commodity forwards
    swaps/      — interest rate swaps, CDS
    inverse_etfs/ — SH, PSQ, RWM, SDS, SQQQ
"""
from backend.instruments.base import InstrumentSelector
from backend.instruments.options.selector import OptionsSelector
from backend.instruments.futures.selector import FuturesSelector
from backend.instruments.forwards.selector import ForwardsSelector
from backend.instruments.swaps.selector import SwapsSelector
from backend.instruments.inverse_etfs.selector import InverseETFSelector

# Registry — add new selectors here as Phase 2+ is built
ALL_SELECTORS = [
    OptionsSelector(),
    FuturesSelector(),
    ForwardsSelector(),
    SwapsSelector(),
    InverseETFSelector(),
]

__all__ = [
    "InstrumentSelector",
    "OptionsSelector",
    "FuturesSelector",
    "ForwardsSelector",
    "SwapsSelector",
    "InverseETFSelector",
    "ALL_SELECTORS",
]
