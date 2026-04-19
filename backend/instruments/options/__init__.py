"""Options instrument selection and pricing."""
from backend.instruments.options.selector import OptionsSelector
from backend.instruments.options.bsm_pricer import bsm_price, bsm_greeks, implied_vol
from backend.instruments.options.binomial_pricer import crr_price, crr_greeks
from backend.instruments.options.black76_pricer import black76_price, black76_greeks
from backend.instruments.options.greeks import net_greeks, collar_greeks, bear_put_spread_greeks
from backend.instruments.options.position_sizer import compute_position, partial_hedge_series

__all__ = [
    "OptionsSelector",
    "bsm_price", "bsm_greeks", "implied_vol",
    "crr_price", "crr_greeks",
    "black76_price", "black76_greeks",
    "net_greeks", "collar_greeks", "bear_put_spread_greeks",
    "compute_position", "partial_hedge_series",
]
