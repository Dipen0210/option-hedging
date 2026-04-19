import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from backend.data.data_cache import cache_get, cache_set
from backend.config.settings import settings

TTL = settings.cache_ttl_seconds

# Yield curve tickers via yfinance
YIELD_TICKERS = {
    "3m":  "^IRX",
    "2y":  "^TYX",   # approximate — yfinance doesn't have 2Y directly
    "5y":  "^FVX",
    "10y": "^TNX",
    "30y": "^TYX",
}


def get_yield_curve() -> Dict[str, float]:
    """
    Returns current yield curve as {tenor: rate}.
    Rates are decimal (e.g. 0.045 = 4.5%).
    Cached 1h.
    """
    cached = cache_get("yield_curve")
    if cached:
        return cached

    curve = {}
    for tenor, ticker in YIELD_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curve[tenor] = float(hist["Close"].iloc[-1]) / 100.0
        except Exception:
            pass

    if not curve:
        curve = {"3m": 0.045, "5y": 0.043, "10y": 0.042, "30y": 0.044}

    cache_set("yield_curve", value=curve, ttl=TTL)
    return curve


def get_yield_curve_slope() -> float:
    """
    10Y - 3M spread. Negative = inverted (recession signal).
    Used as a regime feature for HDBSCAN.
    """
    curve = get_yield_curve()
    return curve.get("10y", 0.042) - curve.get("3m", 0.045)


def get_sofr_rate() -> float:
    """
    Returns SOFR rate. Uses 3M T-bill as proxy for v1.
    # TODO: v2 upgrade — pull directly from FRED API (SOFR series)
    """
    cached = cache_get("sofr_rate")
    if cached is not None:
        return float(cached)

    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100.0
            cache_set("sofr_rate", value=rate, ttl=TTL)
            return rate
    except Exception:
        pass

    return 0.045
