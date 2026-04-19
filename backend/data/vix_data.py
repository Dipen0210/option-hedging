import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict
from backend.data.data_cache import cache_get, cache_set
from backend.config.settings import settings

TTL = settings.cache_ttl_seconds

VIX_TICKERS = {
    "spot": "^VIX",
    "vix3m": "^VIX3M",
    "vix6m": "^VIX6M",
}


def get_vix() -> Dict:
    """
    Returns VIX spot + term structure.
    {
        'level': float,
        'vix3m': float,
        'vix6m': float,
        'term_structure': 'contango' | 'backwardation' | 'flat',
        'vix_5d_change': float,
        'put_call_ratio': float  (SPY proxy)
    }
    """
    cached = cache_get("vix_data")
    if cached:
        return cached

    result = {}

    for key, ticker in VIX_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="10d")
            if not hist.empty:
                result[key] = float(hist["Close"].iloc[-1])
                if key == "spot" and len(hist) >= 5:
                    result["vix_5d_change"] = float(
                        hist["Close"].iloc[-1] - hist["Close"].iloc[-5]
                    )
        except Exception:
            result[key] = None

    # Rename spot → level
    result["level"] = result.pop("spot", 20.0)
    result.setdefault("vix3m", result["level"] * 1.02)
    result.setdefault("vix6m", result["level"] * 1.04)
    result.setdefault("vix_5d_change", 0.0)

    # Term structure classification
    spot = result["level"]
    v3m = result["vix3m"]
    if v3m > spot * 1.02:
        result["term_structure"] = "contango"
    elif v3m < spot * 0.98:
        result["term_structure"] = "backwardation"
    else:
        result["term_structure"] = "flat"

    # SPY put/call ratio proxy (volume-based)
    result["put_call_ratio"] = _get_put_call_ratio()

    cache_set("vix_data", value=result, ttl=300)   # 5-min TTL for VIX
    return result


def _get_put_call_ratio() -> float:
    """
    Approximate put/call ratio from SPY options volume.
    Returns 1.0 as fallback if unavailable.
    """
    try:
        spy = yf.Ticker("SPY")
        expiries = spy.options
        if not expiries:
            return 1.0
        chain = spy.option_chain(expiries[0])
        put_vol = chain.puts["volume"].fillna(0).sum()
        call_vol = chain.calls["volume"].fillna(0).sum()
        if call_vol > 0:
            return round(float(put_vol / call_vol), 3)
    except Exception:
        pass
    return 1.0


def classify_vix_regime(vix_level: float) -> str:
    """Rule-based VIX regime label used to annotate HDBSCAN clusters."""
    if vix_level < 15:
        return "low_vol"
    elif vix_level < 25:
        return "mid_vol"
    else:
        return "high_vol"
