"""
VIX spot and term structure.

Reads through market_data.get_price_history, which uses the direct Yahoo chart
API (reliable on cloud IPs) and records provenance. Unavailable levels are
reported as None and disclosed — never replaced with an invented index level,
because the VIX level gates strategy selection and anomaly detection.
"""

from typing import Dict, Optional

from backend.data.data_cache import cache_get_entry, cache_set
from backend.data import data_quality as dq
from backend.config.settings import settings

TTL = settings.cache_ttl_seconds

VIX_TICKERS = {
    "level": "^VIX",
    "vix3m": "^VIX3M",
    "vix6m": "^VIX6M",
}


def _last_close(ticker: str) -> Optional[float]:
    from backend.data.market_data import get_price_history
    try:
        close = get_price_history(ticker, period="1mo")["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


def get_vix() -> Dict:
    """
    Returns VIX spot + term structure.

    {
        'level': float | None,
        'vix3m': float | None,
        'vix6m': float | None,
        'term_structure': 'contango' | 'backwardation' | 'flat' | None,
        'vix_5d_change': float | None,
        'put_call_ratio': float | None,
        'available': bool,
    }
    """
    entry = cache_get_entry("vix_data_v2")
    if entry is not None:
        dq.record("vix", dq.SOURCE_CACHE, age_seconds=entry[1])
        return entry[0]

    from backend.data.market_data import get_price_history

    result: Dict = {}
    for key, ticker in VIX_TICKERS.items():
        result[key] = _last_close(ticker)

    # 5-day change needs the series, not just the last point.
    result["vix_5d_change"] = None
    try:
        close = get_price_history("^VIX", period="1mo")["Close"].dropna()
        if len(close) >= 6:
            result["vix_5d_change"] = float(close.iloc[-1] - close.iloc[-6])
    except Exception:
        pass

    spot, v3m = result.get("level"), result.get("vix3m")
    if spot is not None and v3m is not None and spot > 0:
        if v3m > spot * 1.02:
            result["term_structure"] = "contango"
        elif v3m < spot * 0.98:
            result["term_structure"] = "backwardation"
        else:
            result["term_structure"] = "flat"
    else:
        result["term_structure"] = None

    result["put_call_ratio"] = _get_put_call_ratio()
    result["available"] = result.get("level") is not None

    if result["available"]:
        dq.record("vix", dq.SOURCE_LIVE)
    else:
        dq.record("vix", dq.SOURCE_UNAVAILABLE, detail="^VIX quote unavailable")

    cache_set("vix_data_v2", value=result, ttl=300)   # 5-min TTL for VIX
    return result


def _get_put_call_ratio() -> Optional[float]:
    """
    Approximate put/call ratio from front-expiry SPY option volume.
    Returns None when the chain is unavailable rather than an invented 1.0.
    """
    try:
        from backend.data.options_chain import get_available_expiries, _chain_frame
        expiries = get_available_expiries("SPY")
        if not expiries:
            return None
        puts = _chain_frame("SPY", expiries[0], "put")
        calls = _chain_frame("SPY", expiries[0], "call")
        put_vol = float(puts["volume"].fillna(0).sum())
        call_vol = float(calls["volume"].fillna(0).sum())
        if call_vol > 0:
            return round(put_vol / call_vol, 3)
    except Exception:
        pass
    return None


def classify_vix_regime(vix_level: float) -> str:
    """Rule-based VIX regime label, used as a disclosed fallback when the
    trained percentile thresholds are unavailable."""
    if vix_level < 15:
        return "low_vol"
    elif vix_level < 25:
        return "mid_vol"
    else:
        return "high_vol"
