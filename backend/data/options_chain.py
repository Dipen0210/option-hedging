import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import date, timedelta
from backend.data.data_cache import cache_get, cache_set
from backend.config.settings import settings

TTL = settings.cache_ttl_seconds


def get_options_chain(
    ticker: str,
    expiry_range_days: tuple = (30, 200),
) -> Dict:
    """
    Fetches option chain for a ticker, filtered to expiries within range.
    Returns: {
        'puts': pd.DataFrame,
        'calls': pd.DataFrame,
        'expiries': List[str],
        'selected_expiry': str
    }
    # TODO: v2 upgrade — use live broker options chain for real-time IV surface
    """
    cached = cache_get("options_chain", ticker, expiry_range_days[0], expiry_range_days[1])
    if cached:
        return {
            "puts": pd.DataFrame(cached["puts"]),
            "calls": pd.DataFrame(cached["calls"]),
            "expiries": cached["expiries"],
            "selected_expiry": cached["selected_expiry"],
        }

    t = yf.Ticker(ticker)
    today = date.today()
    min_date = today + timedelta(days=expiry_range_days[0])
    max_date = today + timedelta(days=expiry_range_days[1])

    valid_expiries = [
        e for e in (t.options or [])
        if min_date <= date.fromisoformat(e) <= max_date
    ]

    if not valid_expiries:
        raise ValueError(f"No options expiring in {expiry_range_days} days for {ticker}")

    # Select expiry closest to 90 days out
    target = today + timedelta(days=90)
    selected = min(valid_expiries, key=lambda e: abs((date.fromisoformat(e) - target).days))

    chain = t.option_chain(selected)
    puts = chain.puts.copy()
    calls = chain.calls.copy()

    # Clean up IV — replace zero/NaN with NaN for later estimation
    for df in [puts, calls]:
        df["impliedVolatility"] = df["impliedVolatility"].replace(0, np.nan)

    result = {
        "puts": puts.to_dict("records"),
        "calls": calls.to_dict("records"),
        "expiries": valid_expiries,
        "selected_expiry": selected,
    }
    cache_set("options_chain", ticker, expiry_range_days[0], expiry_range_days[1],
              value=result, ttl=TTL)

    return {
        "puts": puts,
        "calls": calls,
        "expiries": valid_expiries,
        "selected_expiry": selected,
    }


def get_expiry_dates_until(ticker: str, max_date, min_dte: int = 7) -> list:
    """
    Return all available option expiry dates for ticker from
    (today + min_dte) up to max_date (inclusive).

    Args:
        ticker:   underlying ticker
        max_date: date or ISO string — upper bound for expiry
        min_dte:  minimum days-to-expiry (skip near-dated weeklies)

    Returns:
        Sorted list of ISO date strings e.g. ["2025-05-16", "2025-06-20", ...]
    """
    from datetime import date, timedelta

    if isinstance(max_date, str):
        max_date = date.fromisoformat(max_date)

    cache_key = ("expiry_dates_until", ticker, max_date.isoformat())
    cached = cache_get(*cache_key)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker)
        today = date.today()
        floor = today + timedelta(days=min_dte)

        valid = sorted([
            e for e in (t.options or [])
            if floor <= date.fromisoformat(e) <= max_date
        ])

        cache_set(*cache_key, value=valid, ttl=TTL)
        return valid

    except Exception:
        return []


def get_strike_iv(
    ticker: str,
    expiry_str: str,
    strike: float,
    option_type: str = "put",
) -> Optional[float]:
    """
    Return the implied volatility for a specific strike and expiry from the
    real option chain using linear smile interpolation.

    Method:
      1. Sort chain by |strike - target|.
      2. If the target strike is within 1% of a real strike, return it directly.
      3. Otherwise find the two nearest strikes that bracket the target
         (one below, one above) and linearly interpolate their IVs.
      4. If no bracketing pair exists, fall back to the nearest strike within 15%.

    This correctly prices the volatility skew/smile — OTM puts have higher IV
    than ATM, so using ATM IV for all strikes systematically underprices them.

    Args:
        ticker:     underlying ticker  (e.g. "AAPL")
        expiry_str: ISO expiry date    (e.g. "2026-06-19")
        strike:     target strike price
        option_type: "put" or "call"

    Returns:
        Implied volatility as a decimal (e.g. 0.34 = 34%) or None on failure.
    """
    cache_key = ("strike_iv_v2", ticker, expiry_str, round(strike, 2), option_type)
    cached = cache_get(*cache_key)
    if cached is not None:
        return float(cached)

    try:
        t  = yf.Ticker(ticker)
        ch = t.option_chain(expiry_str)
        df = (ch.puts if option_type == "put" else ch.calls).copy()

        df["impliedVolatility"] = df["impliedVolatility"].replace(0, np.nan)
        df = df.dropna(subset=["impliedVolatility"])

        if df.empty:
            return None

        df = df.sort_values("strike").reset_index(drop=True)
        strikes = df["strike"].values
        ivs     = df["impliedVolatility"].values

        # ── Step 2: exact / near-exact match (within 1%) ─────────────────────
        df["dist"] = (df["strike"] - strike).abs()
        nearest_row = df.sort_values("dist").iloc[0]
        if nearest_row["dist"] <= strike * 0.01:
            iv = float(nearest_row["impliedVolatility"])
            if 0.01 < iv < 5.0:
                cache_set(*cache_key, value=iv, ttl=TTL)
                return iv

        # ── Step 3: linear interpolation between bracketing strikes ──────────
        below = df[df["strike"] <= strike].sort_values("strike", ascending=False)
        above = df[df["strike"] >  strike].sort_values("strike", ascending=True)

        if not below.empty and not above.empty:
            k_lo, iv_lo = float(below.iloc[0]["strike"]), float(below.iloc[0]["impliedVolatility"])
            k_hi, iv_hi = float(above.iloc[0]["strike"]), float(above.iloc[0]["impliedVolatility"])
            # Only interpolate if both bracket strikes are within 15% of target
            if (abs(k_lo - strike) <= strike * 0.15
                    and abs(k_hi - strike) <= strike * 0.15
                    and k_hi > k_lo):
                t_frac = (strike - k_lo) / (k_hi - k_lo)
                iv = iv_lo + t_frac * (iv_hi - iv_lo)
                if 0.01 < iv < 5.0:
                    cache_set(*cache_key, value=iv, ttl=TTL)
                    return iv

        # ── Step 4: nearest-strike fallback (within 15%) ─────────────────────
        if nearest_row["dist"] <= strike * 0.15:
            iv = float(nearest_row["impliedVolatility"])
            if 0.01 < iv < 5.0:
                cache_set(*cache_key, value=iv, ttl=TTL)
                return iv

    except Exception:
        pass

    return None


def check_option_liquidity(
    ticker: str,
    expiry_str: str,
    strike: float,
    option_type: str = "put",
    min_oi: int = 100,
    max_spread_pct: float = 0.05,
) -> Dict:
    """
    Return liquidity metrics for a specific option contract.

    Checks:
      - open_interest ≥ min_oi  (institutional threshold: 500; retail OK: 100)
      - bid/ask spread ≤ max_spread_pct of mid price (3–5% is acceptable)

    Returns:
        {
            "passes":         bool,
            "open_interest":  int,
            "volume":         int,
            "bid":            float,
            "ask":            float,
            "spread_pct":     float,   # (ask-bid)/mid
            "reason":         str,     # why it failed (empty string if passes)
        }
    """
    cache_key = ("option_liquidity", ticker, expiry_str, round(strike, 2), option_type)
    cached = cache_get(*cache_key)
    if cached is not None:
        return cached

    default_fail = {
        "passes": False, "open_interest": 0, "volume": 0,
        "bid": 0.0, "ask": 0.0, "spread_pct": 1.0,
        "reason": "chain unavailable",
    }

    try:
        t  = yf.Ticker(ticker)
        ch = t.option_chain(expiry_str)
        df = (ch.puts if option_type == "put" else ch.calls).copy()

        if df.empty:
            cache_set(*cache_key, value=default_fail, ttl=TTL)
            return default_fail

        df["dist"] = (df["strike"] - strike).abs()
        row = df.sort_values("dist").iloc[0]

        oi     = int(row.get("openInterest", 0) or 0)
        vol    = int(row.get("volume", 0) or 0)
        bid    = float(row.get("bid", 0.0) or 0.0)
        ask    = float(row.get("ask", 0.0) or 0.0)
        mid    = (bid + ask) / 2 if (bid + ask) > 0 else 0.0
        spread = (ask - bid) / mid if mid > 0 else 1.0

        reasons = []
        if oi < min_oi:
            reasons.append(f"OI={oi} < {min_oi}")
        if spread > max_spread_pct:
            reasons.append(f"spread={spread:.1%} > {max_spread_pct:.1%}")

        result = {
            "passes":        len(reasons) == 0,
            "open_interest": oi,
            "volume":        vol,
            "bid":           bid,
            "ask":           ask,
            "spread_pct":    round(spread, 4),
            "reason":        "; ".join(reasons),
        }
        cache_set(*cache_key, value=result, ttl=TTL)
        return result

    except Exception:
        cache_set(*cache_key, value=default_fail, ttl=TTL)
        return default_fail


def get_atm_iv(ticker: str, option_type: str = "put") -> float:
    """
    Returns at-the-money implied volatility.
    Falls back to historical vol if chain unavailable.
    # TODO: v2 upgrade — fit full IV surface (SABR/SVI)
    """
    from backend.data.market_data import get_current_price, get_historical_volatility
    try:
        chain = get_options_chain(ticker)
        spot = get_current_price(ticker)
        df = chain["puts"] if option_type == "put" else chain["calls"]

        # Find closest strike to spot
        df = df.copy()
        df["dist"] = (df["strike"] - spot).abs()
        atm = df.sort_values("dist").head(3)
        iv = atm["impliedVolatility"].dropna().mean()
        if iv and iv > 0.01:
            return float(iv)
    except Exception:
        pass

    return get_historical_volatility(ticker)
