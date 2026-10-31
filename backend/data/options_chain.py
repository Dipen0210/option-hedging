"""
Options chain retrieval.

Every contract this module returns is a real listed contract: the strike, expiry,
bid, ask, implied volatility and open interest all come from the live chain.
When the chain cannot be retrieved, callers get an explicit
``OptionsDataUnavailable`` error or ``None`` — never a synthesized strike or a
realized-volatility stand-in dressed up as implied volatility.

IMPORTANT: do not pass a custom ``requests.Session`` to yfinance. Modern
yfinance requires a curl_cffi session and raises YFDataException for a plain
requests.Session, which silently emptied every chain lookup in this module.
"""

import logging

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date, timedelta
from backend.data.data_cache import cache_get, cache_get_entry, cache_set
from backend.data import data_quality as dq
from backend.config.settings import settings

logger = logging.getLogger(__name__)

TTL = settings.cache_ttl_seconds
# Transport failures are retried quickly rather than being cached for an hour.
FAILURE_TTL = 60
# Quotes move; chains are cached briefly so one analysis sees a consistent snapshot.
CHAIN_TTL = 300

# Plausible implied-volatility band for a listed equity / ETF option.
#
# The previous band was 0.01 < iv < 5.0, which accepted obvious junk rows. A
# live SPY 2027-02-26 749 put came back with bid=0, ask=0 and IV=0.0156 — a
# 1.56% implied vol. No listed equity or index option trades there; the row is
# a data artifact. Pricing on it produced a premium of effectively $0 and a
# delta of ~1e-5, and the sizer then asked for 1,500,000 contracts.
#
# 5% is below the lowest IV any liquid index option sustains, so the floor
# rejects artifacts without discarding genuinely quiet markets. 3.0 (300%) is
# above anything outside a distressed single name.
MIN_PLAUSIBLE_IV = 0.05
MAX_PLAUSIBLE_IV = 3.0


def _is_plausible_iv(iv) -> bool:
    """True when an implied-vol reading could have come from a real market."""
    try:
        v = float(iv)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and MIN_PLAUSIBLE_IV < v < MAX_PLAUSIBLE_IV


def _quoted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows with a two-sided quote — the only rows whose implied volatility is a
    market reading.

    The provider does not publish IV; it solves for it, and when there is no
    quote it solves against `lastPrice`, which can be hours or weeks stale.
    Outside 09:30-16:00 ET bid and ask are cleared for the whole chain, and
    the solver then fails in a visible pattern: successive halvings of 1/16
    (0.0625, 0.03126, 0.015635, 0.007822, ... down to 1e-05) applied in blocks
    across adjacent strikes. Measured on the real SPY 2027-01-29 put chain
    outside the session, 164 of 165 strikes carried one of those values and
    reported zero open interest; the single strike with a live two-sided quote
    (674, bid 6.40 / ask 6.52, OI 120) was also the only one with a credible
    IV, 0.2502.

    The top rung of that ladder, 0.0625, clears MIN_PLAUSIBLE_IV, so the
    magnitude band alone cannot reject it: a 90%-moneyness three-month SPY put
    was being priced at 6.25% vol against an at-the-money reading of 16.4%, an
    inverted skew that no index market produces. The quote is what makes the
    number a measurement, so IV is read only where one exists. Where none
    does, these functions return None and the caller descends the documented
    provenance ladder with the substitution disclosed, rather than pricing a
    hedge off a solver artifact.
    """
    if "bid" not in df.columns or "ask" not in df.columns:
        return df.iloc[0:0]
    bid = pd.to_numeric(df["bid"], errors="coerce")
    ask = pd.to_numeric(df["ask"], errors="coerce")
    return df[(bid > 0) & (ask > 0) & (ask >= bid)]


class OptionsDataUnavailable(RuntimeError):
    """Raised when an option chain cannot be retrieved for a ticker/expiry."""


def _record_chain_failure(ticker: str, detail: str) -> None:
    dq.record(f"chain:{ticker}", dq.SOURCE_UNAVAILABLE, detail=detail[:120])


# ── Expiries ─────────────────────────────────────────────────────────────────
def get_available_expiries(ticker: str) -> List[str]:
    """
    All expiry dates listed for `ticker`, as ISO strings.

    Raises OptionsDataUnavailable if the chain cannot be reached, so an outage
    is never mistaken for "this underlying has no options".
    """
    entry = cache_get_entry("expiries_all_v2", ticker)
    if entry is not None:
        dq.record(f"chain:{ticker}", dq.SOURCE_CACHE, age_seconds=entry[1])
        return list(entry[0])

    try:
        expiries = list(yf.Ticker(ticker).options or [])
    except Exception as exc:
        _record_chain_failure(ticker, f"expiry lookup failed: {exc}")
        raise OptionsDataUnavailable(f"Option chain unavailable for {ticker}: {exc}") from exc

    if not expiries:
        _record_chain_failure(ticker, "no listed expiries returned")
        raise OptionsDataUnavailable(f"No listed option expiries for {ticker}")

    dq.record(f"chain:{ticker}", dq.SOURCE_LIVE)
    cache_set("expiries_all_v2", ticker, value=expiries, ttl=CHAIN_TTL)
    return expiries


def get_expiry_dates_until(ticker: str, max_date, min_dte: int = 7) -> List[str]:
    """
    Real listed expiries between (today + min_dte) and max_date inclusive.

    Returns [] only when the chain was retrieved successfully but holds no
    expiry in the window. Raises OptionsDataUnavailable when the chain itself
    could not be fetched.
    """
    if isinstance(max_date, str):
        max_date = date.fromisoformat(max_date)

    expiries = get_available_expiries(ticker)
    floor = date.today() + timedelta(days=min_dte)
    return sorted(
        e for e in expiries
        if floor <= date.fromisoformat(e) <= max_date
    )


def select_expiry_nearest(ticker: str, target_days: int, min_dte: int = 7) -> Optional[str]:
    """Listed expiry closest to `target_days` out, or None if none qualify."""
    expiries = [
        e for e in get_available_expiries(ticker)
        if date.fromisoformat(e) >= date.today() + timedelta(days=min_dte)
    ]
    if not expiries:
        return None
    target = date.today() + timedelta(days=target_days)
    return min(expiries, key=lambda e: abs((date.fromisoformat(e) - target).days))


def _assess_quote_health(ticker: str, expiry_str: str, side: str, df: pd.DataFrame) -> None:
    """
    Tell "this contract is illiquid" apart from "the provider sent no quotes".

    A chain side where *not one* strike has a two-sided quote and the whole
    side reports zero open interest is not a market observation — it is an
    unusable snapshot. Outside US market hours Yahoo clears bid/ask and does
    not republish open interest until the next session, so a 1-DTE SPY chain
    comes back with 110 strikes, 0 bids and 0 open interest. Treating that as
    "illiquid" lets the pipeline fall through to theoretical pricing and then
    present the result with spread 0.0000 and OI 0, which reads to a user as a
    perfectly tight market rather than as no market data at all.

    Recording it as UNAVAILABLE (not a fallback) puts it in the response's
    degraded fields so the user is told the quotes are missing.
    """
    if df.empty:
        return

    bid = pd.to_numeric(df.get("bid"), errors="coerce").fillna(0.0)
    ask = pd.to_numeric(df.get("ask"), errors="coerce").fillna(0.0)
    two_sided = int(((bid > 0) & (ask > 0)).sum())
    total_oi = int(pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0).sum())

    if two_sided == 0 and total_oi == 0:
        dq.record(
            f"quotes:{ticker}", dq.SOURCE_UNAVAILABLE,
            detail=(
                f"{ticker} {expiry_str} {side}: provider returned {len(df)} strikes "
                f"with no two-sided quote and no open interest. Option quotes are "
                f"not published outside US market hours (09:30-16:00 ET); any "
                f"premium for this expiry is theoretical, not executable."
            ),
        )



# ── Chain snapshots ──────────────────────────────────────────────────────────
def _chain_frame(ticker: str, expiry_str: str, option_type: str) -> pd.DataFrame:
    """
    One side of a real chain as a DataFrame, cached briefly.

    Raises OptionsDataUnavailable on transport failure or empty chain.
    """
    side = "puts" if option_type == "put" else "calls"
    entry = cache_get_entry("chain_side_v2", ticker, expiry_str, side)
    if entry is not None:
        df = pd.DataFrame(entry[0])
        dq.record(f"chain:{ticker}", dq.SOURCE_CACHE, age_seconds=entry[1])
        if df.empty:
            raise OptionsDataUnavailable(f"No {side} listed for {ticker} {expiry_str}")
        _assess_quote_health(ticker, expiry_str, side, df)
        return df

    try:
        chain = yf.Ticker(ticker).option_chain(expiry_str)
        df = (chain.puts if option_type == "put" else chain.calls).copy()
    except Exception as exc:
        _record_chain_failure(ticker, f"{expiry_str} {side}: {exc}")
        raise OptionsDataUnavailable(
            f"Option chain unavailable for {ticker} {expiry_str}: {exc}"
        ) from exc

    if df.empty:
        _record_chain_failure(ticker, f"{expiry_str} {side}: empty")
        raise OptionsDataUnavailable(f"No {side} listed for {ticker} {expiry_str}")

    keep = [c for c in
            ["strike", "bid", "ask", "lastPrice", "impliedVolatility",
             "openInterest", "volume", "contractSymbol"]
            if c in df.columns]
    df = df[keep].copy()
    df["impliedVolatility"] = df["impliedVolatility"].replace(0, np.nan)

    dq.record(f"chain:{ticker}", dq.SOURCE_LIVE, as_of=expiry_str)
    _assess_quote_health(ticker, expiry_str, side, df)
    cache_set("chain_side_v2", ticker, expiry_str, side,
              value=df.to_dict("records"), ttl=CHAIN_TTL)
    return df


def get_listed_strikes(ticker: str, expiry_str: str, option_type: str = "put") -> List[float]:
    """The actual strike ladder for a real expiry — never a synthesized grid."""
    df = _chain_frame(ticker, expiry_str, option_type)
    return sorted(float(s) for s in df["strike"].dropna().unique())


def select_listed_strike(
    ticker: str,
    expiry_str: str,
    target_strike: float,
    option_type: str = "put",
    max_distance_pct: float = 0.10,
) -> Optional[float]:
    """
    Nearest real listed strike to `target_strike`, or None if the closest listed
    strike is further than `max_distance_pct` away.
    """
    strikes = get_listed_strikes(ticker, expiry_str, option_type)
    if not strikes:
        return None
    nearest = min(strikes, key=lambda k: abs(k - target_strike))
    if abs(nearest - target_strike) > target_strike * max_distance_pct:
        return None
    return nearest


def get_contract_quote(
    ticker: str,
    expiry_str: str,
    strike: float,
    option_type: str = "put",
    max_distance_pct: float = 0.01,
) -> Optional[Dict]:
    """
    The live quote for one real contract.

    `strike` must already be a listed strike (see select_listed_strike);
    max_distance_pct guards against silently quoting a different contract.

    Returns None when that contract is not present in the chain. Raises
    OptionsDataUnavailable if the chain itself is unreachable.
    """
    df = _chain_frame(ticker, expiry_str, option_type)
    df = df.assign(_dist=(df["strike"] - strike).abs())
    row = df.sort_values("_dist").iloc[0]

    if float(row["_dist"]) > strike * max_distance_pct:
        return None

    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    iv_raw = row.get("impliedVolatility")
    iv = float(iv_raw) if iv_raw is not None and pd.notna(iv_raw) else None

    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
    spread_pct = (ask - bid) / mid if mid > 0 else None

    return {
        "ticker": ticker,
        "expiry": expiry_str,
        "strike": float(row["strike"]),
        "option_type": option_type,
        "contract_symbol": row.get("contractSymbol") or "",
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
        # Only where a two-sided quote backs it — see `_quoted`.
        "implied_volatility": (
            iv if (bid > 0 and ask > 0 and _is_plausible_iv(iv)) else None
        ),
        "open_interest": int(row.get("openInterest") or 0),
        "volume": int(row.get("volume") or 0),
        "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "has_two_sided_quote": bid > 0 and ask > 0,
    }


# ── Implied volatility ───────────────────────────────────────────────────────
def get_strike_iv(
    ticker: str,
    expiry_str: str,
    strike: float,
    option_type: str = "put",
) -> Optional[float]:
    """
    Implied volatility for a specific strike and expiry from the real chain,
    using linear smile interpolation so the volatility skew is respected.

    Method:
      1. If the target strike is within 1% of a quoted strike, use its IV.
      2. Otherwise linearly interpolate the two nearest bracketing strikes
         (both must be within 15% of the target).
      3. Otherwise fall back to the nearest quoted strike within 2% — close
         enough that the skew difference is small. A more distant quote is
         not this contract's IV and is not reported as one.

    Returns None when no usable IV exists. Callers must handle None explicitly —
    substituting realized volatility here would misprice every hedge.
    """
    # v4: entries written under v3 were computed before implied vol required a
    # two-sided quote behind it and before the borrow tolerance was narrowed,
    # so they can hold the provider's solver artifacts. The cache is on disk
    # and outlives a deploy; serving those values would reintroduce the defect
    # for as long as they live, so the version is bumped rather than relying
    # on expiry.
    cache_key = ("strike_iv_v4", ticker, expiry_str, round(strike, 2), option_type)
    cached = cache_get(*cache_key)
    if cached is not None:
        return float(cached)

    try:
        df = _chain_frame(ticker, expiry_str, option_type)
    except OptionsDataUnavailable:
        return None

    df = df.dropna(subset=["impliedVolatility"])

    # Screen the inputs, not just the result. `_accept` below checks the
    # number this function returns, which is enough for the exact-match and
    # nearest-strike paths but not for interpolation: a stale 0.00001 next to
    # a genuine 0.30 interpolates to 0.150005, which sits comfortably inside
    # the plausibility band and was returned as `contract_iv` — the top of the
    # provenance ladder, the reading the UI calls "this strike's own implied
    # vol" — while being half composed of an artifact. Dropping implausible
    # rows first means the smile is built only from quotes that could have
    # come from a real market.
    n_quoted_before = len(df)
    df = _quoted(df)
    if df.empty:
        dq.record(
            f"iv:{ticker}:{expiry_str}", dq.SOURCE_UNAVAILABLE,
            detail=(
                f"no two-sided quote on any of {n_quoted_before} strikes; "
                f"implied vol is not a market reading"
            ),
        )
        return None

    n_before = len(df)
    df = df[df["impliedVolatility"].apply(_is_plausible_iv)]
    if len(df) < n_before:
        logger.debug(
            "%s %s: dropped %d implausible IV quote(s) before interpolation",
            ticker, expiry_str, n_before - len(df),
        )

    if df.empty:
        dq.record(f"iv:{ticker}:{expiry_str}", dq.SOURCE_UNAVAILABLE,
                  detail="chain has no usable implied volatility")
        return None

    df = df.sort_values("strike").reset_index(drop=True)
    df["dist"] = (df["strike"] - strike).abs()
    nearest = df.sort_values("dist").iloc[0]

    def _accept(iv: float) -> Optional[float]:
        if _is_plausible_iv(iv):
            cache_set(*cache_key, value=iv, ttl=CHAIN_TTL)
            dq.record(f"iv:{ticker}:{expiry_str}", dq.SOURCE_LIVE)
            return iv
        return None

    # 1 — near-exact match
    if nearest["dist"] <= strike * 0.01:
        got = _accept(float(nearest["impliedVolatility"]))
        if got is not None:
            return got

    # 2 — interpolate across the smile
    below = df[df["strike"] <= strike].sort_values("strike", ascending=False)
    above = df[df["strike"] > strike].sort_values("strike", ascending=True)
    if not below.empty and not above.empty:
        k_lo, iv_lo = float(below.iloc[0]["strike"]), float(below.iloc[0]["impliedVolatility"])
        k_hi, iv_hi = float(above.iloc[0]["strike"]), float(above.iloc[0]["impliedVolatility"])
        if (abs(k_lo - strike) <= strike * 0.15
                and abs(k_hi - strike) <= strike * 0.15
                and k_hi > k_lo):
            t_frac = (strike - k_lo) / (k_hi - k_lo)
            got = _accept(iv_lo + t_frac * (iv_hi - iv_lo))
            if got is not None:
                return got

    # 3 — nearest quoted strike, but only a genuinely nearby one.
    #
    # This path borrows one strike's IV wholesale, and the caller labels the
    # result `contract_iv` — "this strike's own implied vol". At the old 15%
    # tolerance that was 97 points on SPY, across which the skew changes by
    # several volatility points, so an at-the-money reading could be reported
    # as a deep-out-of-the-money contract's own IV with nothing to flag the
    # substitution. Interpolation (path 2) earns its 15% because two
    # bracketing quotes describe the smile between them; a single distant
    # quote does not. Beyond this band the honest answer is None, and the
    # caller descends to `atm_iv` — which means the same number, disclosed
    # as an at-the-money substitute rather than presented as measured here.
    NEAREST_STRIKE_TOLERANCE = 0.02
    if nearest["dist"] <= strike * NEAREST_STRIKE_TOLERANCE:
        got = _accept(float(nearest["impliedVolatility"]))
        if got is not None:
            return got

    dq.record(
        f"iv:{ticker}:{expiry_str}", dq.SOURCE_UNAVAILABLE,
        detail=(
            f"no quoted IV within {NEAREST_STRIKE_TOLERANCE:.0%} of strike "
            f"{strike} and none bracketing it"
        ),
    )
    return None


def get_atm_iv(ticker: str, option_type: str = "put", target_days: int = 90) -> Optional[float]:
    """
    At-the-money implied volatility from the real chain, or None.

    Returns None rather than substituting realized volatility: index options
    trade at a persistent premium to realized vol, so that substitution
    understates hedge premiums by tens of percent. Callers decide whether a
    disclosed realized-vol fallback is acceptable for their purpose.
    """
    from backend.data.market_data import get_current_price

    try:
        expiry = select_expiry_nearest(ticker, target_days)
        if expiry is None:
            return None
        df = _chain_frame(ticker, expiry, option_type).dropna(subset=["impliedVolatility"])
        # Same precondition as `get_strike_iv`: an implied vol with no quote
        # behind it is the provider's solver output, not the market's view.
        # Averaging the three nearest strikes made this worse rather than
        # better — one artifact among three pulled the mean below the
        # plausibility floor, so the whole reading was discarded and the
        # reason recorded as "implausible" rather than "unquoted".
        df = _quoted(df)
        df = df[df["impliedVolatility"].apply(_is_plausible_iv)]
        if df.empty:
            dq.record(f"atm_iv:{ticker}", dq.SOURCE_UNAVAILABLE,
                      detail="no two-sided quote with a usable implied vol")
            return None
        spot = get_current_price(ticker)
        df = df.assign(dist=(df["strike"] - spot).abs()).sort_values("dist")
        # Only strikes near the money; after screening, the nearest surviving
        # quotes can sit far from spot, and a deep-OTM reading is not an ATM
        # one. 10% is wide enough to find a quote and narrow enough that the
        # skew has not meaningfully changed the level.
        near = df[df["dist"] <= spot * 0.10]
        if near.empty:
            dq.record(f"atm_iv:{ticker}", dq.SOURCE_UNAVAILABLE,
                      detail="no quoted strike within 10% of spot")
            return None
        iv = float(near.head(3)["impliedVolatility"].mean())
    except OptionsDataUnavailable:
        return None
    except Exception as exc:
        logger.warning("atm_iv %s failed: %s", ticker, exc)
        return None

    if not _is_plausible_iv(iv):
        dq.record(f"atm_iv:{ticker}", dq.SOURCE_UNAVAILABLE,
                  detail=f"averaged implied vol {iv:.4f} outside plausible band")
        return None
    dq.record(f"atm_iv:{ticker}", dq.SOURCE_LIVE)
    return iv


# ── Liquidity ────────────────────────────────────────────────────────────────
def check_option_liquidity(
    ticker: str,
    expiry_str: str,
    strike: float,
    option_type: str = "put",
    min_oi: int = 100,
    max_spread_pct: float = 0.05,
) -> Dict:
    """
    Liquidity metrics for one real contract.

    Checks open interest >= min_oi and bid/ask spread <= max_spread_pct of mid.
    `available` distinguishes a genuine liquidity rejection from a data outage:
    a contract that failed the OI test is a different fact from a chain we could
    not read, and they must not look the same to a user.
    """
    cache_key = ("option_liquidity_v2", ticker, expiry_str, round(strike, 2), option_type)
    cached = cache_get(*cache_key)
    if cached is not None:
        return cached

    unavailable = {
        "passes": False, "available": False, "open_interest": 0, "volume": 0,
        "bid": 0.0, "ask": 0.0, "spread_pct": None,
        "reason": "option chain unavailable",
    }

    try:
        quote = get_contract_quote(ticker, expiry_str, strike, option_type)
    except OptionsDataUnavailable as exc:
        unavailable["reason"] = f"option chain unavailable: {str(exc)[:80]}"
        cache_set(*cache_key, value=unavailable, ttl=FAILURE_TTL)
        return unavailable

    if quote is None:
        result = dict(unavailable, reason=f"strike {strike} not listed for {expiry_str}")
        cache_set(*cache_key, value=result, ttl=CHAIN_TTL)
        return result

    oi = quote["open_interest"]
    spread = quote["spread_pct"]

    reasons = []
    if oi < min_oi:
        reasons.append(f"OI={oi} < {min_oi}")
    if not quote["has_two_sided_quote"]:
        reasons.append("no two-sided quote")
    elif spread is not None and spread > max_spread_pct:
        reasons.append(f"spread={spread:.1%} > {max_spread_pct:.1%}")

    result = {
        "passes": len(reasons) == 0,
        "available": True,
        "open_interest": oi,
        "volume": quote["volume"],
        "bid": quote["bid"],
        "ask": quote["ask"],
        "spread_pct": spread,
        "reason": "; ".join(reasons),
    }
    cache_set(*cache_key, value=result, ttl=CHAIN_TTL)
    return result

