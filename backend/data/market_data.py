import logging

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from backend.data.data_cache import cache_get_entry, cache_set
from backend.data import data_quality as dq
from backend.config.settings import settings

logger = logging.getLogger(__name__)

TTL = settings.cache_ttl_seconds

# Ranges the Yahoo chart API accepts. An unmapped period must raise rather than
# silently serve a shorter window than the caller asked for — a 10-year regime
# model quietly trained on 2 years of data is worse than a hard failure.
VALID_PERIODS = frozenset({
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
})

# Browser-like session for the direct Yahoo chart API, which the yfinance
# wrapper does not mediate. This bypasses Yahoo's cloud-IP blocking.
#
# NOTE: this session must NOT be handed to yf.Ticker()/yf.download(); modern
# yfinance requires a curl_cffi session and raises YFDataException when given a
# plain requests.Session. Let yfinance manage its own transport.
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://finance.yahoo.com/",
})


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def _yahoo_price_direct(ticker: str) -> float:
    """
    Direct call to Yahoo Finance chart API — more reliable than the yfinance
    wrapper on cloud server IPs where the cookie/crumb flow gets blocked.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = _SESSION.get(url, params={"interval": "1d", "range": "5d"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    closes = [c for c in closes if c is not None]
    if not closes:
        raise ValueError(f"No close prices in Yahoo chart API for {ticker}")
    return float(closes[-1])


def _yahoo_history_direct(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Direct Yahoo Finance chart API for historical OHLCV data, total-return
    adjusted.

    `includeAdjustedClose` is requested because this path and the yfinance
    fallback below must return the same series. The fallback passes
    `auto_adjust=True`; this path read `indicators.quote[0].close`, which is
    split-adjusted but *not* dividend-adjusted. So the same ticker returned a
    different history depending only on whether Yahoo's direct endpoint
    happened to be reachable — the two paths disagreed silently, and the
    primary one carried a dividend-sized error.

    The error is not small for the instruments that matter most here. Measured
    over 2024-10..2026-10, the unadjusted 1-year return for HYG is -4.25%
    against +1.63% adjusted — a sign flip — with TLT and LQD off by ~4.5pp and
    their 2-year price levels off by 9-12%. Three of the five MACRO_TICKERS in
    backend/data/covariance.py are among the worst affected, because a
    high-yield instrument's return is mostly coupon. Realized volatility
    barely moves (<= 0.37pp), which is why this survived the vol-focused
    checks.

    The factor is applied to Open/High/Low/Close together rather than
    substituting `adjclose` for `Close`. `adjclose` is scaled and the other
    three are not, so on this NVDA 5-year sample it sits outside [Low, High]
    on 107 of 1254 bars; swapping it in would have produced bars that fail the
    OHLC consistency check in backend/data/data_quality.py. Volume is left
    alone: the chart API's `quote.close` is already split-adjusted (NVDA's
    2024-06-10 10:1 split shows no discontinuity in it), so the factor carries
    dividends only, and dividends do not change share counts.
    """
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Unsupported history period {period!r} for {ticker}; "
            f"expected one of {sorted(VALID_PERIODS)}"
        )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = _SESSION.get(
        url,
        params={
            "interval": "1d",
            "range": period,
            "events": "div,split",
            "includeAdjustedClose": "true",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    # Yahoo stamps each daily bar at the session open in UTC, which differs by
    # asset class (equities 13:30/14:30, index products 07:00) and shifts with
    # DST. Normalizing to midnight makes cross-asset joins align on trade date;
    # without this, joining SPY against ^VIX yields an empty frame.
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()

    df = pd.DataFrame({
        "Open":   quotes.get("open", []),
        "High":   quotes.get("high", []),
        "Low":    quotes.get("low", []),
        "Close":  quotes.get("close", []),
        "Volume": quotes.get("volume", []),
    }, index=index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    return _apply_dividend_adjustment(df, result, ticker)


def _apply_dividend_adjustment(
    df: pd.DataFrame,
    result: dict,
    ticker: str,
) -> pd.DataFrame:
    """
    Scale OHLC onto a total-return basis using Yahoo's own adjusted close.

    The per-bar factor is `adjclose / close`, a step function of the dividends
    still to be paid, so forward-filling across bars Yahoo left blank is
    correct where interpolating a price would not be.

    When the endpoint returns no `adjclose` at all the bars are returned
    unadjusted and the shortfall is recorded, rather than being passed off as
    adjusted. For the index products this mostly affects (^VIX, ^GSPC) the two
    bases are identical anyway — an index pays no dividend — so this is a
    disclosure, not a degradation.
    """
    adj_block = (result.get("indicators", {}) or {}).get("adjclose") or []
    adj_raw = (adj_block[0] or {}).get("adjclose") if adj_block else None
    if not adj_raw:
        logger.info(
            "market_data: %s returned no adjusted close — OHLC left on a "
            "price basis, so dividends are not included in its returns",
            ticker,
        )
        dq.record(
            f"history_adjustment:{ticker}", dq.SOURCE_UNAVAILABLE,
            detail=(
                f"{ticker}: Yahoo returned no adjusted close; prices are "
                f"split-adjusted but not dividend-adjusted"
            ),
        )
        return df

    adj = pd.Series(adj_raw, index=pd.to_datetime(
        result["timestamp"], unit="s", utc=True).tz_convert(None).normalize())
    adj = adj[~adj.index.duplicated(keep="last")].sort_index().reindex(df.index)

    factor = (adj / df["Close"]).replace([np.inf, -np.inf], np.nan)
    factor = factor.ffill().bfill().fillna(1.0)
    if not np.isfinite(factor).all() or (factor <= 0).any():
        # A non-positive or non-finite factor would silently invert or blank a
        # price. Returning the unadjusted bars is the honest degradation.
        logger.warning(
            "market_data: %s adjusted-close factor is not usable — keeping "
            "unadjusted prices", ticker,
        )
        dq.record(
            f"history_adjustment:{ticker}", dq.SOURCE_UNAVAILABLE,
            detail=f"{ticker}: unusable adjusted-close factor; prices left unadjusted",
        )
        return df

    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col] * factor
    return out


def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Returns OHLCV DataFrame with a normalized (midnight) DatetimeIndex. Cached 1h."""
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Unsupported history period {period!r} for {ticker}; "
            f"expected one of {sorted(VALID_PERIODS)}"
        )

    entry = cache_get_entry("price_history", ticker, period)
    if entry is not None:
        cached, age = entry
        df_cached = pd.DataFrame(cached)
        if "Date" in df_cached.columns:
            df_cached["Date"] = pd.to_datetime(df_cached["Date"])
            df_cached = df_cached.set_index("Date")
        # Older cache entries may predate index normalization.
        df_cached.index = pd.to_datetime(df_cached.index).normalize()
        dq.record(f"history:{ticker}", dq.SOURCE_CACHE, age_seconds=age)
        return df_cached

    try:
        df = _yahoo_history_direct(ticker, period)
        source = dq.SOURCE_LIVE
    except Exception as exc:
        # Fall back to the yfinance wrapper (its own transport, no custom session).
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.index = pd.to_datetime(df.index).normalize()
            df = df[~df.index.duplicated(keep="last")].sort_index()
            source = dq.SOURCE_LIVE
        except Exception:
            dq.record(f"history:{ticker}", dq.SOURCE_UNAVAILABLE, detail=str(exc)[:120])
            raise ValueError(f"No price data found for {ticker}") from exc

    if df.empty:
        dq.record(f"history:{ticker}", dq.SOURCE_UNAVAILABLE, detail="empty series")
        raise ValueError(f"No price data found for {ticker}")

    last = df.index[-1]
    dq.record(f"history:{ticker}", source, as_of=str(last.date()))
    cache_set("price_history", ticker, period, value=df.reset_index().to_dict("records"), ttl=TTL)
    return df


def get_current_price(ticker: str) -> float:
    """Returns latest closing price. Cached 5 min."""
    entry = cache_get_entry("current_price", ticker)
    if entry is not None:
        dq.record(f"spot:{ticker}", dq.SOURCE_CACHE, age_seconds=entry[1])
        return float(entry[0])

    # Try direct API first, fall back to the yfinance wrapper.
    try:
        price = _yahoo_price_direct(ticker)
    except Exception as exc:
        try:
            hist = _ticker(ticker).history(period="5d")
            if hist.empty:
                raise ValueError(f"Cannot fetch current price for {ticker}")
            price = float(hist["Close"].iloc[-1])
        except Exception:
            dq.record(f"spot:{ticker}", dq.SOURCE_UNAVAILABLE, detail=str(exc)[:120])
            raise ValueError(f"Cannot fetch current price for {ticker}") from exc

    dq.record(f"spot:{ticker}", dq.SOURCE_LIVE)
    cache_set("current_price", ticker, value=price, ttl=300)
    return price


def get_dividend_yield(ticker: str) -> float:
    """Returns trailing dividend yield (annualized). Cached 1h."""
    entry = cache_get_entry("div_yield", ticker)
    if entry is not None:
        dq.record(f"dividend:{ticker}", dq.SOURCE_CACHE, age_seconds=entry[1])
        return float(entry[0])

    try:
        info = _ticker(ticker).info
        yield_val = float(info.get("dividendYield") or 0.0)
        if yield_val > 0.15:
            yield_val = yield_val / 100.0
        dq.record(f"dividend:{ticker}", dq.SOURCE_LIVE)
    except Exception as exc:
        # A zero dividend yield slightly overprices puts, so this fallback is
        # conservative for a hedge buyer — but it is still disclosed.
        yield_val = 0.0
        dq.record(
            f"dividend:{ticker}", dq.SOURCE_FALLBACK,
            detail=f"assumed 0% (lookup failed: {str(exc)[:60]})",
        )
    cache_set("div_yield", ticker, value=yield_val, ttl=TTL)
    return yield_val


# Used when the 3-month Treasury quote cannot be retrieved.
FALLBACK_RISK_FREE_RATE = 0.045


def get_risk_free_rate() -> float:
    """Returns 3-month Treasury yield as risk-free rate proxy. Cached 1h."""
    entry = cache_get_entry("risk_free_rate")
    if entry is not None:
        dq.record("risk_free_rate", dq.SOURCE_CACHE, age_seconds=entry[1])
        return float(entry[0])

    try:
        rate = _yahoo_price_direct("^IRX") / 100.0
        if not (0.0 <= rate < 0.25):
            raise ValueError(f"implausible ^IRX rate {rate}")
        dq.record("risk_free_rate", dq.SOURCE_LIVE)
        cache_set("risk_free_rate", value=rate, ttl=TTL)
        return rate
    except Exception as exc:
        dq.record(
            "risk_free_rate", dq.SOURCE_FALLBACK,
            detail=f"assumed {FALLBACK_RISK_FREE_RATE:.3f} (^IRX failed: {str(exc)[:60]})",
        )
        return FALLBACK_RISK_FREE_RATE


def get_historical_volatility(ticker: str, window: int = 20) -> float:
    """Annualized realized volatility from log returns over `window` trading days."""
    df = get_price_history(ticker, period="1y")
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(log_ret) < window:
        raise ValueError(
            f"Insufficient history for {ticker}: {len(log_ret)} returns, need {window}"
        )
    vol = float(log_ret.rolling(window).std().iloc[-1] * np.sqrt(252))
    if not np.isfinite(vol) or vol <= 0:
        raise ValueError(f"Invalid realized volatility for {ticker}: {vol}")
    return vol


def get_returns(ticker: str, period: str = "2y") -> pd.Series:
    """Daily log returns series, indexed by normalized trade date."""
    df = get_price_history(ticker, period=period)
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def align_series(**named_series: pd.Series) -> pd.DataFrame:
    """
    Join named series on their shared trade dates.

    Centralizes the cross-asset join so callers cannot reintroduce the
    timestamp-misalignment defect. Raises when the overlap is empty.
    """
    if not named_series:
        raise ValueError("align_series requires at least one series")

    normalized = {}
    for name, s in named_series.items():
        if s is None or len(s) == 0:
            raise ValueError(f"align_series: series {name!r} is empty")
        s = s.copy()
        idx = pd.to_datetime(s.index)
        # Drop the timezone before normalising. Yahoo returns tz-aware stamps
        # on some paths and naive ones on others, and pandas refuses to join
        # the two at all — `normalize()` alone keeps the tz, so a tz-aware and
        # a tz-naive series raised TypeError here instead of aligning. That
        # error is not the ValueError this function documents, so a caller
        # guarding against misalignment did not catch it and the series was
        # silently dropped. `tz_localize(None)` keeps the wall-clock date,
        # which for daily bars is the trading date we want to match on;
        # converting to UTC first would roll a late-session stamp onto the
        # next day.
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
        normalized[name] = s[~s.index.duplicated(keep="last")]

    df = pd.DataFrame(normalized).dropna()
    if df.empty:
        spans = {
            n: f"{s.index.min().date()}..{s.index.max().date()} ({len(s)})"
            for n, s in normalized.items()
        }
        raise ValueError(f"No overlapping dates across series: {spans}")
    return df
