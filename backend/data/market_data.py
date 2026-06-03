import yfinance as yf
import pandas as pd
import numpy as np
import requests
from backend.data.data_cache import cache_get, cache_set
from backend.config.settings import settings

TTL = settings.cache_ttl_seconds

# Browser-like session to bypass Yahoo Finance's cloud-IP blocking
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
    return yf.Ticker(symbol, session=_SESSION)


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
    """Direct Yahoo Finance chart API for historical OHLCV data."""
    range_map = {"1y": "1y", "2y": "2y", "6mo": "6mo", "1mo": "1mo", "5d": "5d"}
    yf_range = range_map.get(period, "2y")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = _SESSION.get(url, params={"interval": "1d", "range": yf_range}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Open":   quotes.get("open", []),
        "High":   quotes.get("high", []),
        "Low":    quotes.get("low", []),
        "Close":  quotes.get("close", []),
        "Volume": quotes.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None))
    df.index.name = "Date"
    return df.dropna(subset=["Close"])


def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Returns OHLCV DataFrame with DatetimeIndex. Cached 1h."""
    cached = cache_get("price_history", ticker, period)
    if cached:
        df_cached = pd.DataFrame(cached)
        if "Date" in df_cached.columns:
            df_cached["Date"] = pd.to_datetime(df_cached["Date"])
            df_cached = df_cached.set_index("Date")
        return df_cached

    try:
        df = _yahoo_history_direct(ticker, period)
    except Exception:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False, session=_SESSION)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    cache_set("price_history", ticker, period, value=df.reset_index().to_dict("records"), ttl=TTL)
    return df


def get_current_price(ticker: str) -> float:
    """Returns latest closing price. Cached 5 min."""
    cached = cache_get("current_price", ticker)
    if cached:
        return float(cached)

    # Try direct API first, fall back to yfinance wrapper
    try:
        price = _yahoo_price_direct(ticker)
    except Exception:
        hist = _ticker(ticker).history(period="5d")
        if hist.empty:
            raise ValueError(f"Cannot fetch current price for {ticker}")
        price = float(hist["Close"].iloc[-1])

    cache_set("current_price", ticker, value=price, ttl=300)
    return price


def get_dividend_yield(ticker: str) -> float:
    """Returns trailing dividend yield (annualized). Cached 1h."""
    cached = cache_get("div_yield", ticker)
    if cached is not None:
        return float(cached)

    try:
        info = _ticker(ticker).info
        yield_val = float(info.get("dividendYield") or 0.0)
        if yield_val > 0.15:
            yield_val = yield_val / 100.0
    except Exception:
        yield_val = 0.0
    cache_set("div_yield", ticker, value=yield_val, ttl=TTL)
    return yield_val


def get_risk_free_rate() -> float:
    """Returns 3-month Treasury yield as risk-free rate proxy. Cached 1h."""
    cached = cache_get("risk_free_rate")
    if cached is not None:
        return float(cached)

    try:
        rate = _yahoo_price_direct("^IRX") / 100.0
        cache_set("risk_free_rate", value=rate, ttl=TTL)
        return rate
    except Exception:
        pass

    return 0.045


def get_historical_volatility(ticker: str, window: int = 20) -> float:
    """Annualized historical volatility from log returns."""
    df = get_price_history(ticker, period="1y")
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return float(log_ret.rolling(window).std().iloc[-1] * np.sqrt(252))


def get_returns(ticker: str, period: str = "2y") -> pd.Series:
    """Daily log returns series."""
    df = get_price_history(ticker, period=period)
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()
