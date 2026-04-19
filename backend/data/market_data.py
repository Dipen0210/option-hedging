import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from backend.data.data_cache import cache_get, cache_set
from backend.config.settings import settings

TTL = settings.cache_ttl_seconds


def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Returns OHLCV DataFrame with DatetimeIndex. Cached 1h."""
    cached = cache_get("price_history", ticker, period)
    if cached:
        df_cached = pd.DataFrame(cached)
        if "Date" in df_cached.columns:
            df_cached["Date"] = pd.to_datetime(df_cached["Date"])
            df_cached = df_cached.set_index("Date")
        return df_cached

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    # yfinance ≥0.2.x returns MultiIndex columns (e.g. ('Close','SPY')) — flatten to simple strings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    cache_set("price_history", ticker, period, value=df.reset_index().to_dict("records"), ttl=TTL)
    return df


def get_current_price(ticker: str) -> float:
    """Returns latest closing price. Cached 5 min."""
    cached = cache_get("current_price", ticker)
    if cached:
        return float(cached)

    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="1d")
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

    info = yf.Ticker(ticker).info
    yield_val = float(info.get("dividendYield") or 0.0)
    # yfinance sometimes returns yield as a percentage integer (e.g. 1.24 meaning 1.24%)
    # rather than a decimal (0.0124). Clamp to realistic range: 0–15% annual yield.
    if yield_val > 0.15:
        yield_val = yield_val / 100.0
    cache_set("div_yield", ticker, value=yield_val, ttl=TTL)
    return yield_val


def get_risk_free_rate() -> float:
    """
    Returns 3-month Treasury yield as risk-free rate proxy.
    Uses ^IRX (13-week T-bill). Cached 1h.
    """
    cached = cache_get("risk_free_rate")
    if cached is not None:
        return float(cached)

    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100.0
            cache_set("risk_free_rate", value=rate, ttl=TTL)
            return rate
    except Exception:
        pass

    # Fallback
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
