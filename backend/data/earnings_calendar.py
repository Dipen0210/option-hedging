"""
Earnings calendar utilities.

Gap 9 fix: Buying a hedge that expires through an earnings announcement is
dangerous.  IV spikes 2× before earnings then collapses immediately after
(IV crush) — a long put or call can lose 30–50% of its value the day after
the print even if the stock barely moves.

Functions:
  get_next_earnings_date(ticker)       → Optional[str]  ISO date or None
  crosses_earnings(ticker, expiry_str) → Optional[str]  earnings date if crossed
"""
import logging
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

from backend.data.data_cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Cache earnings dates for 24 hours (they don't change intraday)
_EARNINGS_TTL = 86_400


def get_next_earnings_date(ticker: str) -> Optional[str]:
    """
    Return the next (or most recent upcoming) earnings date for ticker as an
    ISO string (e.g. "2026-05-01"), or None if unavailable.

    Data source: yfinance earnings_dates — contains both historical and
    upcoming earnings with an "Earnings Date" column.

    Returns None when:
      - yfinance has no data for the ticker
      - No future earnings date found within the next 18 months
    """
    cache_key = ("next_earnings", ticker)
    cached = cache_get(*cache_key)
    if cached is not None:
        return cached if cached != "__none__" else None

    try:
        t = yf.Ticker(ticker)

        # yfinance.earnings_dates: DatetimeIndex → DataFrame
        ed = t.earnings_dates
        if ed is None or ed.empty:
            cache_set(*cache_key, value="__none__", ttl=_EARNINGS_TTL)
            return None

        today = date.today()
        cutoff = today + timedelta(days=548)   # 18 months forward

        # earnings_dates index is timezone-aware — normalise to date
        future = []
        for ts in ed.index:
            try:
                d = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
                if today <= d <= cutoff:
                    future.append(d)
            except Exception:
                continue

        if not future:
            cache_set(*cache_key, value="__none__", ttl=_EARNINGS_TTL)
            return None

        result = min(future).isoformat()
        cache_set(*cache_key, value=result, ttl=_EARNINGS_TTL)
        return result

    except Exception as e:
        logger.debug("earnings_calendar: failed for %s: %s", ticker, e)
        cache_set(*cache_key, value="__none__", ttl=_EARNINGS_TTL)
        return None


def crosses_earnings(
    ticker: str,
    expiry_str: str,
    window_before_days: int = 3,
) -> Optional[str]:
    """
    Check whether the period [today, expiry_str] contains a known earnings date.

    An expiry "crosses" earnings when it lands within `window_before_days`
    days *before* the earnings date up through the earnings date itself.
    That window is where IV inflates most — buying into it and expiring into
    the crush destroys the hedge value.

    Args:
        ticker:             underlying ticker
        expiry_str:         ISO expiry date (e.g. "2026-05-16")
        window_before_days: how many days before earnings to flag (default 3)

    Returns:
        The earnings date (ISO string) if the expiry crosses it, else None.

    Example:
        Earnings on 2026-05-01, expiry 2026-05-16
        → window = 2026-04-28 … 2026-05-01
        → expiry (2026-05-16) is AFTER the window, so this does NOT cross.
        → But if expiry = 2026-04-30, it lands inside the IV-spike window.
        → And if expiry = 2026-05-02, it lands the day after the crush.

    The most dangerous window for a hedge buyer:
        expiry between (earnings_date - 5 days) and (earnings_date + 1 day)
    """
    try:
        earnings_iso = get_next_earnings_date(ticker)
        if not earnings_iso:
            return None

        today    = date.today()
        expiry   = date.fromisoformat(expiry_str)
        earnings = date.fromisoformat(earnings_iso)

        # Flag: expiry falls in [earnings - window_before, earnings + 1]
        danger_start = earnings - timedelta(days=window_before_days)
        danger_end   = earnings + timedelta(days=1)   # day after = IV crush day

        if danger_start <= expiry <= danger_end:
            return earnings_iso

        return None

    except Exception as e:
        logger.debug("crosses_earnings: error for %s/%s: %s", ticker, expiry_str, e)
        return None
