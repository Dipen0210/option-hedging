"""
Covariance-based cross-hedge asset ranking.

For each user asset, correlates the project hedge universe against it over a
one-year window and returns a ranked list of candidates with:
  - correlation / R² (for basis_risk_r2)
  - hedge_direction: "opposite" (positive corr → buy puts/short)
                     "same"     (negative corr → buy calls/long)
  - hedge_category:  "cross_hedge" | "macro_hedge"

Only tickers already in the project universe are considered.

Two defects in the previous version produced the kind of result that
discredits the whole feature — XLP, a defensive staples ETF, ranked as the
best cross hedge for AAPL at R²=0.10, meaning 90% of AAPL's movement was left
unhedged by the instrument offered to hedge it:

1. The admission gates were stated as correlations while the quantity that
   decides how much of a move a hedge removes is R². `MIN_CORR_THRESHOLD` of
   0.25 admits R²=0.0625; the macro threshold of 0.10 admits R²=0.01. A
   candidate explaining 6% of the variance is not a hedge, and once admitted
   it could rank first simply because nothing better cleared the bar. The
   gates are now expressed as R², at levels where the instrument actually
   offsets the exposure.

2. Each series was truncated to its own last N rows and only then aligned, so
   two series ending on different dates were compared over different calendar
   windows, and `dropna()` silently shrank the sample. With a floor of 20
   overlapping observations a correlation could be reported off 20 points. The
   alignment now happens first and the window is taken from the aligned frame,
   so both legs always span the same dates.

Window length is a variance/responsiveness trade-off. At 60 observations the
standard error of a correlation near 0.3 is about 0.12, giving a 95% interval
of roughly [0.07, 0.53] — ranking candidates on that is largely ranking noise.
One year (252 trading days) cuts it to about 0.06 while still tracking a
change in correlation regime within a quarter. The window and the sample size
are returned with every candidate so the caller can disclose the basis of the
estimate rather than presenting R² as exact.
"""
import logging
import pandas as pd
from typing import List, Dict
from backend.data.data_cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# ── Hedge universe (ETFs + macro products only — not individual stocks) ────────

CROSS_HEDGE_UNIVERSE: List[str] = [
    # Index ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # Hedge / macro products
    "GLD", "TLT", "HYG", "LQD", "UUP",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE",
]

# Tickers treated as macro/diversification hedges (low avg corr, crisis protection)
MACRO_TICKERS = {"GLD", "TLT", "UUP", "HYG", "LQD"}

# ── Admission gates, expressed as R² ─────────────────────────────────────────
#
# R² is the share of the asset's variance the hedge instrument explains, so it
# is also, to first order, the share of the move a correctly-sized hedge
# removes. Stating the gate in correlation units made a weak hedge look twice
# as good as it was.

# A cross hedge below this leaves most of the exposure in place; offering it as
# a hedge misrepresents what it does.
MIN_CROSS_R2 = 0.30            # |rho| >= 0.55

# Macro hedges are held for crisis diversification rather than variance
# matching, so a lower bar is legitimate — but not an arbitrarily low one.
MIN_MACRO_R2 = 0.10            # |rho| >= 0.32

# Retained for callers that still reference the correlation form.
MIN_CORR_THRESHOLD = MIN_CROSS_R2 ** 0.5
MACRO_CORR_THRESHOLD = MIN_MACRO_R2 ** 0.5

# Lookback window in trading days (approximately one year).
LOOKBACK_DAYS = 252

# Minimum aligned observations before a correlation is trusted at all. Twenty
# points, the old floor, is not evidence of a relationship.
MIN_OBSERVATIONS = 120

# Cache TTL: 1 hour (correlations don't change fast)
CORR_CACHE_TTL = 3600


def get_ranked_cross_hedges(
    asset_ticker: str,
    n_top: int = 3,
) -> List[Dict]:
    """
    Rank all universe tickers by their rolling correlation with asset_ticker.

    Returns a list of dicts (sorted by |correlation| descending), split into:
      - positive-corr candidates (hedge_direction = "opposite" → buy puts)
      - negative-corr candidates (hedge_direction = "same"     → buy calls)

    At most n_top candidates from each side are returned.

    Each dict:
        ticker          str
        correlation     float   (signed, over the last LOOKBACK_DAYS
                                aligned observations — 252, not the 60
                                this line claimed; the window was
                                lengthened and the docstring was not,
                                and a 60-day R² is roughly four times
                                noisier than the one actually reported)
        covariance      float   (signed)
        r2              float   (correlation²)
        hedge_direction str     "opposite" | "same"
        hedge_category  str     "cross_hedge" | "macro_hedge"
    """
    cache_key = ("ranked_cross_hedges", asset_ticker, n_top)
    cached = cache_get(*cache_key)
    if cached:
        return cached

    from backend.data.market_data import align_series, get_returns

    try:
        # Not truncated here. Each series is cut to the window only after it
        # has been aligned with the asset, so the two legs always cover the
        # same trading dates.
        asset_ret = get_returns(asset_ticker)
    except Exception as e:
        logger.warning("covariance: cannot fetch returns for %s: %s", asset_ticker, e)
        return []

    results: List[Dict] = []

    for ticker in CROSS_HEDGE_UNIVERSE:
        if ticker.upper() == asset_ticker.upper():
            continue
        try:
            hedge_ret = get_returns(ticker)

            # Align on common trading dates first, then take the window from
            # the aligned frame. Truncating each series to its own last N rows
            # before aligning compared different calendar periods whenever the
            # two series ended on different dates.
            #
            # `align_series` rather than a bare `pd.concat`: it exists to
            # centralise this join and it normalises both indexes first. A
            # concat across a tz-aware and a tz-naive index, or across
            # midnight-stamped and intraday-stamped dates, matches almost
            # nothing and then `dropna()` hands back a handful of rows that
            # look like a valid sample.
            aligned = align_series(
                asset=asset_ret, hedge=hedge_ret
            ).iloc[-LOOKBACK_DAYS:]

            n_obs = len(aligned)
            if n_obs < MIN_OBSERVATIONS:
                logger.debug(
                    "covariance: %s vs %s — %d aligned observations, below the "
                    "%d needed to estimate a correlation; skipped",
                    asset_ticker, ticker, n_obs, MIN_OBSERVATIONS,
                )
                continue

            corr = float(aligned["asset"].corr(aligned["hedge"]))
            cov  = float(aligned["asset"].cov(aligned["hedge"]))

            if pd.isna(corr):
                continue

            r2 = corr ** 2
            min_r2 = MIN_MACRO_R2 if ticker in MACRO_TICKERS else MIN_CROSS_R2
            if r2 < min_r2:
                logger.debug(
                    "covariance: %s vs %s rho=%.3f R2=%.3f below the %.2f R2 "
                    "floor — skipped",
                    asset_ticker, ticker, corr, r2, min_r2,
                )
                continue

            results.append({
                "ticker":      ticker,
                "correlation": corr,
                "covariance":  cov,
                "r2":          r2,
                # Returned so the caller can disclose what the estimate rests
                # on instead of presenting R2 as an exact quantity.
                "n_obs":       n_obs,
                "window_days": LOOKBACK_DAYS,
                "corr_stderr": round((1 - r2) / ((n_obs - 1) ** 0.5), 4),
            })

        except Exception as e:
            logger.debug("covariance: failed for %s: %s", ticker, e)
            continue

    # Split into positive / negative correlation pools
    positive = sorted(
        [r for r in results if r["correlation"] > 0],
        key=lambda x: x["correlation"],
        reverse=True,
    )
    negative = sorted(
        [r for r in results if r["correlation"] < 0],
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )

    selected: List[Dict] = []

    for r in positive[:n_top]:
        selected.append({
            **r,
            "hedge_direction": "opposite",   # positive corr → buy puts (opposite direction)
            "hedge_category":  "macro_hedge" if r["ticker"] in MACRO_TICKERS else "cross_hedge",
        })

    for r in negative[:n_top]:
        selected.append({
            **r,
            "hedge_direction": "same",       # negative corr → buy calls (same direction, it rises when asset falls)
            "hedge_category":  "macro_hedge" if r["ticker"] in MACRO_TICKERS else "cross_hedge",
        })

    logger.info(
        "covariance: %s → %d positive-corr + %d negative-corr cross-hedge candidates",
        asset_ticker, len(positive[:n_top]), len(negative[:n_top]),
    )

    cache_set(*cache_key, value=selected, ttl=CORR_CACHE_TTL)
    return selected
