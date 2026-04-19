"""
Covariance-based cross-hedge asset ranking.

For each user asset, computes rolling 60-day correlation against the
project's hedge universe and returns a ranked list of candidates with:
  - correlation / R² (for basis_risk_r2)
  - hedge_direction: "opposite" (positive corr → buy puts/short)
                     "same"     (negative corr → buy calls/long)
  - hedge_category:  "cross_hedge" | "macro_hedge"

Only tickers already in the project universe are considered.
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

# Minimum absolute correlation to qualify as a useful hedge
MIN_CORR_THRESHOLD = 0.25

# Looser threshold for macro tickers (they have lower unconditional corr)
MACRO_CORR_THRESHOLD = 0.10

# Lookback window in trading days
LOOKBACK_DAYS = 60

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
        correlation     float   (signed, 60-day rolling)
        covariance      float   (signed)
        r2              float   (correlation²)
        hedge_direction str     "opposite" | "same"
        hedge_category  str     "cross_hedge" | "macro_hedge"
    """
    cache_key = ("ranked_cross_hedges", asset_ticker, n_top)
    cached = cache_get(*cache_key)
    if cached:
        return cached

    from backend.data.market_data import get_returns

    try:
        asset_ret = get_returns(asset_ticker).iloc[-LOOKBACK_DAYS:]
    except Exception as e:
        logger.warning("covariance: cannot fetch returns for %s: %s", asset_ticker, e)
        return []

    results: List[Dict] = []

    for ticker in CROSS_HEDGE_UNIVERSE:
        if ticker.upper() == asset_ticker.upper():
            continue
        try:
            hedge_ret = get_returns(ticker).iloc[-LOOKBACK_DAYS:]

            # Align on common trading dates
            aligned = pd.concat(
                [asset_ret.rename("asset"), hedge_ret.rename("hedge")],
                axis=1,
            ).dropna()

            if len(aligned) < 20:
                logger.debug("covariance: insufficient overlap for %s (%d rows)", ticker, len(aligned))
                continue

            corr = float(aligned["asset"].corr(aligned["hedge"]))
            cov  = float(aligned["asset"].cov(aligned["hedge"]))

            if pd.isna(corr):
                continue

            threshold = MACRO_CORR_THRESHOLD if ticker in MACRO_TICKERS else MIN_CORR_THRESHOLD
            if abs(corr) < threshold:
                logger.debug("covariance: %s vs %s corr=%.3f below threshold — skipped", asset_ticker, ticker, corr)
                continue

            results.append({
                "ticker":      ticker,
                "correlation": corr,
                "covariance":  cov,
                "r2":          corr ** 2,
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
