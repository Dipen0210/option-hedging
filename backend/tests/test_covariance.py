"""
Regression tests for cross-hedge ranking.

The audit found XLP — a defensive staples ETF — ranked as the best cross hedge
for AAPL at R²=0.10, i.e. the instrument offered to hedge the position left 90%
of its movement unhedged. Three independent causes are pinned here:

1. The admission gates were stated in correlation units while the quantity that
   governs how much of a move a hedge removes is R². A 0.25 correlation floor
   admits R²=0.0625, so a candidate explaining 6% of the variance qualified and
   could rank first for want of anything better.
2. Each series was truncated to its own last N rows *before* being aligned, so
   two series ending on different dates were correlated over different calendar
   windows.
3. The join was a bare `pd.concat`, which does not normalise the index.
   Mismatched index stamps collapse the overlap, and the old 20-row floor then
   let a correlation be reported off a handful of points.

These tests substitute only `market_data.get_returns` — the series source. The
alignment, windowing, thresholds, direction split and ranking are the product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data import covariance as cov


# ── Series construction ───────────────────────────────────────────────────────

def _dates(n: int, end: str = "2026-10-28", tz: str | None = None):
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return idx.tz_localize(tz) if tz else idx


def _correlated(base: pd.Series, rho: float, seed: int) -> pd.Series:
    """A series with (approximately) the requested correlation to `base`."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, base.std(), len(base))
    mixed = rho * base.to_numpy() + np.sqrt(max(1 - rho ** 2, 0.0)) * noise
    return pd.Series(mixed, index=base.index)


@pytest.fixture
def returns(monkeypatch):
    """
    A controllable `get_returns`. Register a series per ticker; the ranker sees
    exactly what is registered and nothing else.
    """
    store: dict[str, pd.Series] = {}

    def fake_get_returns(ticker: str, period: str = "2y") -> pd.Series:
        if ticker not in store:
            raise ValueError(f"no series registered for {ticker}")
        return store[ticker]

    import backend.data.market_data as md
    monkeypatch.setattr(md, "get_returns", fake_get_returns)
    # Correlations are cached for an hour; a cached result would mask the
    # behaviour under test.
    monkeypatch.setattr(cov, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(cov, "cache_set", lambda *a, **k: None)
    return store


def _populate(store, asset_rho: dict[str, float], n: int = 300, seed: int = 7):
    """Register ASSET plus one series per ticker at the requested correlation."""
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    asset = pd.Series(rng.normal(0, 0.015, n), index=idx)
    store["ASSET"] = asset
    for i, (ticker, rho) in enumerate(asset_rho.items()):
        store[ticker] = _correlated(asset, rho, seed=100 + i)
    return asset


# ── 1. R²-based admission ─────────────────────────────────────────────────────

class TestAdmissionGates:
    def test_a_weak_cross_hedge_is_not_offered(self, returns, monkeypatch):
        # rho ~0.32 -> R2 ~0.10. This is the XLP-vs-AAPL case: it cleared the
        # old 0.25 correlation floor and was recommended.
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["XLP"])
        _populate(returns, {"XLP": 0.32})
        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert ranked == [], (
            f"a hedge explaining ~10% of the variance was offered: {ranked}"
        )

    def test_a_strong_cross_hedge_is_offered(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        _populate(returns, {"QQQ": 0.85})
        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert len(ranked) == 1
        assert ranked[0]["ticker"] == "QQQ"
        assert ranked[0]["r2"] >= cov.MIN_CROSS_R2

    def test_the_gate_is_r2_not_correlation(self, returns, monkeypatch):
        # Exactly the trap: 0.40 correlation passes a 0.25 correlation floor
        # but gives R2=0.16, well under a 0.30 R2 floor.
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["XLU"])
        _populate(returns, {"XLU": 0.40})
        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert ranked == [], f"R2={ranked[0]['r2']:.3f} was admitted" if ranked else ""

    def test_macro_tickers_keep_a_lower_but_real_floor(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["GLD"])
        _populate(returns, {"GLD": 0.40})          # R2 ~0.16
        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert len(ranked) == 1, "a macro hedge above the macro floor was dropped"
        assert ranked[0]["hedge_category"] == "macro_hedge"

    def test_even_macro_tickers_have_a_floor(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["GLD"])
        _populate(returns, {"GLD": 0.15})          # R2 ~0.02
        assert cov.get_ranked_cross_hedges("ASSET") == []

    def test_thresholds_are_internally_consistent(self):
        assert cov.MIN_CORR_THRESHOLD ** 2 == pytest.approx(cov.MIN_CROSS_R2)
        assert cov.MACRO_CORR_THRESHOLD ** 2 == pytest.approx(cov.MIN_MACRO_R2)
        assert cov.MIN_MACRO_R2 < cov.MIN_CROSS_R2


# ── 2. Align before truncating ────────────────────────────────────────────────

class TestWindowAlignment:
    def test_series_ending_on_different_dates_are_compared_over_shared_dates(
        self, returns, monkeypatch
    ):
        # The hedge series stops three months early. Truncating each series to
        # its own last 252 rows would compare a window ending in October
        # against one ending in July.
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        asset = _populate(returns, {}, n=400)
        stale = _correlated(asset, 0.85, seed=3).iloc[:-60]
        returns["QQQ"] = stale

        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert len(ranked) == 1
        # Only the overlapping dates may be used, so the sample cannot exceed
        # the shorter series.
        assert ranked[0]["n_obs"] <= len(stale)
        assert ranked[0]["n_obs"] <= cov.LOOKBACK_DAYS

    def test_the_window_is_capped_at_the_lookback(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        _populate(returns, {"QQQ": 0.85}, n=600)
        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert ranked[0]["n_obs"] == cov.LOOKBACK_DAYS

    def test_a_mismatched_index_does_not_silently_shrink_the_sample(
        self, returns, monkeypatch
    ):
        # One leg tz-aware, the other naive. A bare pd.concat matches nothing
        # here; align_series normalises both and the correlation survives.
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        asset = _populate(returns, {})
        hedge = _correlated(asset, 0.85, seed=11)
        hedge.index = hedge.index.tz_localize("UTC")
        returns["QQQ"] = hedge

        ranked = cov.get_ranked_cross_hedges("ASSET")
        assert len(ranked) == 1, (
            "a timezone difference between the two index types destroyed the "
            "overlap instead of being normalised away"
        )
        assert ranked[0]["n_obs"] >= cov.MIN_OBSERVATIONS

    def test_too_few_observations_is_not_a_correlation(self, returns, monkeypatch):
        # 40 aligned points cleared the old floor of 20.
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        asset = _populate(returns, {}, n=40)
        returns["QQQ"] = _correlated(asset, 0.95, seed=5)
        assert cov.get_ranked_cross_hedges("ASSET") == [], (
            "a correlation was reported off fewer observations than it takes "
            "to estimate one"
        )

    def test_no_overlap_at_all_is_handled_not_raised(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        _populate(returns, {}, n=300)
        disjoint = pd.Series(
            np.random.default_rng(1).normal(0, 0.015, 300),
            index=_dates(300, end="2020-01-02"),
        )
        returns["QQQ"] = disjoint
        assert cov.get_ranked_cross_hedges("ASSET") == []


# ── 3. Estimate provenance ────────────────────────────────────────────────────

class TestEstimateProvenance:
    def test_every_candidate_reports_what_the_estimate_rests_on(
        self, returns, monkeypatch
    ):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        _populate(returns, {"QQQ": 0.85})
        c = cov.get_ranked_cross_hedges("ASSET")[0]
        assert c["window_days"] == cov.LOOKBACK_DAYS
        assert c["n_obs"] >= cov.MIN_OBSERVATIONS
        assert c["corr_stderr"] > 0

    def test_a_longer_window_gives_a_tighter_estimate(self):
        # The reason the window was widened: the standard error of a
        # correlation near 0.3 over 60 observations is about 0.12, so ranking
        # candidates on a 60-day correlation is largely ranking noise.
        def se(rho, n):
            return (1 - rho ** 2) / ((n - 1) ** 0.5)

        assert se(0.3, 60) > 2 * se(0.3, 252)
        assert cov.LOOKBACK_DAYS == 252

    def test_r2_is_the_square_of_the_reported_correlation(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        _populate(returns, {"QQQ": 0.85})
        c = cov.get_ranked_cross_hedges("ASSET")[0]
        assert c["r2"] == pytest.approx(c["correlation"] ** 2)


# ── 4. Direction split (unchanged behaviour, pinned) ──────────────────────────

class TestDirection:
    def test_positive_correlation_means_buy_puts(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ"])
        _populate(returns, {"QQQ": 0.85})
        assert cov.get_ranked_cross_hedges("ASSET")[0]["hedge_direction"] == "opposite"

    def test_negative_correlation_means_buy_calls(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["TLT"])
        _populate(returns, {"TLT": -0.60})
        c = cov.get_ranked_cross_hedges("ASSET")[0]
        assert c["correlation"] < 0
        assert c["hedge_direction"] == "same"

    def test_the_asset_is_never_its_own_cross_hedge(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["ASSET", "QQQ"])
        _populate(returns, {"QQQ": 0.85})
        tickers = [c["ticker"] for c in cov.get_ranked_cross_hedges("ASSET")]
        assert "ASSET" not in tickers

    def test_candidates_are_ranked_strongest_first(self, returns, monkeypatch):
        monkeypatch.setattr(cov, "CROSS_HEDGE_UNIVERSE", ["QQQ", "IWM", "DIA"])
        _populate(returns, {"QQQ": 0.90, "IWM": 0.70, "DIA": 0.60})
        ranked = [c for c in cov.get_ranked_cross_hedges("ASSET")
                  if c["correlation"] > 0]
        r2s = [c["r2"] for c in ranked]
        assert r2s == sorted(r2s, reverse=True), ranked


# ── 5. The shared alignment helper ────────────────────────────────────────────

class TestAlignSeries:
    """
    `market_data.align_series` is the project's single cross-asset join, and
    its docstring says it exists so callers "cannot reintroduce the
    timestamp-misalignment defect". It did not handle a timezone-awareness
    mismatch: `normalize()` preserves the tz, pandas refuses to join tz-aware
    to tz-naive, and the resulting TypeError is not the ValueError the
    function documents — so a caller guarding against misalignment did not
    catch it. Yahoo returns tz-aware stamps on some paths and naive ones on
    others, so the mismatch is reachable in production.
    """

    def test_tz_aware_joins_tz_naive(self):
        from backend.data.market_data import align_series

        idx = _dates(10)
        naive = pd.Series(np.arange(10.0), index=idx)
        aware = pd.Series(np.arange(10.0), index=idx.tz_localize("UTC"))
        assert len(align_series(a=naive, b=aware)) == 10

    def test_two_different_timezones_join(self):
        from backend.data.market_data import align_series

        idx = _dates(10)
        utc = pd.Series(np.arange(10.0), index=idx.tz_localize("UTC"))
        ny = pd.Series(np.arange(10.0), index=idx.tz_localize("America/New_York"))
        assert len(align_series(a=utc, b=ny)) == 10

    def test_the_trading_date_is_preserved_not_shifted(self):
        # tz_localize(None) keeps the wall-clock date. Converting to UTC first
        # would roll a late-session stamp onto the following day.
        from backend.data.market_data import align_series

        idx = pd.DatetimeIndex(["2026-10-28"]).tz_localize("America/New_York")
        s = pd.Series([1.0], index=idx)
        out = align_series(only=s)
        assert out.index[0] == pd.Timestamp("2026-10-28")

    def test_a_genuinely_empty_overlap_still_raises_value_error(self):
        from backend.data.market_data import align_series

        a = pd.Series([1.0], index=pd.DatetimeIndex(["2026-10-28"]))
        b = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-02"]))
        with pytest.raises(ValueError, match="No overlapping dates"):
            align_series(a=a, b=b)

    def test_an_empty_series_raises_value_error(self):
        from backend.data.market_data import align_series

        with pytest.raises(ValueError, match="empty"):
            align_series(a=pd.Series(dtype=float))
