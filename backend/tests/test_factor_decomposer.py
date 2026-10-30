"""
Regression tests for the factor/beta estimation path.

Each class below pins one defect that shipped in
`backend/core/risk/factor_decomposer.py` and its Layer 3 consumer. All of them
are failures of honesty rather than of arithmetic: the module answered every
question it was asked, including the ones it had no data to answer, and the
answers flowed straight into position sizing.

  1. The window was taken before the alignment, so two series that ended on
     different dates were compared over different calendar periods, and a
     tz-aware/tz-naive pair collapsed the overlap to nothing.
  2. `return 1.0` stood in for an unmeasurable beta — the specific claim that
     the holding moves one-for-one with the benchmark.
  3. `factor_contributions` reported a hardcoded `sector: 0.2` and
     `market: |beta| x 0.6` as variance shares.
  4. L3 substituted `basis_r2 = 0.5` when the hedge regression failed, and
     0.5 outscored an honestly measured weak hedge.
  5. L3's `_fallback_profile` invented a whole risk profile (15% VaR, 20%
     CVaR, 0.7 tail correlation) and L4 built tradeable recommendations on it.

Everything here is synthetic and offline: `get_returns` is monkeypatched, so
no test depends on a market being open or on Yahoo answering.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.core.risk.factor_decomposer import (
    DEFAULT_WINDOW_DAYS,
    MIN_REGRESSION_OBS,
    FactorDecomposer,
    InsufficientFactorData,
)


# ── Synthetic series helpers ──────────────────────────────────────────────────

def _series(n: int, *, end: date = date(2026, 6, 30), tz: str | None = None,
            seed: int = 0, beta: float = 1.0, base: pd.Series | None = None):
    """
    A daily return series of `n` business days ending on `end`.

    When `base` is given the series is `beta * base + noise`, so a regression
    against `base` has a known slope — that is what lets a test assert on the
    estimate rather than merely on its type.
    """
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    if tz:
        idx = idx.tz_localize(tz)
    rng = np.random.default_rng(seed)
    if base is None:
        vals = rng.normal(0.0, 0.01, n)
    else:
        vals = beta * base.values[-n:] + rng.normal(0.0, 0.0005, n)
    return pd.Series(vals, index=idx)


def _patch_returns(monkeypatch, mapping: dict):
    """Point `market_data.get_returns` at a fixed {ticker: Series} mapping."""
    import backend.data.market_data as md

    def fake_get_returns(ticker: str, period: str = "2y") -> pd.Series:
        if ticker not in mapping:
            raise ValueError(f"no synthetic series for {ticker}")
        return mapping[ticker]

    monkeypatch.setattr(md, "get_returns", fake_get_returns)


@pytest.fixture
def decomposer():
    return FactorDecomposer()


# ── 1. Align before windowing ─────────────────────────────────────────────────

class TestAlignBeforeWindow:
    """
    The old code did `get_returns(a).iloc[-window:]` and
    `get_returns(b).iloc[-window:]` and only then intersected the indexes.
    """

    def test_a_tz_aware_and_a_tz_naive_series_still_regress(
        self, decomposer, monkeypatch
    ):
        """
        Yahoo returns tz-aware stamps on one fetch route and naive ones on the
        other. Truncate-then-intersect produced an empty overlap here, which
        read as "insufficient data" and became beta 1.0. Both legs cover the
        same sessions, so the regression must succeed.
        """
        spy = _series(400, seed=1)
        aapl_naive = _series(400, base=spy, beta=1.3, seed=2)
        aapl = pd.Series(
            aapl_naive.values,
            index=aapl_naive.index.tz_localize("America/New_York"),
        )
        _patch_returns(monkeypatch, {"AAPL": aapl, "SPY": spy})

        beta = decomposer.compute_beta("AAPL", "SPY")
        assert beta == pytest.approx(1.3, abs=0.05)

    def test_legs_ending_on_different_dates_are_compared_over_one_period(
        self, decomposer, monkeypatch
    ):
        """
        The hedge leg is missing its five most recent bars — a stale feed, a
        half-day, a different listing calendar. Truncating each leg to its own
        last 252 rows compared the asset's most recent year against a year of
        the hedge that ended a week earlier, and the slope was fitted across
        mismatched days.

        Aligning first gives 252 genuinely shared sessions, so the known
        slope comes back. The test asserts the estimate, not just that no
        exception was raised: a misaligned fit on this data returns a slope
        near zero, because shifted returns are uncorrelated.
        """
        spy_full = _series(500, seed=3)
        aapl = _series(500, base=spy_full, beta=0.8, seed=4)
        spy_stale = spy_full.iloc[:-5]
        _patch_returns(monkeypatch, {"AAPL": aapl, "SPY": spy_stale})

        out = decomposer.compute_beta_with_error("AAPL", "SPY")
        assert out["beta"] == pytest.approx(0.8, abs=0.05)
        assert out["n_obs"] == DEFAULT_WINDOW_DAYS

    def test_the_window_is_counted_in_shared_sessions(
        self, decomposer, monkeypatch
    ):
        """
        `n_obs` must describe the aligned sample. Before the fix a caller
        asking for 252 days could be answered from far fewer overlapping ones
        with no indication the sample had shrunk.
        """
        spy = _series(300, seed=5)
        aapl = _series(300, base=spy, beta=1.0, seed=6).iloc[-180:]
        _patch_returns(monkeypatch, {"AAPL": aapl, "SPY": spy})

        out = decomposer.compute_beta_with_error("AAPL", "SPY", window_days=252)
        assert out["n_obs"] == 180
        assert out["window_days"] == 252


# ── 2. No placeholder betas ───────────────────────────────────────────────────

class TestUnmeasurableRaises:

    def test_too_few_observations_raises_instead_of_returning_one(
        self, decomposer, monkeypatch
    ):
        """
        `beta = 1.0` is a position-sizing instruction, not a null. The old
        guard returned it below 30 observations and the caller could not tell.
        """
        spy = _series(300, seed=7)
        new_listing = _series(40, base=spy, beta=1.1, seed=8)
        _patch_returns(monkeypatch, {"NEWCO": new_listing, "SPY": spy})

        with pytest.raises(InsufficientFactorData) as exc:
            decomposer.compute_beta("NEWCO", "SPY")
        # The message has to name the shortfall, because it is forwarded to
        # the user as the reason the holding could not be hedged.
        assert "40" in str(exc.value)
        assert str(MIN_REGRESSION_OBS) in str(exc.value)

    def test_no_overlap_at_all_raises(self, decomposer, monkeypatch):
        spy = _series(300, end=date(2026, 6, 30), seed=9)
        delisted = _series(300, end=date(2020, 1, 31), seed=10)
        _patch_returns(monkeypatch, {"OLDCO": delisted, "SPY": spy})

        with pytest.raises((InsufficientFactorData, ValueError)):
            decomposer.compute_beta("OLDCO", "SPY")

    def test_a_flat_benchmark_raises_rather_than_dividing_by_zero(
        self, decomposer, monkeypatch
    ):
        spy_flat = pd.Series(
            np.zeros(300), index=pd.bdate_range(end="2026-06-30", periods=300)
        )
        aapl = _series(300, seed=11)
        _patch_returns(monkeypatch, {"AAPL": aapl, "SPY": spy_flat})

        with pytest.raises(InsufficientFactorData):
            decomposer.compute_beta("AAPL", "SPY")

    @pytest.mark.parametrize(
        "method,args",
        [
            ("compute_beta", ("NEWCO", "SPY")),
            ("compute_beta_with_error", ("NEWCO", "SPY")),
            ("compute_ols_hedge_ratio", ("NEWCO", "SPY")),
            ("compute_ols_hedge_ratio_with_r2", ("NEWCO", "SPY")),
            ("compute_min_variance_hedge_ratio", ("NEWCO", "SPY")),
        ],
    )
    def test_every_estimator_refuses_rather_than_substitutes(
        self, decomposer, monkeypatch, method, args
    ):
        """
        All five share `_aligned`. A later edit that reintroduces a fallback
        on any one of them — each used to have its own `return 1.0` or
        `r2: 0.0` branch — fails here.
        """
        spy = _series(300, seed=12)
        _patch_returns(
            monkeypatch,
            {"NEWCO": _series(50, base=spy, seed=13), "SPY": spy},
        )
        with pytest.raises(InsufficientFactorData):
            getattr(decomposer, method)(*args)

    def test_a_failed_tlt_regression_does_not_become_a_rate_exposure(
        self, decomposer, monkeypatch
    ):
        """
        `compute_beta(ticker, "TLT")` shared the `return 1.0` line, and
        `decompose_factors` negated it, so a measurement that never happened
        was published as `rate_sensitivity = -1.0`: a strong rate exposure
        asserted on no data. It must be None and listed as unavailable.
        """
        spy = _series(300, seed=14)
        aapl = _series(300, base=spy, beta=1.2, seed=15)
        _patch_returns(monkeypatch, {"AAPL": aapl, "SPY": spy})  # no TLT

        out = decomposer.decompose_factors("AAPL", notional=100_000)
        assert out["rate_sensitivity"] is None
        assert "rate_sensitivity" in out["unavailable"]
        assert out["systematic_beta"] == pytest.approx(1.2, abs=0.05)

    def test_one_missing_factor_does_not_lose_the_others(
        self, decomposer, monkeypatch
    ):
        """
        Exposures are attempted independently: a ticker with no overlapping
        UUP history should still get its SPY beta.
        """
        spy = _series(300, seed=16)
        _patch_returns(monkeypatch, {
            "AAPL": _series(300, base=spy, beta=1.1, seed=17),
            "SPY":  spy,
            "TLT":  _series(300, base=spy, beta=-0.3, seed=18),
        })

        out = decomposer.decompose_factors("AAPL", notional=100_000)
        assert out["systematic_beta"] is not None
        assert out["rate_sensitivity"] is not None
        assert out["usd_sensitivity"] is None
        assert out["credit_spread_sensitivity"] is None
        assert set(out["unavailable"]) >= {
            "usd_sensitivity", "credit_spread_sensitivity"
        }


# ── 3. The variance decomposition is measured, not assumed ────────────────────

class TestFactorContributions:

    def _decompose(self, decomposer, monkeypatch, beta):
        spy = _series(400, seed=19)
        _patch_returns(monkeypatch, {
            "AAPL": _series(400, base=spy, beta=beta, seed=20),
            "SPY":  spy,
        })
        return decomposer.decompose_factors("AAPL", notional=250_000)

    def test_no_fabricated_sector_share(self, decomposer, monkeypatch):
        """`sector: round(0.2, 3)` was a literal, labelled a variance share."""
        out = self._decompose(decomposer, monkeypatch, 1.0)
        assert "sector" not in out["factor_contributions"]

    def test_shares_are_r2_and_its_complement_and_sum_to_one(
        self, decomposer, monkeypatch
    ):
        out = self._decompose(decomposer, monkeypatch, 1.0)
        contrib = out["factor_contributions"]
        assert contrib["market"] == pytest.approx(out["beta_r2"], abs=0.002)
        assert sum(contrib.values()) == pytest.approx(1.0, abs=0.01)

    def test_high_beta_does_not_inflate_the_market_share_past_one(
        self, decomposer, monkeypatch
    ):
        """
        `market = |beta| x 0.6` crossed 1.0 at beta 1.67 and made
        `idiosyncratic` negative — a negative share of variance. R² cannot.
        """
        out = self._decompose(decomposer, monkeypatch, 2.5)
        contrib = out["factor_contributions"]
        assert 0.0 <= contrib["market"] <= 1.0
        assert contrib["idiosyncratic"] >= 0.0

    def test_contributions_are_omitted_when_the_regression_failed(
        self, decomposer, monkeypatch
    ):
        """An empty dict, not a decomposition of something unmeasured."""
        _patch_returns(monkeypatch, {"AAPL": _series(300, seed=21)})  # no SPY
        out = decomposer.decompose_factors("AAPL", notional=100_000)
        assert out["factor_contributions"] == {}
        assert out["systematic_beta"] is None
        assert "systematic_beta" in out["unavailable"]

    def test_the_estimate_carries_its_own_error(self, decomposer, monkeypatch):
        """
        R², standard error and sample size travel with the beta so a
        120-point estimate cannot be read as a 1,000-point one.
        """
        out = self._decompose(decomposer, monkeypatch, 1.4)
        assert 0.0 <= out["beta_r2"] <= 1.0
        assert out["beta_stderr"] > 0
        assert out["beta_n_obs"] == DEFAULT_WINDOW_DAYS
