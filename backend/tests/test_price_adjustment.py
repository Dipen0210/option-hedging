"""
Price history must be total-return adjusted, on every fetch path.

The defect this file guards against: `_yahoo_history_direct` — the primary
path — read `indicators.quote[0].close`, which Yahoo split-adjusts but does
not dividend-adjust, while the yfinance fallback immediately below it passed
`auto_adjust=True` and got the dividend-adjusted series. So the same ticker
returned a different price history depending only on whether Yahoo's direct
endpoint happened to answer, and nothing anywhere said so.

The size of the error, measured over 2024-10..2026-10 at the time of the fix:

    ticker   unadjusted 1y    adjusted 1y
    HYG            -4.25%         +1.63%     <- sign flip
    LQD            -7.45%         -2.80%
    TLT           -11.14%         -6.60%

HYG, LQD and TLT are three of the five MACRO_TICKERS in
backend/data/covariance.py, because a high-yield instrument's total return is
mostly coupon. Realized volatility moves by at most 0.37pp, which is why the
error survived every volatility-focused check: the quantity that was wrong was
return, not dispersion.

The fix scales Open/High/Low/Close by Yahoo's own `adjclose / close` factor.
It does *not* substitute `adjclose` for `Close`, and the tests below pin that
distinction: `adjclose` is scaled while High and Low are not, so on a 5-year
NVDA sample it sits outside [Low, High] on 107 of 1254 bars. Substituting it
would have swapped a dividend error for bars that fail the OHLC consistency
check in backend/data/data_quality.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data import data_quality as dq
from backend.data import market_data as md


# ── Fake provider ────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _payload(closes, adjcloses=None, *, include_adjclose=True, volumes=None):
    """
    A chart-API response shaped like Yahoo's.

    High/Low are set a fixed distance around the unadjusted close, so any
    implementation that scales Close without scaling High and Low produces a
    bar whose Close is out of band — which is exactly what the substitution
    bug looked like.
    """
    n = len(closes)
    ts = [1730332800 + 86400 * i for i in range(n)]      # 2024-10-31 onward
    indicators = {
        "quote": [{
            "open":   [c * 0.99 for c in closes],
            "high":   [c * 1.01 for c in closes],
            "low":    [c * 0.98 for c in closes],
            "close":  list(closes),
            "volume": list(volumes) if volumes else [1_000_000] * n,
        }],
    }
    if include_adjclose:
        indicators["adjclose"] = [{"adjclose": list(adjcloses)}]
    return {"chart": {"result": [{"timestamp": ts, "indicators": indicators}]}}


@pytest.fixture
def capture_request(monkeypatch):
    """Install a fake session and expose the params the code sent."""
    sent = {}

    def _install(payload):
        def fake_get(url, params=None, timeout=None):
            sent["url"] = url
            sent["params"] = params or {}
            return _Resp(payload)
        monkeypatch.setattr(md._SESSION, "get", fake_get)
        return sent

    return _install


# ── The request ──────────────────────────────────────────────────────────────

class TestTheRequestAsksForAnAdjustedSeries:

    def test_it_requests_the_adjusted_close(self, capture_request):
        sent = capture_request(_payload([100.0, 101.0], [99.0, 100.0]))
        md._yahoo_history_direct("HYG", "2y")
        assert sent["params"].get("includeAdjustedClose") == "true", (
            "without this parameter Yahoo omits adjclose entirely and the "
            "series silently reverts to a price basis"
        )
        assert "div" in sent["params"].get("events", "")

    def test_it_still_sends_the_interval_and_range(self, capture_request):
        sent = capture_request(_payload([100.0], [100.0]))
        md._yahoo_history_direct("SPY", "5y")
        assert sent["params"]["interval"] == "1d"
        assert sent["params"]["range"] == "5y"


# ── The adjustment ───────────────────────────────────────────────────────────

class TestTheFactorIsAppliedToEveryPriceColumn:

    def test_all_four_prices_are_scaled(self, capture_request):
        # A flat 0.90 factor: every bar's adjclose is 90% of its close.
        closes = [100.0, 110.0, 120.0]
        capture_request(_payload(closes, [c * 0.90 for c in closes]))
        df = md._yahoo_history_direct("HYG", "2y")

        assert df["Close"].tolist() == pytest.approx([90.0, 99.0, 108.0])
        assert df["Open"].tolist() == pytest.approx([c * 0.99 * 0.90 for c in closes])
        assert df["High"].tolist() == pytest.approx([c * 1.01 * 0.90 for c in closes])
        assert df["Low"].tolist() == pytest.approx([c * 0.98 * 0.90 for c in closes])

    def test_volume_is_not_scaled(self, capture_request):
        # The chart API's `close` is already split-adjusted, so the factor
        # carries dividends only — and a dividend does not change share count.
        closes = [100.0, 110.0]
        capture_request(_payload(closes, [c * 0.5 for c in closes],
                                 volumes=[1_234_000, 2_345_000]))
        df = md._yahoo_history_direct("HYG", "2y")
        assert df["Volume"].tolist() == [1_234_000, 2_345_000]

    def test_the_bars_stay_internally_consistent(self, capture_request):
        # The substitution bug: adjclose is scaled, High/Low are not, so
        # dropping adjclose into Close puts Close below Low on every bar
        # carrying a dividend adjustment. 0.90 < the 0.98 Low multiplier, so
        # a substituting implementation fails here and a scaling one passes.
        closes = [100.0, 110.0, 120.0]
        capture_request(_payload(closes, [c * 0.90 for c in closes]))
        df = md._yahoo_history_direct("HYG", "2y")

        assert (df["Close"] >= df["Low"] - 1e-9).all()
        assert (df["Close"] <= df["High"] + 1e-9).all()
        assert (df["Open"] >= df["Low"] - 1e-9).all()
        assert (df["Open"] <= df["High"] + 1e-9).all()

    def test_the_factor_varies_per_bar(self, capture_request):
        # A real factor is a step function rising to 1.0 at the right edge, as
        # the dividends still to be paid run out. A single scalar taken from
        # one bar would get the other two wrong.
        closes = [100.0, 100.0, 100.0]
        capture_request(_payload(closes, [97.0, 98.5, 100.0]))
        df = md._yahoo_history_direct("HYG", "2y")
        assert df["Close"].tolist() == pytest.approx([97.0, 98.5, 100.0])

    def test_a_blank_adjusted_close_is_forward_filled(self, capture_request):
        # Yahoo leaves holes. The factor is a step function of cumulative
        # dividends, so carrying the previous factor forward is right where
        # interpolating a price would not be.
        capture_request(_payload([100.0, 100.0, 100.0], [95.0, None, 100.0]))
        df = md._yahoo_history_direct("HYG", "2y")
        assert len(df) == 3
        assert df["Close"].iloc[0] == pytest.approx(95.0)
        assert df["Close"].iloc[1] == pytest.approx(95.0)   # factor carried
        assert df["Close"].iloc[2] == pytest.approx(100.0)


# ── The financial consequence ────────────────────────────────────────────────

class TestTheAdjustmentChangesTheAnswer:

    def test_a_coupon_heavy_series_changes_sign(self, capture_request):
        """
        The HYG case, reduced to its essentials: a price that drifts down
        while the total return is positive. Before the fix the product read
        the first series and reported a loss on an instrument that made money.
        """
        n = 60
        price = [100.0 - 0.05 * i for i in range(n)]          # -3% on price
        total = [100.0 + 0.05 * i for i in range(n)]          # +3% total return
        capture_request(_payload(price, total))
        df = md._yahoo_history_direct("HYG", "2y")

        adjusted_ret = np.log(df["Close"].iloc[-1] / df["Close"].iloc[0])
        price_ret = np.log(price[-1] / price[0])

        assert price_ret < 0, "fixture precondition: the price basis shows a loss"
        assert adjusted_ret > 0, (
            "the adjusted series must show the gain the holder actually made"
        )


# ── Degradation is disclosed, never silent ───────────────────────────────────

class TestAMissingAdjustmentIsDisclosedNotFaked:

    def test_no_adjclose_returns_unadjusted_bars(self, capture_request):
        closes = [100.0, 110.0]
        capture_request(_payload(closes, include_adjclose=False))
        df = md._yahoo_history_direct("^VIX", "2y")
        assert df["Close"].tolist() == pytest.approx(closes)

    def test_no_adjclose_is_recorded_against_the_ledger(self, capture_request):
        capture_request(_payload([100.0, 110.0], include_adjclose=False))
        with dq.capture() as log:
            md._yahoo_history_direct("^VIX", "2y")
        names = [p.name for p in log.points]
        assert "history_adjustment:^VIX" in names, (
            "a series that is not dividend-adjusted must say so rather than "
            "passing itself off as a total-return series"
        )
        point = next(p for p in log.points if p.name == "history_adjustment:^VIX")
        assert point.source == dq.SOURCE_UNAVAILABLE
        assert point.degraded is True

    def test_an_empty_adjclose_list_is_treated_as_missing(self, capture_request):
        closes = [100.0, 110.0]
        capture_request(_payload(closes, []))
        df = md._yahoo_history_direct("^VIX", "2y")
        assert df["Close"].tolist() == pytest.approx(closes)

    def test_a_zero_factor_is_refused_rather_than_zeroing_the_prices(self, capture_request):
        # adjclose of 0 would scale every price to 0 — a series of free
        # instruments. Keep the unadjusted bars and record the shortfall.
        closes = [100.0, 110.0]
        capture_request(_payload(closes, [0.0, 0.0]))
        with dq.capture() as log:
            df = md._yahoo_history_direct("XYZ", "2y")
        assert df["Close"].tolist() == pytest.approx(closes)
        assert (df["Close"] > 0).all()
        assert any(p.name == "history_adjustment:XYZ" for p in log.points)

    def test_a_negative_factor_is_refused(self, capture_request):
        closes = [100.0, 110.0]
        capture_request(_payload(closes, [-50.0, -55.0]))
        df = md._yahoo_history_direct("XYZ", "2y")
        assert (df["Close"] > 0).all()
        assert df["Close"].tolist() == pytest.approx(closes)

    def test_the_prices_remain_finite_and_positive_in_every_case(self, capture_request):
        for adj in ([95.0, 100.0], [0.0, 0.0], [-1.0, -1.0], None):
            capture_request(_payload([100.0, 110.0], adj,
                                     include_adjclose=adj is not None))
            df = md._yahoo_history_direct("XYZ", "2y")
            for col in ("Open", "High", "Low", "Close"):
                assert np.isfinite(df[col]).all(), f"{col} not finite for adj={adj}"
                assert (df[col] > 0).all(), f"{col} not positive for adj={adj}"


# ── The offline harness must stay faithful to the provider ───────────────────

class TestTheSyntheticHarnessSuppliesAnAdjustedClose:
    """
    A guard on the guard.

    The offline harness stands in for Yahoo in most of this suite. When it
    omitted `adjclose`, every synthetic run recorded a
    `history_adjustment: unavailable` degradation for all 25 fetched symbols —
    a failure mode that cannot happen against the real endpoint, which returns
    the field for equities, ETFs and the dividend-free ^VIX family alike. The
    harness was asserting a defect into existence.
    """

    def test_the_chart_payload_includes_an_adjusted_close(self):
        from backend.tests.fixtures.synthetic_market import FakeYahoo

        market = FakeYahoo()
        resp = market.session_get(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            params={"interval": "1d", "range": "2y"},
        )
        indicators = resp.json()["chart"]["result"][0]["indicators"]

        assert "adjclose" in indicators, (
            "the harness must return the adjusted close Yahoo always returns, "
            "or every offline run is spuriously marked degraded"
        )
        adj = indicators["adjclose"][0]["adjclose"]
        close = indicators["quote"][0]["close"]
        assert len(adj) == len(close)
        assert adj == pytest.approx(close), (
            "the fixture pays no dividends, so the factor must be exactly 1.0 "
            "and no asserted price may shift"
        )

    def test_a_synthetic_fetch_records_no_adjustment_degradation(self, monkeypatch):
        from backend.tests.fixtures import synthetic_market as sm

        sm.install(monkeypatch, sm.ChainConfig())
        with dq.capture() as log:
            df = md.get_price_history("SPY", "2y")
        assert not df.empty
        assert not any(p.name.startswith("history_adjustment:") for p in log.points), (
            "an offline run must not report a missing adjusted close"
        )
