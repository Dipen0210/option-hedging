"""
Live-data tests. Opt-in: these call real providers.

Every other test file in this suite is deterministic — synthetic fixtures,
monkeypatched fetches, no network — because a test that depends on Yahoo
answering is a test that fails for reasons having nothing to do with the code.
But a hedging product whose only evidence is mocked fetches has verified its
mocks, not its data path. The defects this file guards against were all
provider-shaped and invisible to a mock:

  * the tz-aware/tz-naive index mismatch introduced when a second Yahoo fetch
    route was added, which collapsed every cross-asset overlap to nothing;
  * each series being windowed to its own last N rows before the join, so two
    assets were compared over different calendar periods;
  * `impliedVolatility: 0.00001` on stale contracts;
  * option bid/ask being published only between 09:30 and 16:00 ET, so an
    overnight run is entirely model-priced.

Run them with:

    OPTIONQ_LIVE_TESTS=1 pytest backend/tests/test_live_market_data.py -v

They are skipped otherwise, and excluded from a default run by `-m "not live"`.
Several assertions below are market-hours dependent and say so; they soften
rather than fail outside the session, because a closed market is not a bug.
"""
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("OPTIONQ_LIVE_TESTS"),
        reason="live provider tests — set OPTIONQ_LIVE_TESTS=1 to run",
    ),
]

# Assets the pipeline actually depends on: the benchmark, the vol complex, and
# the two factor proxies whose alignment with the benchmark the risk layer
# regresses against.
CORE_TICKERS = ["SPY", "TLT", "HYG", "IWM"]


def _is_us_market_hours() -> bool:
    """
    09:30–16:00 America/New_York weekday check, used only to decide whether a
    missing quote is an outage or a closed market. Deliberately not
    holiday-aware: it gates a softened assertion, never a hard one.

    This used to subtract a flat five hours from UTC and call the result ET,
    which is EST and therefore an hour behind during the ~8 months of EDT.
    The window it actually enforced was 10:30–17:00 EDT, so it skipped the
    two-sided-quote assertion for the first hour of every summer session —
    the hour this repair most needed it — and, worse, kept asserting for an
    hour after the close, where an absent quote is not a defect.
    """
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


# ── Price history ─────────────────────────────────────────────────────────────

class TestPriceHistory:

    @pytest.mark.parametrize("ticker", CORE_TICKERS)
    def test_history_is_non_empty_and_recent(self, ticker):
        from backend.data.market_data import get_price_history

        df = get_price_history(ticker, period="1y")
        assert not df.empty, f"{ticker}: no price history returned"
        assert "Close" in df.columns
        assert len(df) >= 200, f"{ticker}: only {len(df)} rows in a 1y window"

        last = df.index[-1]
        age = (date.today() - last.date()).days
        # Four calendar days covers a long weekend plus a holiday.
        assert age <= 5, f"{ticker}: last bar is {age} days old ({last.date()})"

    @pytest.mark.parametrize("ticker", CORE_TICKERS)
    def test_prices_are_positive_and_finite(self, ticker):
        import numpy as np
        from backend.data.market_data import get_price_history

        close = get_price_history(ticker, period="6mo")["Close"]
        assert (close > 0).all()
        assert np.isfinite(close).all()

    def test_a_current_price_is_available(self):
        from backend.data.market_data import get_current_price

        assert get_current_price("SPY") > 0

    @pytest.mark.parametrize("ticker", ["HYG", "TLT", "LQD", "SPY", "AAPL"])
    def test_both_fetch_paths_return_the_same_adjusted_series(self, ticker):
        """
        The direct chart API and the yfinance fallback must agree.

        They did not. The direct path read the unadjusted `quote.close` while
        the fallback passed `auto_adjust=True`, so which history a caller got
        depended on whether Yahoo's direct endpoint answered. On the coupon-
        heavy instruments in covariance.MACRO_TICKERS the gap reached 5.9
        percentage points of annual return and flipped HYG's sign. A mock
        cannot catch this: the two paths were each self-consistent, and only
        the real provider shows that they disagree with each other.
        """
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from backend.data.market_data import _yahoo_history_direct

        direct = _yahoo_history_direct(ticker, "2y")["Close"]

        fallback = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if isinstance(fallback.columns, pd.MultiIndex):
            fallback.columns = [c[0] for c in fallback.columns]
        fallback = fallback["Close"]
        fallback.index = pd.to_datetime(fallback.index).normalize()

        shared = direct.index.intersection(fallback.index)
        assert len(shared) >= 200, (
            f"{ticker}: only {len(shared)} shared bars between the two paths"
        )

        rel = ((direct.loc[shared] - fallback.loc[shared]).abs()
               / fallback.loc[shared])
        assert rel.max() < 0.005, (
            f"{ticker}: the two fetch paths disagree by up to {rel.max():.2%} "
            f"on price level — one of them is not dividend-adjusted"
        )

        d_ret = np.log(direct.loc[shared]).diff().dropna()
        f_ret = np.log(fallback.loc[shared]).diff().dropna()
        k = d_ret.index.intersection(f_ret.index)
        rho = float(np.corrcoef(d_ret.loc[k], f_ret.loc[k])[0, 1])
        assert rho > 0.999, f"{ticker}: return series correlate at only {rho:.4f}"

    @pytest.mark.parametrize("ticker", ["HYG", "TLT", "LQD", "XOM", "SPY"])
    def test_the_adjusted_bars_are_internally_consistent(self, ticker):
        """
        Close must sit inside [Low, High] after adjustment.

        Yahoo's `adjclose` is scaled while its `high` and `low` are not, so
        substituting it for Close puts Close out of band on every bar carrying
        a dividend adjustment — 107 of 1254 on a 5-year NVDA sample. The fix
        scales all four prices by the same factor instead, and this is what
        tells the two implementations apart on live data.
        """
        from backend.data.market_data import _yahoo_history_direct

        df = _yahoo_history_direct(ticker, "2y")
        tol = 1e-6
        below = df["Close"] < df["Low"] - tol
        above = df["Close"] > df["High"] + tol
        assert not below.any(), (
            f"{ticker}: Close below Low on {int(below.sum())} bars "
            f"(first {df.index[below][0].date() if below.any() else '-'})"
        )
        assert not above.any(), (
            f"{ticker}: Close above High on {int(above.sum())} bars"
        )


# ── Cross-asset alignment ─────────────────────────────────────────────────────

class TestAlignment:
    """
    The Phase 1 defect, checked against the real feed rather than a fixture:
    the join must survive whatever index types the two fetch routes produce.
    """

    def test_the_whole_core_set_aligns_with_enough_history_to_regress(self):
        from backend.data.covariance import MIN_OBSERVATIONS
        from backend.data.market_data import align_series, get_returns

        aligned = align_series(**{t: get_returns(t) for t in CORE_TICKERS})
        assert len(aligned) >= MIN_OBSERVATIONS, (
            f"only {len(aligned)} overlapping sessions across {CORE_TICKERS}; "
            f"the risk layer needs {MIN_OBSERVATIONS}"
        )
        assert aligned.index.is_monotonic_increasing
        assert aligned.index.tz is None, "aligned index must be tz-naive"
        assert not aligned.isna().any().any()

    def test_the_vol_complex_aligns_with_the_benchmark(self):
        """
        SPY / ^VIX / ^VIX3M come from different Yahoo endpoints, which is
        exactly where the tz mismatch appeared.
        """
        from backend.data.market_data import align_series, get_returns

        aligned = align_series(
            spy=get_returns("SPY"),
            vix=get_returns("^VIX"),
            vix3m=get_returns("^VIX3M"),
        )
        assert len(aligned) >= 120

    def test_spy_and_vix_returns_are_negatively_correlated(self):
        """
        A sanity check on the join itself, not on the market: if the two
        series were misaligned by even a day this correlation collapses
        toward zero. SPY/VIX daily returns run around -0.75.
        """
        from backend.data.market_data import align_series, get_returns

        aligned = align_series(spy=get_returns("SPY"), vix=get_returns("^VIX"))
        rho = aligned["spy"].corr(aligned["vix"])
        assert rho < -0.5, f"SPY/VIX correlation {rho:.3f} suggests misalignment"

    def test_a_real_beta_is_measurable_and_sane(self):
        from backend.core.risk.factor_decomposer import FactorDecomposer

        out = FactorDecomposer().compute_beta_with_error("IWM", "SPY")
        # IWM is a broad US equity index; its SPY beta is near 1 by
        # construction. A value outside this range means the regression is
        # being fed something other than what it thinks.
        assert 0.7 < out["beta"] < 1.6, out
        assert out["r2"] > 0.5, out
        assert out["n_obs"] >= 120


# ── VIX ───────────────────────────────────────────────────────────────────────

class TestVix:

    def test_vix_level_is_in_a_possible_range(self):
        from backend.data.vix_data import get_vix

        vix = get_vix()
        assert vix["available"], vix
        assert 5 < vix["level"] < 150, vix

    def test_the_term_structure_is_classified(self):
        """
        ^VIX and ^VIX3M come from separate quotes; a missing one leaves the
        term structure None while the regime layer carries on, so the pair
        is checked together rather than one at a time.
        """
        from backend.data.vix_data import get_vix

        vix = get_vix()
        assert vix["vix3m"] is not None, vix
        assert vix["term_structure"] in {"contango", "backwardation", "flat"}, vix


# ── Option chain ──────────────────────────────────────────────────────────────

class TestOptionChain:

    def test_expiries_are_real_future_dates(self):
        from backend.data.options_chain import get_available_expiries

        expiries = get_available_expiries("SPY")
        assert expiries, "SPY has no listed expiries"
        parsed = [date.fromisoformat(e) for e in expiries]
        assert all(e >= date.today() for e in parsed)
        assert parsed == sorted(parsed)

    def test_strikes_are_listed_not_generated(self):
        """
        Synthetic strike generation was removed in Phase 1. Real SPY strikes
        fall on exchange increments; a model-generated round number derived
        from spot x moneyness generally does not.
        """
        from backend.data.options_chain import (
            get_listed_strikes,
            select_expiry_nearest,
        )

        expiry = select_expiry_nearest("SPY", target_days=90)
        assert expiry is not None
        strikes = get_listed_strikes("SPY", expiry, "put")
        assert len(strikes) > 20, f"only {len(strikes)} strikes listed"
        assert strikes == sorted(strikes)
        # SPY lists $1 and $5 increments; every strike is a whole or half dollar.
        assert all(abs(k * 2 - round(k * 2)) < 1e-6 for k in strikes)

    def test_a_quote_carries_an_occ_symbol_for_the_strike_it_claims(self):
        from backend.data.market_data import get_current_price
        from backend.data.options_chain import (
            get_contract_quote,
            select_expiry_nearest,
            select_listed_strike,
        )

        spot = get_current_price("SPY")
        expiry = select_expiry_nearest("SPY", target_days=90)
        strike = select_listed_strike("SPY", expiry, spot * 0.95, "put")
        assert strike is not None

        quote = get_contract_quote("SPY", expiry, strike, "put")
        assert quote is not None
        symbol = quote.get("contract_symbol")
        assert symbol, "no OCC symbol on the quote"
        # The OCC symbol encodes the strike in thousandths; it must match the
        # strike the recommendation would be built on.
        assert symbol.endswith(f"{int(round(strike * 1000)):08d}")
        assert "SPY" in symbol

    def test_implied_vol_on_a_real_chain_is_plausible(self):
        """
        The stale-quote artifact (0.00001) in its natural habitat: ATM SPY IV
        is the single most liquid reading there is, so anything outside a
        wide band here means the band is being fed junk.
        """
        from backend.data.options_chain import get_atm_iv

        iv = get_atm_iv("SPY", target_days=90)
        if iv is None:
            pytest.skip("no ATM IV published for SPY right now")
        assert 0.05 < iv < 1.0, f"implausible SPY ATM IV {iv}"

    def test_the_skew_is_present_rather_than_flat(self):
        """
        Equity index puts are priced with a downside skew: a 90%-moneyness
        put carries a higher IV than an ATM one. A flat reading across
        strikes is the signature of an ATM value being substituted for the
        contract's own, which the provenance ladder calls `atm_iv` and the UI
        is required to disclose.
        """
        from backend.data.market_data import get_current_price
        from backend.data.options_chain import (
            get_strike_iv,
            select_expiry_nearest,
            select_listed_strike,
        )

        spot = get_current_price("SPY")
        expiry = select_expiry_nearest("SPY", target_days=90)
        atm_k = select_listed_strike("SPY", expiry, spot, "put")
        otm_k = select_listed_strike("SPY", expiry, spot * 0.90, "put")

        atm_iv = get_strike_iv("SPY", expiry, atm_k, "put")
        otm_iv = get_strike_iv("SPY", expiry, otm_k, "put")
        if atm_iv is None or otm_iv is None:
            pytest.skip("chain did not publish IV for both strikes")
        assert otm_iv > atm_iv, (
            f"no downside skew: {otm_k} IV {otm_iv:.4f} <= {atm_k} IV {atm_iv:.4f}"
        )

    def test_two_sided_quotes_exist_during_market_hours(self):
        """
        What makes a recommendation executable rather than theoretical. Yahoo
        clears bid/ask outside 09:30–16:00 ET, so this is an assertion during
        the session and an explicit skip outside it — the absence of a quote
        at 2am is not a defect, and pretending the test passed would hide
        that the overnight run is entirely model-priced.
        """
        from backend.data.market_data import get_current_price
        from backend.data.options_chain import (
            get_contract_quote,
            select_expiry_nearest,
            select_listed_strike,
        )

        spot = get_current_price("SPY")
        expiry = select_expiry_nearest("SPY", target_days=90)
        strike = select_listed_strike("SPY", expiry, spot * 0.95, "put")
        quote = get_contract_quote("SPY", expiry, strike, "put")
        assert quote is not None

        bid, ask = quote.get("bid"), quote.get("ask")
        if not _is_us_market_hours():
            pytest.skip(
                f"outside US market hours — bid={bid} ask={ask}; every premium "
                f"in a run now is model-priced, which the response must disclose"
            )
        assert bid is not None and ask is not None
        assert 0 < bid < ask, f"bid {bid} / ask {ask}"
        assert (ask - bid) / ((ask + bid) / 2) < 0.25
