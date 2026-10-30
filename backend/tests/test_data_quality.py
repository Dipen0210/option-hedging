"""
Regression tests for the data-provenance ledger and portfolio valuation.

Each test here pins a specific defect found in the readiness audit. They are
deterministic — no live market data — so they can run in CI. Live-data checks
live in test_live_market_data.py and are opt-in.

Covered defects:
  P0-7  total_notional was client-supplied cost basis, never reconciled against
        live prices, mis-scaling every VaR / stress loss / contract count.
  P1-x  Silent fallbacks: substituted or missing inputs left no trace, so a
        degraded answer was indistinguishable from a fully live one.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pytest

import pandas as pd

from backend.data import data_quality as dq
from backend.data import options_chain
from backend.models.portfolio_models import PortfolioInput, StockPosition, OptionsPosition


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _stock(ticker="AAPL", shares=200, price=210.0):
    return StockPosition(
        ticker=ticker, shares=shares, purchase_price=price,
        purchase_date=date(2024, 1, 15),
    )


def _portfolio(holdings=None, **kwargs):
    return PortfolioInput(holdings=holdings or [_stock()], **kwargs)


def _degraded_names(log):
    """`degraded_fields` carries detail dicts; tests assert on the field names."""
    return [f["name"] for f in log.summary()["degraded_fields"]]


# ── Ledger semantics ──────────────────────────────────────────────────────────

class TestDataQualityLedger:

    def test_record_outside_capture_is_a_noop(self):
        """A library call with no active request must not raise or leak state."""
        dq.record("spot:AAPL", dq.SOURCE_LIVE)
        assert dq.current_log() is None

    def test_live_only_run_is_not_degraded(self):
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
            dq.record("spot:SPY", dq.SOURCE_LIVE)
        s = log.summary()
        assert s["is_degraded"] is False
        assert s["degraded_fields"] == []
        assert s["source_counts"] == {"live": 2}

    def test_fallback_marks_run_degraded(self):
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
            dq.record("garch:SPY", dq.SOURCE_FALLBACK, detail="realized vol substituted")
        s = log.summary()
        assert s["is_degraded"] is True
        assert _degraded_names(log) == ["garch:SPY"]
        assert any("garch" in w and "SPY" in w for w in s["warnings"])

    def test_unavailable_marks_run_degraded(self):
        with dq.capture() as log:
            dq.record("vix", dq.SOURCE_UNAVAILABLE, detail="provider outage")
        assert log.summary()["is_degraded"] is True
        assert log.has_unavailable("vix")

    def test_fresh_cache_is_not_degraded_but_stale_cache_is(self):
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_CACHE, age_seconds=60)
            dq.record("spot:SPY", dq.SOURCE_CACHE,
                      age_seconds=dq.STALE_AFTER_SECONDS + 1)
        fields = _degraded_names(log)
        assert "spot:AAPL" not in fields
        assert "spot:SPY" in fields

    def test_worst_observation_wins(self):
        """
        One later live fetch of a field must not erase an earlier degraded read
        of it — otherwise a partially stale answer reports itself as clean.
        """
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_FALLBACK, detail="stale")
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
        assert log.summary()["is_degraded"] is True
        assert _degraded_names(log) == ["spot:AAPL"]

    def test_nested_capture_reuse_joins_outer_ledger(self):
        """
        reuse=True must not shadow the caller's ledger, or provenance recorded
        before the inner entry point would be silently dropped.
        """
        with dq.capture() as outer:
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
            with dq.capture(reuse=True) as inner:
                assert inner is outer
                dq.record("regime", dq.SOURCE_FALLBACK, detail="GMM unavailable")
            assert _degraded_names(outer) == ["regime"]
            assert "live" in outer.summary()["source_counts"]

    def test_nested_capture_without_reuse_isolates(self):
        with dq.capture() as outer:
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
            with dq.capture() as inner:
                assert inner is not outer
                dq.record("vix", dq.SOURCE_UNAVAILABLE)
            # inner ledger discarded; outer must be untouched
            assert outer.summary()["is_degraded"] is False
        assert inner.summary()["is_degraded"] is True

    def test_ledger_resets_after_capture_exits(self):
        with dq.capture():
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
        assert dq.current_log() is None


# ── Portfolio input validation ────────────────────────────────────────────────

class TestPortfolioInputValidation:

    def test_total_notional_is_optional(self):
        """The server computes market value; the client need not supply one."""
        p = _portfolio()
        assert p.total_notional is None

    def test_empty_portfolio_rejected(self):
        with pytest.raises(ValueError, match="at least one position"):
            PortfolioInput(holdings=[])

    def test_duplicate_stock_ticker_rejected(self):
        """Two entries for one underlying would be hedged twice on one budget."""
        with pytest.raises(ValueError, match="duplicate position"):
            _portfolio([_stock("AAPL", 100), _stock("AAPL", 200)])

    def test_distinct_tickers_accepted(self):
        p = _portfolio([_stock("AAPL"), _stock("SPY", 150, 500.0)])
        assert len(p.holdings) == 2

    def test_too_many_holdings_rejected(self):
        from backend.config.instrument_universe import supported_tickers_list
        universe = supported_tickers_list()
        n = PortfolioInput.MAX_HOLDINGS + 1
        assert len(universe) >= n, "universe too small to exercise this bound"
        with pytest.raises(ValueError, match="maximum supported"):
            _portfolio([_stock(t, 10, 100.0) for t in universe[:n]])

    def test_expired_option_rejected(self):
        with pytest.raises(ValueError, match="in the past"):
            OptionsPosition(
                ticker="SPY", option_type="put", strike=500.0,
                expiry=date.today() - timedelta(days=1), contracts=1,
            )

    def test_cost_basis_property_sums_what_user_paid(self):
        p = _portfolio([
            _stock("AAPL", 200, 210.0),      # 42,000
            _stock("SPY", 150, 500.0),       # 75,000
        ])
        assert p.cost_basis == pytest.approx(117_000.0)

    def test_cost_basis_includes_option_premium_with_multiplier(self):
        p = _portfolio([
            OptionsPosition(
                ticker="SPY", option_type="put", strike=500.0,
                expiry=date.today() + timedelta(days=60),
                contracts=10, premium_paid=4.25,
            ),
        ])
        # 10 contracts x 100 shares x $4.25 premium
        assert p.cost_basis == pytest.approx(4_250.0)


# ── Valuation ─────────────────────────────────────────────────────────────────

class TestPortfolioValuation:
    """
    Spot prices are injected, so these assert the *arithmetic and reconciliation
    logic* without depending on the market being open.
    """

    def _value(self, portfolio, spots):
        from backend.data.portfolio_valuation import value_portfolio
        return value_portfolio(portfolio, spot_prices=spots)

    def test_market_value_uses_current_prices_not_cost_basis(self):
        """P0-7: the core defect. Exposure must be priced at spot."""
        p = _portfolio([_stock("AAPL", 200, 210.0)], total_notional=42_000.0)
        v = self._value(p, {"AAPL": 300.0})
        assert v.market_value == pytest.approx(60_000.0)   # 200 x 300
        assert v.cost_basis == pytest.approx(42_000.0)     # 200 x 210
        assert v.market_value != v.cost_basis

    def test_unrealized_pnl_is_market_value_minus_cost_basis(self):
        p = _portfolio([_stock("AAPL", 200, 210.0)])
        v = self._value(p, {"AAPL": 300.0})
        assert v.unrealized_pnl == pytest.approx(18_000.0)
        assert v.unrealized_pnl_pct == pytest.approx(60_000 / 42_000 - 1)

    def test_weights_are_computed_from_market_value_and_sum_to_one(self):
        p = _portfolio([_stock("AAPL", 100, 100.0), _stock("SPY", 100, 100.0)])
        v = self._value(p, {"AAPL": 300.0, "SPY": 100.0})
        by = {h.ticker: h.weight for h in v.holdings}
        assert by["AAPL"] == pytest.approx(0.75)
        assert by["SPY"] == pytest.approx(0.25)
        assert sum(by.values()) == pytest.approx(1.0)

    def test_declared_notional_divergence_is_reported_not_silently_accepted(self):
        p = _portfolio([_stock("AAPL", 200, 210.0)], total_notional=42_000.0)
        v = self._value(p, {"AAPL": 300.0})
        # client said 42k, real exposure is 60k
        assert v.declared_notional == pytest.approx(42_000.0)
        assert v.declared_divergence_pct == pytest.approx(42_000 / 60_000 - 1)
        assert len(v.warnings) == 1
        assert "market value" in v.warnings[0]

    def test_no_warning_when_declared_value_agrees(self):
        p = _portfolio([_stock("AAPL", 200, 210.0)], total_notional=60_000.0)
        v = self._value(p, {"AAPL": 300.0})
        assert v.warnings == []

    def test_missing_declared_notional_produces_no_divergence_warning(self):
        p = _portfolio([_stock("AAPL", 200, 210.0)])
        v = self._value(p, {"AAPL": 300.0})
        assert v.declared_divergence_pct is None
        assert v.warnings == []

    def test_option_notional_applies_contract_multiplier(self):
        p = _portfolio([
            OptionsPosition(
                ticker="SPY", option_type="put", strike=500.0,
                expiry=date.today() + timedelta(days=60),
                contracts=10, premium_paid=4.25,
            ),
        ])
        v = self._value(p, {"SPY": 600.0})
        # 10 contracts x 100 multiplier x $600 underlying
        assert v.market_value == pytest.approx(600_000.0)
        assert v.cost_basis == pytest.approx(4_250.0)

    def test_unpriced_holding_is_excluded_and_disclosed(self):
        """
        A holding we cannot price must not be valued at purchase price — that
        would silently substitute cost basis for market value.
        """
        p = _portfolio([_stock("AAPL", 200, 210.0), _stock("SPY", 150, 500.0)])
        import backend.data.market_data as md

        def _boom(ticker):
            raise RuntimeError("provider outage")

        orig = md.get_current_price
        md.get_current_price = _boom
        try:
            with dq.capture() as log:
                v = self._value(p, {"AAPL": 300.0})   # SPY must be fetched -> fails
        finally:
            md.get_current_price = orig

        assert v.market_value == pytest.approx(60_000.0)   # AAPL only
        assert [h.ticker for h in v.holdings] == ["AAPL"]
        assert any("SPY" in w for w in v.warnings)
        assert log.summary()["is_degraded"] is True

    def test_fully_unpriced_portfolio_raises_rather_than_returning_zero(self):
        p = _portfolio([_stock("AAPL", 200, 210.0)])
        import backend.data.market_data as md

        orig = md.get_current_price
        md.get_current_price = lambda t: (_ for _ in ()).throw(RuntimeError("outage"))
        try:
            with pytest.raises(ValueError, match="cannot be valued"):
                self._value(p, {})
        finally:
            md.get_current_price = orig


# ── Provider quote outage vs genuine illiquidity ─────────────────────────────

class TestChainQuoteHealth:
    """
    Found at 00:38 ET on 2026-10-28: Yahoo returned complete SPY strike ladders
    with bid=ask=openInterest=0 for every expiry — 110 strikes on the 1-DTE,
    82 on the 2027-02-19 third-Friday LEAP, zero open interest across both.
    SPY LEAPs carry six-figure open interest in reality, so this is a provider
    snapshot that omits quote and OI fields outside 09:30-16:00 ET, not a
    market where nobody wants SPY protection.

    The pipeline read it as illiquidity, fell through to theoretical pricing,
    and presented the result as a recommendation. These pin the distinction.
    """

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(
            rows,
            columns=["strike", "bid", "ask", "lastPrice",
                     "impliedVolatility", "openInterest", "volume",
                     "contractSymbol"],
        )

    def test_chain_with_no_quotes_and_no_oi_is_reported_unavailable(self):
        df = self._frame([
            [770.0, 0.0, 0.0, 18.4, 0.0019, 0, 41, "SPY270129P00770000"],
            [772.0, 0.0, 0.0, 19.1, 0.0002, 0,  2, "SPY270129P00772000"],
            [775.0, 0.0, 0.0, 20.2, 0.0000, 0, 15, "SPY270129P00775000"],
        ])
        with dq.capture() as log:
            options_chain._assess_quote_health("SPY", "2027-01-29", "puts", df)

        degraded = {p.name: p for p in log.degradations}
        assert "quotes:SPY" in degraded
        assert degraded["quotes:SPY"].source == dq.SOURCE_UNAVAILABLE
        assert "market hours" in degraded["quotes:SPY"].detail

    def test_genuinely_thin_chain_with_real_oi_is_not_an_outage(self):
        """
        One two-sided quote is still a market. Low open interest is a liquidity
        fact the scorer should weigh, not a data outage to disclose.
        """
        df = self._frame([
            [81.0, 2.39, 2.45, 2.40, 0.1355, 10_564, 120, "TLT270219P00081000"],
            [77.0, 0.00, 0.00, 0.95, 0.1400,      3,   0, "TLT270219P00077000"],
        ])
        with dq.capture() as log:
            options_chain._assess_quote_health("TLT", "2027-02-19", "puts", df)
        assert not any(p.name == "quotes:TLT" for p in log.degradations)

    def test_oi_without_quotes_is_not_an_outage(self):
        """
        Open interest present but no live bid/ask is the normal post-close
        state once OI has been published — degraded, but the OI is real, so it
        is not the all-fields-missing snapshot this check is for.
        """
        df = self._frame([
            [770.0, 0.0, 0.0, 18.4, 0.19, 120_400, 9_100, "SPY270115P00770000"],
        ])
        with dq.capture() as log:
            options_chain._assess_quote_health("SPY", "2027-01-15", "puts", df)
        assert not any(p.name == "quotes:SPY" for p in log.degradations)

    def test_empty_frame_records_nothing(self):
        with dq.capture() as log:
            options_chain._assess_quote_health("X", "2027-01-15", "puts",
                                               self._frame([]))
        assert log.degradations == []


# ── Warning grouping ─────────────────────────────────────────────────────────

class TestWarningsAreGroupedByCondition:
    """
    A line per degraded data point produced 77 warnings in a real out-of-hours
    run, burying the two that a user had to act on. Grouping must compress the
    presentation without dropping a condition, a detail or a subject.
    """

    def test_one_condition_affecting_several_tickers_is_one_warning(self):
        with dq.capture() as log:
            for t in ("AAPL", "GLD", "IWM"):
                dq.record(f"atm_iv:{t}", dq.SOURCE_UNAVAILABLE,
                          detail="no quoted strike within 10% of spot")
        w = log.summary()["warnings"]
        assert len(w) == 1, w
        assert all(t in w[0] for t in ("AAPL", "GLD", "IWM"))
        assert "unavailable" in w[0]
        assert "no quoted strike within 10% of spot" in w[0]

    def test_different_details_are_not_merged(self):
        with dq.capture() as log:
            dq.record("atm_iv:AAPL", dq.SOURCE_UNAVAILABLE, detail="no quoted strike")
            dq.record("atm_iv:GLD", dq.SOURCE_UNAVAILABLE, detail="chain empty")
        w = log.summary()["warnings"]
        assert len(w) == 2, w
        assert {"no quoted strike", "chain empty"} == {
            line.split("— ")[1] for line in w
        }

    def test_different_sources_are_not_merged(self):
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_UNAVAILABLE)
            dq.record("spot:GLD", dq.SOURCE_FALLBACK, detail="last close")
        w = log.summary()["warnings"]
        assert len(w) == 2, w

    def test_a_cached_group_reports_the_age_of_its_oldest_member(self):
        # Reporting the youngest would make the group look fresher than its
        # worst value actually is.
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_CACHE, age_seconds=1200)   # 20 min
            dq.record("spot:GLD", dq.SOURCE_CACHE, age_seconds=5400)    # 90 min
        w = log.summary()["warnings"]
        assert len(w) == 1, w
        assert "90 min" in w[0], w[0]
        assert "20 min" not in w[0]

    def test_every_degraded_point_is_still_individually_recorded(self):
        with dq.capture() as log:
            for t in ("AAPL", "GLD", "IWM", "TLT"):
                dq.record(f"atm_iv:{t}", dq.SOURCE_UNAVAILABLE, detail="same reason")
        s = log.summary()
        assert len(s["warnings"]) == 1
        assert len(s["degraded_fields"]) == 4
        assert {p["name"] for p in s["degraded_fields"]} == {
            "atm_iv:AAPL", "atm_iv:GLD", "atm_iv:IWM", "atm_iv:TLT"
        }

    def test_a_long_subject_list_is_truncated_but_counted(self):
        tickers = [f"T{i:02d}" for i in range(25)]
        with dq.capture() as log:
            for t in tickers:
                dq.record(f"spot:{t}", dq.SOURCE_UNAVAILABLE, detail="provider outage")
        w = log.summary()["warnings"]
        assert len(w) == 1
        assert f"and {25 - dq.MAX_SUBJECTS_PER_WARNING} more" in w[0], w[0]
        # Truncation is presentational only.
        assert len(log.summary()["degraded_fields"]) == 25

    def test_an_unnamed_subject_produces_no_empty_parentheses(self):
        with dq.capture() as log:
            dq.record("vix", dq.SOURCE_UNAVAILABLE, detail="provider outage")
        w = log.summary()["warnings"]
        assert w == ["vix: unavailable — provider outage"], w

    def test_a_clean_run_warns_about_nothing(self):
        with dq.capture() as log:
            dq.record("spot:AAPL", dq.SOURCE_LIVE)
        assert log.summary()["warnings"] == []
