"""
Regression tests for the implied-volatility plausibility band.

The defect this band exists for: Yahoo publishes `impliedVolatility: 0.00001`
on stale or never-traded contracts. Pricing on it produced a premium of
effectively $0 and a delta around 1e-5, and the sizer — asked to cover a
position with a hedge whose delta was near zero — requested roughly 1.5
million contracts. The number was presented as a recommendation.

A plausibility band is a place where it is very easy to drift back into
fabrication: rejecting the reading is correct, *clamping* it to the floor is
not, because a clamped 5% IV is a priced, tradeable-looking recommendation
built on a quote nobody made. These tests pin both halves — that implausible
readings are rejected, and that rejection returns None rather than a
substitute.

Offline: `_chain_frame` is monkeypatched and the on-disk cache is bypassed, so
no test reads a real quote or writes a synthetic one back.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

import backend.data.options_chain as oc
from backend.data.options_chain import (
    MAX_PLAUSIBLE_IV,
    MIN_PLAUSIBLE_IV,
    _is_plausible_iv,
)


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    """
    Bypass the on-disk cache for the duration of each test.

    `data_cache` is backed by sqlite and shared with real runs, so a test that
    used it could be answered by a genuine cached quote — or, worse, write a
    synthetic IV into the cache that a later real run would read back as a
    market reading.
    """
    monkeypatch.setattr(oc, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(oc, "cache_set", lambda *a, **k: None)


def _chain(strikes_ivs) -> pd.DataFrame:
    return pd.DataFrame({
        "strike": [k for k, _ in strikes_ivs],
        "impliedVolatility": [iv for _, iv in strikes_ivs],
        "bid": [1.0] * len(strikes_ivs),
        "ask": [1.1] * len(strikes_ivs),
        "openInterest": [500] * len(strikes_ivs),
        "volume": [100] * len(strikes_ivs),
        "contractSymbol": [f"SPY270115P{int(k * 1000):08d}" for k, _ in strikes_ivs],
    })


def _patch_chain(monkeypatch, df):
    monkeypatch.setattr(
        oc, "_chain_frame", lambda ticker, expiry_str, option_type: df
    )


# ── The predicate itself ──────────────────────────────────────────────────────

class TestPlausibilityPredicate:

    @pytest.mark.parametrize("iv", [0.00001, 0.0, 0.01, 0.049])
    def test_stale_near_zero_readings_are_rejected(self, iv):
        """0.00001 is the exact artifact that produced the 1.5M-contract order."""
        assert not _is_plausible_iv(iv)

    @pytest.mark.parametrize("iv", [3.0, 5.0, 99.0, 1e6])
    def test_absurdly_high_readings_are_rejected(self, iv):
        assert not _is_plausible_iv(iv)

    @pytest.mark.parametrize("iv", [0.08, 0.15, 0.2473, 0.65, 1.5, 2.9])
    def test_real_market_readings_are_accepted(self, iv):
        """
        Spanning a quiet index (8%), a typical SPY put (24.7% — the figure
        measured on a live chain during verification), and a distressed
        single name (290%). The band must not be so tight that it discards
        genuine readings; that would be the same defect in reverse.
        """
        assert _is_plausible_iv(iv)

    @pytest.mark.parametrize("iv", [None, "", "abc", float("nan"), float("inf")])
    def test_non_numeric_and_non_finite_are_rejected(self, iv):
        """
        NaN is the one that matters: it passes a `> 0` check, propagates
        through Black-Scholes without raising, and emerges as a NaN premium.
        """
        assert not _is_plausible_iv(iv)

    def test_the_band_is_the_documented_one(self):
        assert (MIN_PLAUSIBLE_IV, MAX_PLAUSIBLE_IV) == (0.05, 3.0)


# ── get_strike_iv ─────────────────────────────────────────────────────────────

class TestStrikeIV:

    def test_an_implausible_exact_match_is_not_returned(self, monkeypatch):
        _patch_chain(monkeypatch, _chain([(650.0, 0.00001)]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_rejection_returns_none_rather_than_the_floor(self, monkeypatch):
        """
        Clamping to MIN_PLAUSIBLE_IV would turn a rejected quote into a
        priced recommendation. None forces the caller to disclose the gap.
        """
        _patch_chain(monkeypatch, _chain([(650.0, 0.00001)]))
        got = oc.get_strike_iv("SPY", "2027-01-15", 650.0)
        assert got is None
        assert got != MIN_PLAUSIBLE_IV

    def test_a_plausible_exact_match_is_returned(self, monkeypatch):
        _patch_chain(monkeypatch, _chain([(650.0, 0.2473)]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) == pytest.approx(0.2473)

    def test_interpolation_across_the_smile_respects_skew(self, monkeypatch):
        """
        The midpoint of a 0.30/0.20 smile is 0.25 — not either endpoint, and
        not an ATM reading applied flat across strikes.
        """
        _patch_chain(monkeypatch, _chain([(600.0, 0.30), (700.0, 0.20)]))
        got = oc.get_strike_iv("SPY", "2027-01-15", 650.0)
        assert got == pytest.approx(0.25, abs=0.001)

    def test_an_implausible_interpolation_input_is_not_averaged_in(
        self, monkeypatch
    ):
        """
        Interpolating 0.30 with a stale 0.00001 yields ~0.15, which sits
        inside the band and would pass every downstream check while being
        half composed of an artifact. The quoted strike nearest the target
        must be used instead.
        """
        _patch_chain(monkeypatch, _chain([(640.0, 0.00001), (660.0, 0.30)]))
        got = oc.get_strike_iv("SPY", "2027-01-15", 650.0)
        assert got == pytest.approx(0.30, abs=0.001)
        assert got != pytest.approx(0.150005, abs=0.001)   # the old answer

    def test_a_distant_strike_is_not_borrowed_from(self, monkeypatch):
        """
        A single quote far from the target is a different part of the
        surface. Returning it as this strike's own IV — which is how the
        caller labels a value from this function — overstates what was
        measured; None sends the caller to the disclosed `atm_iv` substitute.
        """
        _patch_chain(monkeypatch, _chain([(400.0, 0.25)]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_a_nearby_quote_is_still_usable(self, monkeypatch):
        """Within 2% the skew difference is small enough to stand in."""
        _patch_chain(monkeypatch, _chain([(655.0, 0.25)]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) == pytest.approx(0.25)

    def test_a_quote_just_outside_the_band_is_not(self, monkeypatch):
        _patch_chain(monkeypatch, _chain([(680.0, 0.25)]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_a_chain_with_no_iv_column_values_returns_none(self, monkeypatch):
        _patch_chain(monkeypatch, _chain([(650.0, float("nan"))]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None


# ── get_atm_iv ────────────────────────────────────────────────────────────────

class TestAtmIV:

    def test_an_implausible_atm_average_is_rejected(self, monkeypatch):
        monkeypatch.setattr(oc, "select_expiry_nearest", lambda *a, **k: "2027-01-15")
        monkeypatch.setattr(
            "backend.data.market_data.get_current_price", lambda t: 650.0
        )
        _patch_chain(
            monkeypatch,
            _chain([(645.0, 0.00001), (650.0, 0.00002), (655.0, 0.00001)]),
        )
        assert oc.get_atm_iv("SPY") is None

    def test_a_plausible_atm_average_is_returned(self, monkeypatch):
        monkeypatch.setattr(oc, "select_expiry_nearest", lambda *a, **k: "2027-01-15")
        monkeypatch.setattr(
            "backend.data.market_data.get_current_price", lambda t: 650.0
        )
        _patch_chain(
            monkeypatch, _chain([(645.0, 0.21), (650.0, 0.20), (655.0, 0.19)])
        )
        assert oc.get_atm_iv("SPY") == pytest.approx(0.20, abs=0.005)


# ── The quote precondition ────────────────────────────────────────────────────

def _unquoted_chain(strikes_ivs) -> pd.DataFrame:
    """A chain with IV but no two-sided quote — the out-of-hours shape."""
    df = _chain(strikes_ivs)
    df["bid"] = 0.0
    df["ask"] = 0.0
    df["openInterest"] = 0
    return df


# The ladder measured on the real SPY 2027-01-29 put chain outside the
# session: successive halvings of 1/16, applied in blocks across adjacent
# strikes, on 164 of 165 strikes. Its top rung clears MIN_PLAUSIBLE_IV, which
# is why the magnitude band alone could not reject it.
SOLVER_LADDER = [
    0.0625, 0.03126, 0.015635, 0.007822, 0.003916, 0.001963, 0.000987,
    0.000254, 0.00001,
]


class TestQuotePrecondition:

    def test_the_solver_ladders_top_rung_clears_the_magnitude_band(self):
        """
        Documents why the band is not sufficient on its own. 0.0625 is a
        provider artifact and `_is_plausible_iv` accepts it — correctly, since
        6.25% is a possible IV for some instrument. It is the missing quote,
        not the magnitude, that disqualifies it.
        """
        assert _is_plausible_iv(0.0625)
        assert not _is_plausible_iv(0.03126)

    def test_an_unquoted_strike_yields_no_contract_iv(self, monkeypatch):
        _patch_chain(monkeypatch, _unquoted_chain([(650.0, 0.0625)]))
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_the_whole_ladder_is_rejected(self, monkeypatch):
        """
        A chain in exactly the observed shape must produce no contract IV at
        any strike — not the 0.0625 top rung, and not an interpolation
        between two rungs.
        """
        strikes = [(650.0 + 5 * i, iv) for i, iv in enumerate(SOLVER_LADDER)]
        _patch_chain(monkeypatch, _unquoted_chain(strikes))
        for k, _ in strikes:
            assert oc.get_strike_iv("SPY", "2027-01-15", k) is None
        assert oc.get_strike_iv("SPY", "2027-01-15", 652.0) is None

    def test_the_one_quoted_strike_is_still_usable(self, monkeypatch):
        """
        The real chain had a single strike with a live two-sided quote and a
        credible IV. Screening must keep it rather than discarding the chain
        wholesale — that strike is genuinely priceable.
        """
        df = _unquoted_chain([(650.0, 0.0625), (674.0, 0.2502), (700.0, 0.03126)])
        df.loc[df["strike"] == 674.0, ["bid", "ask", "openInterest"]] = [6.40, 6.52, 120]
        _patch_chain(monkeypatch, df)

        assert oc.get_strike_iv("SPY", "2027-01-15", 674.0) == pytest.approx(0.2502)
        # And its neighbours are not interpolated out of unquoted rows.
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_an_inverted_skew_cannot_be_built_from_unquoted_rows(
        self, monkeypatch
    ):
        """
        The user-visible defect: a 90%-moneyness put priced at 6.25% vol
        against an at-the-money 16.4% — an inverted index skew, produced
        entirely from solver artifacts. Neither leg may be reported.
        """
        df = _unquoted_chain([(695.0, 0.03126), (772.0, 0.000254)])
        _patch_chain(monkeypatch, df)
        assert oc.get_strike_iv("SPY", "2027-01-15", 695.0) is None
        assert oc.get_strike_iv("SPY", "2027-01-15", 772.0) is None

    def test_a_one_sided_quote_is_not_a_quote(self, monkeypatch):
        """A bid with no ask cannot be traded against and cannot imply a vol."""
        df = _chain([(650.0, 0.25)])
        df["ask"] = 0.0
        _patch_chain(monkeypatch, df)
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_a_crossed_quote_is_rejected(self, monkeypatch):
        """bid > ask is a bad feed, not a market."""
        df = _chain([(650.0, 0.25)])
        df.loc[:, ["bid", "ask"]] = [5.0, 4.0]
        _patch_chain(monkeypatch, df)
        assert oc.get_strike_iv("SPY", "2027-01-15", 650.0) is None

    def test_the_reason_is_recorded_for_the_user(self, monkeypatch):
        """
        Silently returning None would leave the response unable to say why a
        hedge was model-priced. The data-quality ledger carries the reason.
        """
        from backend.data import data_quality as dq

        _patch_chain(monkeypatch, _unquoted_chain([(650.0, 0.0625)]))
        with dq.capture() as log:
            oc.get_strike_iv("SPY", "2027-01-15", 650.0)
            summary = log.summary()

        degraded = [d["name"] for d in summary["degraded_fields"]]
        assert any(n.startswith("iv:SPY") for n in degraded), summary
        detail = " ".join(
            d.get("detail") or "" for d in summary["degraded_fields"]
        )
        assert "two-sided quote" in detail


class TestAtmQuotePrecondition:

    def _patch_spot(self, monkeypatch, spot=772.0):
        monkeypatch.setattr(oc, "select_expiry_nearest", lambda *a, **k: "2027-01-15")
        monkeypatch.setattr(
            "backend.data.market_data.get_current_price", lambda t: spot
        )

    def test_an_unquoted_chain_has_no_atm_iv(self, monkeypatch):
        self._patch_spot(monkeypatch)
        _patch_chain(
            monkeypatch,
            _unquoted_chain([(770.0, 0.0625), (772.0, 0.03126), (775.0, 0.0625)]),
        )
        assert oc.get_atm_iv("SPY") is None

    def test_a_far_out_of_the_money_quote_is_not_reported_as_atm(
        self, monkeypatch
    ):
        """
        After screening, the surviving quotes can sit far from spot. Averaging
        them and labelling the result "at-the-money implied vol" would
        misstate the level that every model-priced premium is built on: the
        one quoted strike here is 13% out of the money.
        """
        self._patch_spot(monkeypatch, spot=772.0)
        df = _unquoted_chain([(674.0, 0.2502), (772.0, 0.000254)])
        df.loc[df["strike"] == 674.0, ["bid", "ask"]] = [6.40, 6.52]
        _patch_chain(monkeypatch, df)
        assert oc.get_atm_iv("SPY") is None

    def test_a_quoted_near_the_money_strike_is_used(self, monkeypatch):
        self._patch_spot(monkeypatch, spot=772.0)
        _patch_chain(monkeypatch, _chain([(770.0, 0.17), (772.0, 0.16), (775.0, 0.15)]))
        assert oc.get_atm_iv("SPY") == pytest.approx(0.16, abs=0.005)


class TestContractQuoteIV:

    def test_an_unquoted_contract_reports_no_implied_volatility(
        self, monkeypatch
    ):
        _patch_chain(monkeypatch, _unquoted_chain([(650.0, 0.0625)]))
        quote = oc.get_contract_quote("SPY", "2027-01-15", 650.0, "put")
        assert quote is not None
        assert quote["implied_volatility"] is None
        assert quote["has_two_sided_quote"] is False

    def test_a_quoted_contract_reports_its_implied_volatility(self, monkeypatch):
        _patch_chain(monkeypatch, _chain([(650.0, 0.2502)]))
        quote = oc.get_contract_quote("SPY", "2027-01-15", 650.0, "put")
        assert quote["implied_volatility"] == pytest.approx(0.2502)
        assert quote["has_two_sided_quote"] is True
