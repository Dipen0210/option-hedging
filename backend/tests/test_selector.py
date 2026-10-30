"""
Regression tests for the options instrument selector (Layer 4).

Every test here pins a defect found by running the real pipeline against the
live chain. Each one is deterministic: the pure scoring/quote/expiry helpers are
exercised directly, and the one end-to-end test stubs only the chain data
functions at the data-layer boundary — the selector itself is never mocked.

Defects pinned:
  * synthetic $5-grid strikes that were never listed      → _resolve_strike
  * a 7-day expiry offered as a 180-day hedge             → _sample_expiries
  * a 34.7%-spread / 20-OI contract ranked best in book   → _liquidity_score
  * ranking computed on the model price, not the fill     → _rescore
  * a credit collar clamped to a $0.01 debit              → _apply_net_market_quote
  * every holding handed the whole portfolio budget       → _holding_budget_pct
  * a model-priced contract with no market offered as #1  → executability filter
"""

import pytest

from backend.data import data_quality as dq
from backend.instruments.options.selector import (
    OptionsSelector,
    GOOD_OI_CONTRACTS,
    MAX_GOOD_SPREAD_PCT,
    MIN_HORIZON_COVERAGE,
    MAX_EXECUTABLE_SPREAD_PCT,
    W_LIQUIDITY,
    is_executably_quoted,
)
from backend.engines.layer_04_instrument_selection import InstrumentSelectionEngine
from backend.models.hedge_models import InstrumentCandidate
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.risk_models import RiskProfile, RegimeState
from backend.data.options_chain import OptionsDataUnavailable


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _portfolio(**overrides) -> PortfolioInput:
    kwargs = dict(
        holdings=[
            StockPosition(ticker="MSFT", shares=100, purchase_price=390.0,
                          purchase_date="2024-01-15"),
        ],
        total_notional=100_000.0,
        max_hedge_cost_pct=0.05,
        hedge_horizon_days=180,
    )
    kwargs.update(overrides)
    return PortfolioInput(**kwargs)


def _profile(**overrides) -> RiskProfile:
    kwargs = dict(
        ticker="MSFT", asset_class="equity", notional_value=100_000.0,
        beta_vs_spy=1.0, var_5pct=15_000.0, cvar_5pct=22_000.0,
        var_pct=0.15, cvar_pct=0.22, notional_at_risk=15_000.0,
        optimal_hedge_ratio=1.0, regression_hedge_ratio=1.0, tail_correlation=0.7,
    )
    kwargs.update(overrides)
    return RiskProfile(**kwargs)


def _quote(bid, ask, *, oi=1000, volume=50, iv=0.25, symbol="TEST000000P00100000",
           strike=100.0, two_sided=None):
    """A quote dict shaped exactly like options_chain.get_contract_quote."""
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
    return {
        "ticker": "TEST", "expiry": "2027-04-16", "strike": strike,
        "option_type": "put", "contract_symbol": symbol,
        "bid": bid, "ask": ask, "mid": mid, "last": mid,
        "implied_volatility": iv, "open_interest": oi, "volume": volume,
        "spread_pct": round((ask - bid) / mid, 4) if mid > 0 else None,
        "has_two_sided_quote": (bid > 0 and ask > 0) if two_sided is None else two_sided,
    }


def _regime_state(**overrides) -> RegimeState:
    """mid_vol regime matching the 2026-10-28 verification run."""
    kwargs = dict(
        regime_label="mid_vol", regime_id=1, is_anomaly=False,
        anomaly_score=0.1, soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3, vix_level=16.91,
        vol_forecast_garch=0.1212, realized_vol_20d=0.0999,
    )
    kwargs.update(overrides)
    return RegimeState(**kwargs)


def _candidate(**overrides) -> InstrumentCandidate:
    kwargs = dict(
        instrument_type="option", asset_class="equity", ticker="MSFT",
        strategy="Protective Put", strike=100.0, expiry_date="2027-04-16",
        option_type="put", n_contracts=10, total_cost=5_000.0,
        basis_risk_r2=1.0, lambda_leverage=5.0, delta=-0.5,
    )
    kwargs.update(overrides)
    return InstrumentCandidate(**kwargs)


# ── Liquidity scoring ─────────────────────────────────────────────────────────

class TestLiquidityScore:
    """
    The live run that motivated this: MSFT261104P00515000 had 20 contracts of
    open interest and a 34.67% bid-ask spread, and scored 75.3 — the best
    candidate in the entire book — because scoring ignored tradeability.
    """

    def test_wide_spread_thin_depth_scores_near_zero(self):
        score = OptionsSelector._liquidity_score(
            open_interest=20, daily_volume=0, spread_pct=0.3467, has_quote=True,
        )
        # Spread is 3.5x the acceptable ceiling, depth is 4% of the good level.
        assert score < 0.05

    def test_tight_spread_deep_market_scores_near_one(self):
        score = OptionsSelector._liquidity_score(
            open_interest=GOOD_OI_CONTRACTS * 3, daily_volume=500,
            spread_pct=0.005, has_quote=True,
        )
        assert score > 0.95

    def test_no_quote_scores_zero_not_neutral(self):
        """
        An unquotable contract must not be scored as average. Treating "unknown"
        as neutral let a contract with no market outrank one with a real market.
        """
        assert OptionsSelector._liquidity_score(
            open_interest=50_000, daily_volume=50_000, spread_pct=0.001,
            has_quote=False,
        ) == 0.0

    def test_spread_at_ceiling_contributes_nothing(self):
        at_ceiling = OptionsSelector._liquidity_score(
            open_interest=0, daily_volume=0,
            spread_pct=MAX_GOOD_SPREAD_PCT, has_quote=True,
        )
        assert at_ceiling == pytest.approx(0.0)

    def test_depth_uses_whichever_of_oi_or_volume_is_larger(self):
        """A contract trading actively today is liquid even with stale OI."""
        oi_only  = OptionsSelector._liquidity_score(GOOD_OI_CONTRACTS, 0, 0.01, True)
        vol_only = OptionsSelector._liquidity_score(0, GOOD_OI_CONTRACTS, 0.01, True)
        assert oi_only == pytest.approx(vol_only)

    def test_score_is_bounded_in_unit_interval(self):
        for oi, vol, spread in [(0, 0, 5.0), (10**9, 10**9, 0.0001), (20, 3, 0.35)]:
            s = OptionsSelector._liquidity_score(oi, vol, spread, True)
            assert 0.0 <= s <= 1.0


# ── Composite score ──────────────────────────────────────────────────────────

class TestScore:

    def _base(self, **kw):
        args = dict(delta_offset=500.0, target_delta=1000.0, cost_pct=0.02,
                    max_cost_pct=0.05, basis_r2=1.0, lambda_=5.0)
        args.update(kw)
        return OptionsSelector._score(**args)

    def test_illiquid_contract_cannot_outrank_identical_liquid_one(self):
        """The exact inversion seen in production, now impossible."""
        liquid   = self._base(liquidity=1.0)
        illiquid = self._base(liquidity=0.0)
        assert liquid > illiquid
        assert liquid - illiquid == pytest.approx(W_LIQUIDITY, abs=0.01)

    def test_liquidity_defaults_to_zero_when_unsupplied(self):
        """
        A caller that has not read the chain must not get free liquidity marks.
        """
        assert self._base() == self._base(liquidity=0.0)

    def test_credit_structure_earns_full_cost_marks_without_overflow(self):
        """
        A credit collar has negative cost_pct. It should max the cost component,
        not scale past it and inflate the total beyond 100.
        """
        credit = self._base(cost_pct=-0.03, liquidity=1.0)
        free   = self._base(cost_pct=0.0,   liquidity=1.0)
        assert credit == free
        assert credit <= 100.0

    def test_perfect_candidate_scores_100(self):
        assert OptionsSelector._score(
            delta_offset=1000.0, target_delta=1000.0, cost_pct=0.0,
            max_cost_pct=0.05, basis_r2=1.0, lambda_=5.0, liquidity=1.0,
        ) == pytest.approx(100.0)

    def test_over_budget_cost_scores_zero_not_negative(self):
        over = self._base(cost_pct=0.50, max_cost_pct=0.05, liquidity=0.0)
        assert over >= 0.0


# ── Expiry horizon coverage ──────────────────────────────────────────────────

class TestExpiryHorizonCoverage:
    """
    MSFT was handed a 2026-11-04 put — 7 days out — as the hedge for a 180-day
    horizon, leaving the position naked for 173 days and pricing it off a
    near-dated, earnings-inflated IV.
    """

    @staticmethod
    def _days_out(n: int) -> str:
        from datetime import date, timedelta
        return (date.today() + timedelta(days=n)).isoformat()

    def test_expiry_far_short_of_horizon_is_excluded(self):
        expiries = [self._days_out(d) for d in (7, 14, 100, 150, 175)]
        picked = OptionsSelector._sample_expiries(
            expiries, 3, horizon_days=180, ticker="MSFT",
        )
        assert self._days_out(7) not in picked
        assert self._days_out(14) not in picked
        assert set(picked) <= {self._days_out(100), self._days_out(150), self._days_out(175)}

    def test_coverage_threshold_is_half_the_horizon(self):
        just_under = int(180 * MIN_HORIZON_COVERAGE) - 2
        just_over  = int(180 * MIN_HORIZON_COVERAGE) + 2
        expiries = [self._days_out(just_under), self._days_out(just_over)]
        picked = OptionsSelector._sample_expiries(expiries, 3, horizon_days=180)
        assert picked == [self._days_out(just_over)]

    def test_falls_back_to_longest_and_discloses_when_none_cover(self):
        """
        An illiquid name may list nothing long-dated. A short hedge beats no
        hedge, but the shortfall must be disclosed, not presented as full cover.
        """
        expiries = [self._days_out(10), self._days_out(30)]
        with dq.capture() as log:
            picked = OptionsSelector._sample_expiries(
                expiries, 3, horizon_days=180, ticker="ILLQ",
            )
        assert picked == expiries          # nothing dropped — a hedge is still offered
        names = [p.name for p in log.degradations]
        assert "expiry_coverage:ILLQ" in names
        detail = next(p.detail for p in log.degradations
                      if p.name == "expiry_coverage:ILLQ")
        assert "180" in detail

    def test_no_horizon_given_preserves_previous_behaviour(self):
        expiries = [self._days_out(d) for d in (7, 90, 400)]
        assert OptionsSelector._sample_expiries(expiries, 3) == expiries

    def test_empty_chain_returns_empty(self):
        assert OptionsSelector._sample_expiries([], 3, horizon_days=180) == []


# ── Single-leg quote reconciliation ──────────────────────────────────────────

class TestApplyMarketQuote:

    def test_executable_cost_replaces_model_cost(self):
        """
        The user must be quoted what they would pay. The model said $15.48; the
        market mid was $17.73 — a 14.5% difference on a real AAPL contract.
        """
        cand = _candidate(n_contracts=10, total_cost=15_477.0)
        OptionsSelector._apply_market_quote(
            cand, _quote(17.40, 18.05, symbol="AAPL270416P00335000"),
            model_premium=15.4773,
        )
        assert cand.premium_source == "market_mid"
        assert cand.market_premium == pytest.approx(17.725)
        assert cand.total_cost == pytest.approx(17.725 * 100 * 10)
        assert cand.model_premium == pytest.approx(15.4773)
        assert cand.premium_vs_model_pct == pytest.approx(0.1452, abs=1e-4)
        assert cand.contract_symbol == "AAPL270416P00335000"
        assert cand.strike_is_listed is True

    def test_one_sided_quote_stays_on_model_price(self):
        cand = _candidate()
        OptionsSelector._apply_market_quote(
            cand, _quote(0.0, 18.05, two_sided=False), model_premium=15.0,
        )
        assert cand.premium_source == "model"
        assert cand.market_premium == 0.0
        # Still a real listed contract — the symbol is recorded.
        assert cand.strike_is_listed is True

    def test_missing_quote_leaves_candidate_unlisted(self):
        cand = _candidate()
        OptionsSelector._apply_market_quote(cand, None, model_premium=15.0)
        assert cand.premium_source == "model"
        assert cand.strike_is_listed is False
        assert cand.contract_symbol is None

    def test_contract_iv_is_recorded_as_its_own_source(self):
        cand = _candidate()
        OptionsSelector._apply_market_quote(
            cand, _quote(14.94, 15.05, iv=0.1321), model_premium=13.45,
        )
        assert cand.iv_used == pytest.approx(0.1321)
        assert cand.iv_source == "contract_iv"


# ── Two-leg quote reconciliation ─────────────────────────────────────────────

class TestApplyNetMarketQuote:

    def test_credit_collar_net_premium_stays_negative(self):
        """
        A collar whose short call is worth more than its long put is a CREDIT.
        Two separate code paths previously clamped it: `net_prc = 0.01` in the
        builder and `max(long_mid - short_mid, 0.0)` in enrichment. Both hid a
        real cash inflow and fed a wrong price to the budget gate.
        """
        cand = _candidate(strategy="Collar", short_strike=120.0,
                          short_option_type="call", n_contracts=5)
        sel = OptionsSelector()
        sel._apply_net_market_quote(
            cand,
            long_quote=_quote(4.00, 4.20, symbol="SPY270331P00770000"),
            short_quote=_quote(9.00, 9.20, symbol="SPY270331C00825000"),
            model_premium=-4.9,
        )
        assert cand.market_premium == pytest.approx(-5.0)
        assert cand.market_total_cost == pytest.approx(-2_500.0)
        assert cand.total_cost == pytest.approx(-2_500.0)
        assert cand.premium_source == "market_mid"

    def test_both_leg_symbols_are_recorded(self):
        cand = _candidate(short_strike=90.0, short_option_type="put")
        sel = OptionsSelector()
        sel._apply_net_market_quote(
            cand,
            long_quote=_quote(20.12, 20.31, symbol="SPY270129P00772000"),
            short_quote=_quote(8.70, 8.86, symbol="SPY270129P00695000"),
            model_premium=11.0,
        )
        assert cand.contract_symbol == "SPY270129P00772000"
        assert cand.short_contract_symbol == "SPY270129P00695000"

    def test_transaction_cost_counts_both_legs(self):
        """
        The user crosses two spreads, and against a small net premium that cost
        is proportionally far larger than either leg's own spread suggests.
        """
        cand = _candidate(short_strike=90.0, short_option_type="put")
        sel = OptionsSelector()
        sel._apply_net_market_quote(
            cand,
            long_quote=_quote(20.00, 20.20),     # $0.20 wide
            short_quote=_quote(18.00, 18.20),    # $0.20 wide
            model_premium=2.0,
        )
        # net mid = 2.00, combined spread = 0.40 → 20% of the net premium,
        # versus ~1% measured on the long leg alone.
        assert cand.market_premium == pytest.approx(2.0)
        assert cand.bid_ask_spread_pct == pytest.approx(0.20, abs=1e-4)

    def test_depth_is_the_thinner_leg(self):
        cand = _candidate(short_strike=90.0, short_option_type="put")
        sel = OptionsSelector()
        sel._apply_net_market_quote(
            cand,
            long_quote=_quote(20.0, 20.2, oi=5000, volume=900),
            short_quote=_quote(8.0, 8.2, oi=12, volume=3),
            model_premium=11.9,
        )
        assert cand.open_interest == 12
        assert cand.daily_volume == 3

    @pytest.mark.parametrize("missing", ["long", "short"])
    def test_one_unquotable_leg_makes_the_structure_unexecutable(self, missing):
        cand = _candidate(short_strike=90.0, short_option_type="put")
        sel = OptionsSelector()
        sel._apply_net_market_quote(
            cand,
            long_quote=None if missing == "long" else _quote(20.0, 20.2),
            short_quote=None if missing == "short" else _quote(8.0, 8.2),
            model_premium=11.9,
        )
        assert cand.premium_source == "model"
        assert cand.market_total_cost == 0.0

    def test_one_sided_leg_quote_is_not_executable(self):
        cand = _candidate(short_strike=90.0, short_option_type="put")
        sel = OptionsSelector()
        sel._apply_net_market_quote(
            cand,
            long_quote=_quote(20.0, 20.2),
            short_quote=_quote(0.0, 8.2, two_sided=False),
            model_premium=11.9,
        )
        assert cand.premium_source == "model"


# ── Re-ranking on the executable price ───────────────────────────────────────

class TestRescore:

    def test_rescore_uses_executable_cost_and_quoted_liquidity(self):
        """
        Scoring ran before the chain was read, so it could not see the fill
        price or the market. Re-scoring is what makes the ranking honest.
        """
        sel = OptionsSelector()
        portfolio = _portfolio()

        thin = _candidate(n_contracts=10)
        sel._apply_market_quote(thin, _quote(6.20, 8.80, oi=20, volume=1),
                                model_premium=8.68)
        sel._rescore(thin, delta_offset=500.0, target_delta=1000.0,
                     portfolio=portfolio)

        deep = _candidate(n_contracts=10)
        sel._apply_market_quote(deep, _quote(7.45, 7.55, oi=1380, volume=200),
                                model_premium=8.68)
        sel._rescore(deep, delta_offset=500.0, target_delta=1000.0,
                     portfolio=portfolio)

        # Near-identical mid prices; the only real difference is tradeability.
        assert thin.market_premium == pytest.approx(7.5)
        assert deep.market_premium == pytest.approx(7.5)
        assert deep.score > thin.score

    def test_unquotable_candidate_loses_the_whole_liquidity_component(self):
        sel = OptionsSelector()
        portfolio = _portfolio()

        quoted = _candidate(n_contracts=10)
        sel._apply_market_quote(quoted, _quote(7.45, 7.55, oi=GOOD_OI_CONTRACTS * 2),
                                model_premium=7.5)
        sel._rescore(quoted, delta_offset=1000.0, target_delta=1000.0,
                     portfolio=portfolio)

        unquoted = _candidate(n_contracts=10, total_cost=quoted.total_cost)
        sel._apply_market_quote(unquoted, None, model_premium=7.5)
        sel._rescore(unquoted, delta_offset=1000.0, target_delta=1000.0,
                     portfolio=portfolio)

        # The gap is exactly the liquidity component the quote earned: the
        # unquotable contract scores 0 there, and nothing else differs.
        earned = OptionsSelector._liquidity_score(
            quoted.open_interest, quoted.daily_volume,
            quoted.bid_ask_spread_pct, has_quote=True,
        )
        assert quoted.score - unquoted.score == pytest.approx(
            W_LIQUIDITY * earned, abs=0.02)
        # A tight, deep market should still capture most of the component.
        assert earned > 0.9


# ── Portfolio budget allocation ──────────────────────────────────────────────

class TestHoldingBudget:
    """
    P0-6: the per-holding budget was
        max_hedge_cost_pct * (portfolio_notional / holding_notional)
    which handed EVERY holding the entire portfolio budget. A three-holding
    portfolio with a 5% cap could be sized to spend 15%.
    """

    def test_budget_does_not_scale_with_holding_size(self):
        portfolio = _portfolio(max_hedge_cost_pct=0.05)
        small = OptionsSelector._holding_budget_pct(
            _profile(notional_value=1_000.0), portfolio)
        large = OptionsSelector._holding_budget_pct(
            _profile(notional_value=900_000.0), portfolio)
        assert small == large == 0.05

    def test_pro_rata_allocation_sums_within_the_cap(self):
        portfolio = _portfolio(max_hedge_cost_pct=0.05, total_notional=300_000.0)
        weights = [100_000.0, 150_000.0, 50_000.0]
        spend = sum(
            OptionsSelector._holding_budget_pct(_profile(notional_value=w), portfolio) * w
            for w in weights
        )
        assert spend <= portfolio.max_hedge_cost_pct * sum(weights) + 1e-9


# ── Strike resolution against the real ladder ────────────────────────────────

class TestResolveStrike:
    """
    Strikes were generated as `round(spot * moneyness / 5) * 5`, which assumes a
    universal $5 ladder. Verified against the live chain on 2026-10-28: with SPY
    at 772.31 and expiry 2026-12-18, the ATM target resolves to the listed 772
    strike while the $5 grid produced 770, and the 0.97-moneyness target
    resolves to 749 where the grid produced 750. Neither grid strike was listed,
    so every number attached to those candidates described a contract that did
    not exist.
    """

    SPY_LADDER = [745.0, 746.0, 747.0, 748.0, 749.0, 750.0, 770.0, 771.0,
                  772.0, 773.0, 774.0, 775.0]

    @pytest.fixture
    def ladder(self, monkeypatch):
        def _fake(ticker, expiry_str, option_type="put"):
            return list(self.SPY_LADDER)
        monkeypatch.setattr(
            "backend.data.options_chain.get_listed_strikes", _fake)
        return _fake

    def test_snaps_to_the_listed_strike_not_the_five_dollar_grid(self, ladder):
        sel = OptionsSelector()
        resolved = sel._resolve_strike("SPY", "2026-12-18", 772.31, "put")
        assert resolved == 772.0
        assert round(772.31 / 5) * 5 == 770.0          # what the old code produced
        assert 770.0 in self.SPY_LADDER                 # listed here, but not nearest

    def test_snaps_to_nearest_even_when_grid_would_disagree(self, ladder):
        sel = OptionsSelector()
        assert sel._resolve_strike("SPY", "2026-12-18", 749.14, "put") == 749.0
        assert round(749.14 / 5) * 5 == 750.0

    def test_returns_none_when_nothing_listed_is_close_enough(self, ladder):
        """
        Beyond the snap tolerance the contract no longer delivers the protection
        level the user asked for, so the candidate must be skipped rather than
        quoted at a strike that misrepresents it.
        """
        sel = OptionsSelector()
        assert sel._resolve_strike("SPY", "2026-12-18", 400.0, "put") is None

    def test_empty_ladder_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "backend.data.options_chain.get_listed_strikes",
            lambda *a, **k: [])
        sel = OptionsSelector()
        assert sel._resolve_strike("SPY", "2026-12-18", 772.31, "put") is None

    def test_chain_outage_returns_none_and_is_disclosed(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("chain unreachable")
        monkeypatch.setattr(
            "backend.data.options_chain.get_listed_strikes", _boom)
        sel = OptionsSelector()
        with dq.capture() as log:
            assert sel._resolve_strike("SPY", "2026-12-18", 772.31, "put") is None
        assert any(p.name.startswith("strikes:SPY") and p.source == dq.SOURCE_UNAVAILABLE
                   for p in log.degradations)


# ── Executability filter ─────────────────────────────────────────────────────

class TestExecutabilityFilter:
    """
    Pins the inversion found in the live run: MSFT270416P00515000 had zero open
    interest and no two-sided quote, so its $1,726.04 "cost" was a model number,
    yet it ranked above three SPY puts with genuine ~1% markets.
    """

    def test_unquotable_candidates_are_dropped_when_a_real_market_exists(self):
        quoted   = _candidate(strategy="Proxy Put", premium_source="market_mid")
        unquoted = _candidate(strategy="Direct Put", premium_source="model")
        kept = OptionsSelector._filter_executable([unquoted, quoted], "MSFT")
        assert kept == [quoted]

    def test_all_unquotable_are_kept_but_flagged_and_disclosed(self):
        """
        Returning nothing would be worse than returning disclosed theoretical
        prices — but the user must be told which they are looking at.
        """
        a = _candidate(strategy="Protective Put", premium_source="model")
        b = _candidate(strategy="Bear Put Spread", premium_source="model")
        with dq.capture() as log:
            kept = OptionsSelector._filter_executable([a, b], "ILLQ")

        assert len(kept) == 2
        assert all(c.liquidity_ok is False for c in kept)
        assert all(any("theoretical" in con for con in c.cons) for c in kept)

        degraded = {p.name: p for p in log.degradations}
        assert "quotes:ILLQ" in degraded
        # UNAVAILABLE, not FALLBACK: nothing was substituted for the quote —
        # there is no quote. A fallback implies a documented stand-in exists.
        assert degraded["quotes:ILLQ"].source == dq.SOURCE_UNAVAILABLE

    def test_fully_quotable_set_passes_through_untouched(self):
        cands = [_candidate(premium_source="market_mid") for _ in range(3)]
        kept = OptionsSelector._filter_executable(cands, "SPY")
        assert kept == cands
        assert all(c.liquidity_ok is True for c in kept)
        assert all(not any("theoretical" in con for con in c.cons) for c in kept)

    def test_empty_input_returns_empty(self):
        assert OptionsSelector._filter_executable([], "SPY") == []

    def test_untradeably_wide_quote_is_rejected_like_a_missing_one(self):
        """
        A live LQD 2027-02-19 99 put quoted 0.05 / 3.00 has a bid-ask spread of
        198% of its mid. It was scoring 49.8 and ranking mid-book, because a
        wide spread only zeroed the 15-point liquidity term. "Cost $773.68" is
        not a number the user can act on when the fill is in a 60x range.
        """
        tight = _candidate(strategy="Macro Put", premium_source="market_mid",
                           bid_ask_spread_pct=0.02)
        wide  = _candidate(strategy="Macro Put", premium_source="market_mid",
                           bid_ask_spread_pct=1.9787)
        with dq.capture() as log:
            kept = OptionsSelector._filter_executable([wide, tight], "LQD")

        assert kept == [tight]
        assert any(p.name == "spreads:LQD" for p in log.degradations)

    def test_wide_quotes_only_fall_back_to_disclosed_theoretical_pricing(self):
        """
        If every quote is untradeably wide there is no executable price at all,
        so the set must be disclosed rather than silently presented as quoted.
        """
        a = _candidate(premium_source="market_mid", bid_ask_spread_pct=0.68)
        b = _candidate(premium_source="market_mid", bid_ask_spread_pct=1.98)
        with dq.capture() as log:
            kept = OptionsSelector._filter_executable([a, b], "LQD")

        assert len(kept) == 2
        assert all(c.liquidity_ok is False for c in kept)
        degraded = {p.name for p in log.degradations}
        assert "quotes:LQD" in degraded and "spreads:LQD" in degraded

    def test_spread_at_the_limit_is_kept(self):
        """The bound is inclusive — exactly 50% is wide, not disqualifying."""
        at_limit = _candidate(premium_source="market_mid",
                              bid_ask_spread_pct=0.50)
        assert OptionsSelector._filter_executable([at_limit], "X") == [at_limit]

    def test_quote_with_unmeasured_spread_is_not_disqualified(self):
        """
        bid_ask_spread_pct is None when the spread was never measured. None must
        not be treated as "infinitely wide" — the quote itself is still real.
        """
        c = _candidate(premium_source="market_mid", bid_ask_spread_pct=None)
        assert OptionsSelector._filter_executable([c], "X") == [c]

    def test_no_recommendation_is_forced_when_there_are_no_candidates(self):
        """
        "Do not force a recommendation when no valid hedge exists" — an empty
        candidate set must stay empty, never be back-filled.
        """
        with dq.capture() as log:
            assert OptionsSelector._filter_executable([], "SPY") == []
        assert log.degradations == []


# ── Unmeasured vs measured-zero liquidity ────────────────────────────────────

class TestUnmeasuredLiquidity:
    """
    Pins the overnight-data defect found at 00:38 ET: Yahoo returned the full
    SPY strike ladder with bid=ask=openInterest=0 for every expiry, including
    1-DTE and the standard third-Friday LEAPs. The pipeline wrote those zeros
    into the candidate and the run reported

        Protective Put SPY K=772.0 ... src=model spread=0.0000 OI=0

    A 0.0000 bid-ask spread reads as a perfectly tight market. The truth was
    that there was no market data at all, so these fields must stay None.
    """

    def test_unquoted_contract_leaves_depth_and_spread_unmeasured(self):
        sel  = OptionsSelector()
        cand = _candidate()
        # The row exists in the chain — it just has no market, which is exactly
        # what the provider returns outside trading hours.
        quote = _quote(0.0, 0.0, oi=0, volume=0, two_sided=False)
        sel._apply_market_quote(cand, quote, model_premium=19.90)

        assert cand.premium_source == "model"
        assert cand.open_interest is None
        assert cand.daily_volume is None
        assert cand.bid_ask_spread_pct is None

    def test_quoted_contract_records_real_depth_and_spread(self):
        sel  = OptionsSelector()
        cand = _candidate()
        sel._apply_market_quote(
            cand, _quote(20.12, 20.31, oi=408, volume=35), model_premium=19.90,
        )
        assert cand.premium_source == "market_mid"
        assert cand.open_interest == 408
        assert cand.bid_ask_spread_pct is not None

    def test_measured_zero_spread_is_not_confused_with_unmeasured(self):
        """
        A genuinely measured 0.0 (bid == ask) is a different fact from None and
        must survive as 0.0 rather than being normalised away.
        """
        sel  = OptionsSelector()
        cand = _candidate()
        sel._apply_market_quote(
            cand, _quote(2.00, 2.00, oi=900, volume=10), model_premium=2.0,
        )
        assert cand.premium_source == "market_mid"
        assert cand.bid_ask_spread_pct == 0.0

    def test_liquidity_score_handles_unmeasured_fields(self):
        """None must not raise, and must not earn depth or tightness marks."""
        assert OptionsSelector._liquidity_score(
            open_interest=None, daily_volume=None, spread_pct=None,
            has_quote=False,
        ) == 0.0
        # has_quote=True with an unmeasured spread: depth alone is the evidence.
        assert OptionsSelector._liquidity_score(
            open_interest=None, daily_volume=None, spread_pct=None,
            has_quote=True,
        ) == pytest.approx(0.3, abs=1e-9)

    def test_rescore_of_an_unquoted_candidate_awards_no_liquidity_points(self):
        sel  = OptionsSelector()
        cand = _candidate(premium_source="model", total_cost=4_691.68)
        cand.open_interest = None
        cand.bid_ask_spread_pct = None
        sel._rescore(cand, delta_offset=0.0, target_delta=1.0,
                     portfolio=_portfolio())
        quoted = _candidate(premium_source="market_mid", total_cost=4_691.68,
                            open_interest=4_000, bid_ask_spread_pct=0.01)
        sel._rescore(quoted, delta_offset=0.0, target_delta=1.0,
                     portfolio=_portfolio())
        assert quoted.score > cand.score
        # The quoted candidate earns the liquidity component it actually merits:
        # full depth marks (4,000 OI) and 90% of the tightness marks for a 1%
        # spread → 0.6*0.9 + 0.4*1.0 = 0.94 of W_LIQUIDITY. The unquoted one
        # earns none of it, and nothing else about the two differs.
        earned = OptionsSelector._liquidity_score(
            open_interest=4_000, daily_volume=0, spread_pct=0.01, has_quote=True,
        )
        assert earned == pytest.approx(0.94, abs=1e-9)
        assert quoted.score - cand.score == pytest.approx(
            W_LIQUIDITY * earned, abs=0.02
        )


# ── Volatility provenance ────────────────────────────────────────────────────

class TestVolProvenance:
    """
    Found in the 2026-10-28 overnight run, where no option chain had a usable
    implied vol. Every candidate in the book — AAPL, SPY and MSFT — was priced
    at sigma=0.1212 and reported iv_source='atm_iv'.

    Two defects compounded:

      1. `_get_vol` preferred `regime.vol_forecast_garch` over per-ticker
         realized vol, but that is a single MARKET-level forecast from Layer 2.
         One number priced the whole book, while the tickers' own 20-day
         realized vols were 0.1362 / 0.0999 / 0.1956. MSFT was marked at 12%
         while realising 20%.
      2. `_price_option` hardcoded `iv_source = "atm_iv"`, so a premium derived
         from that market-wide forecast was presented to the user as coming
         from the at-the-money option market.
    """

    @staticmethod
    def _regime(garch=0.1212):
        return _regime_state(vol_forecast_garch=garch)

    def test_atm_iv_is_preferred_and_labelled(self, monkeypatch):
        from backend.data import options_chain
        monkeypatch.setattr(options_chain, "get_atm_iv",
                            lambda t, ot="put": 0.2473)
        sigma, source = OptionsSelector()._get_vol(
            "SPY", _profile(), self._regime(),
        )
        assert (sigma, source) == (0.2473, "atm_iv")

    def test_ticker_realized_vol_beats_market_garch(self, monkeypatch):
        """
        The ordering defect. With no ATM IV, MSFT must price on MSFT's own
        19.56% realized vol, not the market's 12.12% GARCH forecast.
        """
        from backend.data import options_chain, market_data
        monkeypatch.setattr(options_chain, "get_atm_iv",
                            lambda t, ot="put": None)
        monkeypatch.setattr(market_data, "get_historical_volatility",
                            lambda t, *a, **k: {"MSFT": 0.1956,
                                                "SPY": 0.0999,
                                                "AAPL": 0.1362}[t])
        sel = OptionsSelector()
        for ticker, expected in [("MSFT", 0.1956), ("SPY", 0.0999),
                                 ("AAPL", 0.1362)]:
            sigma, source = sel._get_vol(ticker, _profile(), self._regime())
            assert source == "realized_vol"
            assert sigma == pytest.approx(expected)
            assert sigma != pytest.approx(0.1212), (
                "priced on the market-wide GARCH forecast instead of the "
                "ticker's own volatility"
            )

    def test_distinct_tickers_do_not_share_one_vol(self, monkeypatch):
        """The symptom that exposed it: one sigma across the whole book."""
        from backend.data import options_chain, market_data
        monkeypatch.setattr(options_chain, "get_atm_iv",
                            lambda t, ot="put": None)
        monkeypatch.setattr(market_data, "get_historical_volatility",
                            lambda t, *a, **k: {"MSFT": 0.1956,
                                                "SPY": 0.0999,
                                                "AAPL": 0.1362}[t])
        sel = OptionsSelector()
        sigmas = {
            t: sel._get_vol(t, _profile(), self._regime())[0]
            for t in ("MSFT", "SPY", "AAPL")
        }
        assert len(set(sigmas.values())) == 3, sigmas

    def test_market_garch_is_the_last_resort_and_is_disclosed(self, monkeypatch):
        from backend.data import options_chain, market_data
        monkeypatch.setattr(options_chain, "get_atm_iv",
                            lambda t, ot="put": None)
        monkeypatch.setattr(market_data, "get_historical_volatility",
                            lambda t, *a, **k: (_ for _ in ()).throw(
                                RuntimeError("no history")))
        with dq.capture() as log:
            sigma, source = OptionsSelector()._get_vol(
                "XYZ", _profile(), self._regime(),
            )
        assert (sigma, source) == (0.1212, "garch_forecast")
        detail = {p.name: p.detail for p in log.degradations}["iv:XYZ"]
        assert "market-wide" in detail and "not specific" in detail

    def test_no_vol_at_all_raises_rather_than_inventing_one(self, monkeypatch):
        from backend.data import options_chain, market_data
        monkeypatch.setattr(options_chain, "get_atm_iv",
                            lambda t, ot="put": None)
        monkeypatch.setattr(market_data, "get_historical_volatility",
                            lambda t, *a, **k: (_ for _ in ()).throw(
                                RuntimeError("no history")))
        with pytest.raises(OptionsDataUnavailable):
            OptionsSelector()._get_vol("XYZ", _profile(),
                                       self._regime(garch=0.0))

    def test_price_option_reports_the_callers_vol_source(self):
        """
        `iv_source` must describe where sigma came from. It was hardcoded to
        "atm_iv", which told the user a GARCH-derived premium was a market
        implied-vol premium.
        """
        sel = OptionsSelector()
        out = sel._price_option(
            "put", S=100.0, K=95.0, r=0.04, q=0.0, sigma=0.1212, T=0.5,
            ticker="XYZ", expiry_str=None, sigma_source="garch_forecast",
        )
        assert out is not None
        assert out["iv_source"] == "garch_forecast"
        assert out["sigma_used"] == pytest.approx(0.1212)

    def test_price_option_still_prefers_the_strikes_own_iv(self, monkeypatch):
        """Skew must win over whatever the caller supplied."""
        from backend.data import options_chain
        monkeypatch.setattr(options_chain, "get_strike_iv",
                            lambda t, e, k, ot: 0.2473)
        out = OptionsSelector()._price_option(
            "put", S=100.0, K=85.0, r=0.04, q=0.0, sigma=0.0999, T=0.5,
            ticker="SPY", expiry_str="2027-03-19",
            sigma_source="realized_vol",
        )
        assert out["iv_source"] == "contract_iv"
        assert out["sigma_used"] == pytest.approx(0.2473)


# ── Shortlist diversity ──────────────────────────────────────────────────────

class TestDiversify:
    """
    Pins the 2026-10-28 MSFT run, where the book contained all five advertised
    structures — protective put, direct OTM put, proxy put, collar (net debit
    $737.96) and bear put spread — and the top five shown to the user were

        59.1 Proxy Put / 50.8 Protective Put / 50.2 Direct Put
        48.5 Direct Put / 45.5 Direct Put

    three single-leg put variants and nothing else. Single-leg puts dominate
    the 35-point effectiveness term because their delta offset is uncapped, so
    the cost/protection trade-offs the product exists to compare were built,
    scored, and then trimmed away.
    """

    @staticmethod
    def _book():
        """The real pre-trim MSFT book, scores and strategies as observed."""
        rows = [
            (59.1, "Proxy Put"), (50.8, "Protective Put"),
            (50.2, "Direct Put (3% OTM)"), (48.5, "Direct Put (3% OTM)"),
            (45.5, "Direct Put (3% OTM)"), (45.5, "Collar"),
            (44.2, "Proxy Put (5% OTM)"), (44.2, "Protective Put"),
            (39.6, "Bear Put Spread"), (35.6, "Bear Put Spread"),
        ]
        return [_candidate(strategy=st, score=sc) for sc, st in rows]

    def test_every_built_strategy_reaches_the_shortlist(self):
        picked = OptionsSelector._diversify(self._book(), 5)
        families = {c.strategy.split(" (")[0] for c in picked}
        assert families == {"Proxy Put", "Protective Put", "Direct Put",
                            "Collar", "Bear Put Spread"}

    def test_the_best_candidate_overall_is_still_first(self):
        picked = OptionsSelector._diversify(self._book(), 5)
        assert picked[0].score == 59.1
        assert picked[0].strategy == "Proxy Put"

    def test_output_stays_sorted_by_score(self):
        picked = OptionsSelector._diversify(self._book(), 5)
        assert [c.score for c in picked] == sorted(
            (c.score for c in picked), reverse=True
        )

    def test_each_family_is_represented_by_its_own_best(self):
        """No candidate is promoted above a better one of the same structure."""
        picked = OptionsSelector._diversify(self._book(), 5)
        for c in picked:
            fam = c.strategy.split(" (")[0]
            best_in_fam = max(
                x.score for x in self._book()
                if x.strategy.split(" (")[0] == fam
            )
            assert c.score == best_in_fam

    def test_limit_is_respected(self):
        assert len(OptionsSelector._diversify(self._book(), 5)) == 5
        assert len(OptionsSelector._diversify(self._book(), 3)) == 3

    def test_slots_left_over_are_filled_by_score(self):
        """
        With only two families and five slots, the three spare slots go to the
        next-best candidates rather than being left empty.
        """
        book = [_candidate(strategy=st, score=sc) for sc, st in [
            (60.0, "Protective Put"), (55.0, "Protective Put"),
            (50.0, "Collar"), (45.0, "Protective Put"),
            (40.0, "Collar"), (35.0, "Protective Put"),
        ]]
        picked = OptionsSelector._diversify(book, 5)
        assert len(picked) == 5
        assert [c.score for c in picked] == [60.0, 55.0, 50.0, 45.0, 40.0]

    def test_short_book_is_returned_whole(self):
        book = self._book()[:3]
        assert OptionsSelector._diversify(book, 5) == book

    def test_empty_book_stays_empty(self):
        """No recommendation is invented to fill the shortlist."""
        assert OptionsSelector._diversify([], 5) == []


# ── Basis-risk scoring: measured vs never measured ───────────────────────────
#
# L3 stopped emitting 0.5 for an unmeasurable basis R² because, at a 15%
# scoring weight, 0.5 reads as a middling-but-real relationship and so scored
# an unmeasurable hedge above one honestly measured as weak. L10's
# `_basis_score` then mapped the 0.0 L3 emits straight back to 0.5,
# reinstating the placeholder one layer downstream. These tests pin the
# distinction, which only became expressible once `basis_risk_source` carried
# the sample's provenance out with the figure.

class TestBasisRiskIsScoredOnWhatWasMeasured:

    def test_an_unmeasured_basis_scores_zero_not_neutral(self):
        from backend.engines.layer_10_evaluator import _basis_score

        unmeasured = _candidate(basis_risk_r2=0.0, basis_risk_source="")
        assert _basis_score(unmeasured) == 0.0, (
            "an unmeasured basis must not score as though it were measured"
        )

    def test_a_measured_weak_basis_outscores_an_unmeasured_one(self):
        """
        The ordering the 0.5 placeholder inverted: measuring a hedge and
        finding it weak must rank it above a hedge nobody could measure.
        """
        from backend.engines.layer_10_evaluator import _basis_score

        measured_weak = _candidate(basis_risk_r2=0.083, basis_risk_source="correlation",
                                   basis_risk_n_obs=252)
        unmeasured    = _candidate(basis_risk_r2=0.0,   basis_risk_source="")
        assert _basis_score(measured_weak) > _basis_score(unmeasured)

    def test_a_measured_basis_scores_what_it_measured(self):
        from backend.engines.layer_10_evaluator import _basis_score

        c = _candidate(basis_risk_r2=0.083, basis_risk_source="correlation",
                       basis_risk_n_obs=252)
        assert _basis_score(c) == pytest.approx(0.083)

    def test_a_direct_hedge_scores_one_by_identity(self):
        from backend.engines.layer_10_evaluator import _basis_score

        c = _candidate(basis_risk_r2=1.0, basis_risk_source="identity")
        assert _basis_score(c) == 1.0

    def test_a_tail_correlation_basis_is_treated_as_measured(self):
        from backend.engines.layer_10_evaluator import _basis_score

        c = _candidate(basis_risk_r2=0.42, basis_risk_source="tail_correlation")
        assert _basis_score(c) == pytest.approx(0.42)

    def test_the_basis_score_is_clamped_to_a_valid_r2_range(self):
        from backend.engines.layer_10_evaluator import _basis_score

        assert _basis_score(_candidate(basis_risk_r2=1.4,  basis_risk_source="correlation")) == 1.0
        assert _basis_score(_candidate(basis_risk_r2=-0.2, basis_risk_source="correlation")) == 0.0


class TestBasisRiskProseMatchesTheMeasurement:

    def test_a_weak_measured_basis_is_never_listed_as_a_pro(self):
        """
        "Basis R²=0.08 — reasonable hedge tracking" was asserted at any R²,
        so a measurement showing the hedge barely tracks was presented to the
        user as a point in its favour.
        """
        from backend.instruments.options.selector import _basis_phrase

        assert "0.08" in _basis_phrase(True, 0.083)
        assert "not measured" in _basis_phrase(False, 0.0).lower()

    def test_an_unmeasured_basis_never_prints_a_figure(self):
        from backend.instruments.options.selector import _basis_phrase

        phrase = _basis_phrase(False, 0.0)
        assert "0.00" not in phrase and "0.50" not in phrase, phrase


class TestBasisProvenanceIsAlwaysDeclared:

    def test_a_cross_hedge_row_declares_its_correlation_sample(self):
        from backend.instruments.options.selector import _basis_provenance

        prov = _basis_provenance("cross_hedge", {"n_obs": 252, "window_days": 365,
                                                 "corr_stderr": 0.061})
        assert prov["basis_risk_source"] == "correlation"
        assert prov["basis_risk_n_obs"] == 252
        assert prov["basis_risk_stderr"] == 0.061

    def test_a_direct_hedge_declares_identity_rather_than_a_sample(self):
        from backend.instruments.options.selector import _basis_provenance

        prov = _basis_provenance("direct_hedge", None)
        assert prov["basis_risk_source"] == "identity"
        assert "basis_risk_n_obs" not in prov

    def test_a_cross_hedge_with_no_stats_claims_nothing(self):
        """An empty source is what makes L10 score it as unmeasured."""
        from backend.instruments.options.selector import _basis_provenance

        assert _basis_provenance("cross_hedge", None) == {}


# ── R2: an absent market is not a closed market ───────────────────────────────

class TestUnquotableHedgesAreWithheldOnlyWhenTheVenueIsQuoting:
    """
    The selector discloses a theoretical premium when it finds nothing
    quotable. Whether that is the right answer depends on why nothing was
    quotable, which only a run-level view can tell:

      * nothing quoted anywhere  → the venue is not quoting; disclose
      * quotes exist elsewhere   → this contract has no market; withhold

    Before this fix both cases produced a ranked, sized recommendation
    carrying a model premium, so a user could size a position in a contract
    nobody would fill.
    """

    @staticmethod
    def _rec(ticker, candidates):
        from backend.models.hedge_models import HedgeRecommendation
        return HedgeRecommendation(rank=1, asset_ticker=ticker, candidates=candidates)

    @staticmethod
    def _refuse(recs):
        warnings: list = []
        InstrumentSelectionEngine._refuse_unquotable_hedges(recs, warnings)
        return warnings

    def test_an_unquoted_holding_is_withheld_when_another_holding_is_quoted(self):
        quoted = _candidate(ticker="SPY", premium_source="market_mid",
                            bid_ask_spread_pct=0.01)
        modelled = _candidate(ticker="ILLQ", premium_source="model")
        recs = [self._rec("SPY", [quoted]), self._rec("ILLQ", [modelled])]

        warnings = self._refuse(recs)

        assert [c.ticker for c in recs[0].candidates] == ["SPY"]
        assert recs[1].candidates == [], (
            "a contract with no two-sided quote, while the venue is quoting, "
            "is not a hedge that can be placed"
        )
        assert any("ILLQ" in w and "no hedge is recommended" in w for w in warnings)

    def test_the_warning_distinguishes_an_absent_market_from_a_closed_one(self):
        recs = [
            self._rec("SPY", [_candidate(ticker="SPY", premium_source="market_mid",
                                         bid_ask_spread_pct=0.01)]),
            self._rec("ILLQ", [_candidate(ticker="ILLQ", premium_source="model")]),
        ]
        warnings = self._refuse(recs)
        text = " ".join(warnings)
        # The user has to be able to tell these apart: one means "come back at
        # 09:30", the other means "this cannot be hedged this way at all".
        assert "absent market" in text and "not a closed one" in text
        assert "shown unhedged" in text

    def test_nothing_quoted_anywhere_leaves_every_disclosure_in_place(self):
        a = _candidate(ticker="SPY", premium_source="model")
        b = _candidate(ticker="QQQ", premium_source="model")
        recs = [self._rec("SPY", [a]), self._rec("QQQ", [b])]

        warnings = self._refuse(recs)

        assert recs[0].candidates == [a] and recs[1].candidates == [b], (
            "out of hours no instrument is quoted; withholding everything "
            "would make the product unusable rather than more honest"
        )
        assert warnings == []

    def test_a_quote_too_wide_to_trade_does_not_count_as_quoting(self):
        # 0.05/3.00 — "quoted", but the mid is not a fill. This is the hole a
        # plain `premium_source == "market_mid"` test would leave open: the
        # holding looks quoted and survives, and the venue looks open to
        # every other holding in the run.
        wide = _candidate(ticker="LQD", premium_source="market_mid",
                          bid_ask_spread_pct=1.98)
        tight = _candidate(ticker="SPY", premium_source="market_mid",
                           bid_ask_spread_pct=0.01)
        recs = [self._rec("SPY", [tight]), self._rec("LQD", [wide])]

        warnings = self._refuse(recs)

        assert recs[1].candidates == []
        assert any("LQD" in w for w in warnings)

    def test_a_wide_quote_alone_does_not_make_the_venue_look_open(self):
        wide = _candidate(ticker="LQD", premium_source="market_mid",
                          bid_ask_spread_pct=1.98)
        modelled = _candidate(ticker="ILLQ", premium_source="model")
        recs = [self._rec("LQD", [wide]), self._rec("ILLQ", [modelled])]

        warnings = self._refuse(recs)

        # Nothing executable anywhere, so no refusal may fire off the back of
        # an untradeable quote.
        assert recs[0].candidates == [wide]
        assert recs[1].candidates == [modelled]
        assert warnings == []

    def test_a_holding_with_one_quoted_candidate_keeps_its_whole_set(self):
        quoted = _candidate(ticker="SPY", strike=100.0, premium_source="market_mid",
                            bid_ask_spread_pct=0.01)
        modelled = _candidate(ticker="SPY", strike=95.0, premium_source="model")
        recs = [self._rec("SPY", [quoted, modelled])]

        warnings = self._refuse(recs)

        assert len(recs[0].candidates) == 2, (
            "the refusal is per holding, not per contract — the selector "
            "already ranks within a holding"
        )
        assert warnings == []

    def test_an_already_empty_holding_is_not_warned_about_twice(self):
        recs = [
            self._rec("SPY", [_candidate(ticker="SPY", premium_source="market_mid",
                                         bid_ask_spread_pct=0.01)]),
            self._rec("TLT", []),      # already unhedged for its own reason
        ]
        warnings = self._refuse(recs)
        assert warnings == [], "L4 and L7 would otherwise both claim this holding"

    def test_the_withholding_is_recorded_as_a_data_quality_event(self):
        recs = [
            self._rec("SPY", [_candidate(ticker="SPY", premium_source="market_mid",
                                         bid_ask_spread_pct=0.01)]),
            self._rec("ILLQ", [_candidate(ticker="ILLQ", premium_source="model")]),
        ]
        with dq.capture() as log:
            self._refuse(recs)
            degraded = {p.name: p for p in log.degradations}

        assert "no_market:ILLQ" in degraded, (
            "a withheld hedge is a data-availability fact and belongs in the "
            "disclosure report, not only in the log"
        )
        assert degraded["no_market:ILLQ"].source == dq.SOURCE_UNAVAILABLE


class TestExecutablyQuotedPredicate:
    def test_a_model_premium_is_never_executable(self):
        assert not is_executably_quoted(_candidate(premium_source="model",
                                                   bid_ask_spread_pct=0.01))

    def test_a_tight_quote_is_executable(self):
        assert is_executably_quoted(_candidate(premium_source="market_mid",
                                               bid_ask_spread_pct=0.01))

    def test_an_unmeasured_spread_on_a_real_quote_is_allowed(self):
        # Absence of a spread measurement is not evidence of a bad spread;
        # `_filter_executable` has always treated it this way.
        assert is_executably_quoted(_candidate(premium_source="market_mid",
                                               bid_ask_spread_pct=None))

    def test_the_limit_is_inclusive(self):
        assert is_executably_quoted(
            _candidate(premium_source="market_mid",
                       bid_ask_spread_pct=MAX_EXECUTABLE_SPREAD_PCT))
        assert not is_executably_quoted(
            _candidate(premium_source="market_mid",
                       bid_ask_spread_pct=MAX_EXECUTABLE_SPREAD_PCT + 0.001))
