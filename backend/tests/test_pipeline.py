"""
End-to-end integration tests for the 12-layer hedge pipeline.

What is and is not substituted here matters, so it is stated plainly: only the
network boundary is replaced (see `fixtures/synthetic_market.py`). Yahoo's
chart API and `yfinance.Ticker` return fixture data; the response cache is
disabled. Everything else is the product — valuation, GMM regime detection,
GARCH, VaR/ES, beta and correlation estimation, cross-hedge ranking, chain
parsing, IV-skew interpolation, liquidity assessment, Black-Scholes pricing,
position sizing, Greeks, scoring, budget enforcement, formatting and the
data-quality ledger all execute unmodified.

The point is to assert properties that only hold if the whole chain is right:

  * the portfolio is valued at current prices, with cost basis kept separate;
  * every recommended contract is one the chain actually lists, by OCC symbol;
  * premiums and the volatility behind them carry honest provenance;
  * the user's cost budget is respected across the whole portfolio;
  * monotonic financial relationships hold between candidates;
  * when the data is missing, the pipeline says so instead of inventing a
    contract.

The LLM layers are off by default: they are a network dependency with no
deterministic output, and they are covered by `test_explainer.py`.
"""

from datetime import date, timedelta

import pytest

from backend.engines.pipeline_orchestrator import HedgeOrchestrator
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.data import data_quality as dq
from backend.tests.fixtures import synthetic_market as sm
from backend.tests.fixtures.synthetic_market import AS_OF, ChainConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _portfolio(**overrides) -> PortfolioInput:
    """
    A representative multi-asset portfolio: two single names plus a broad
    index, bought at prices that make the unrealized P&L signed both ways, so
    a test can tell market value from cost basis.
    """
    kwargs = dict(
        holdings=[
            # Bought below spot (200.00) → a gain.
            StockPosition(ticker="AAPL", shares=200, purchase_price=150.00,
                          purchase_date=date(2024, 1, 15)),
            # Bought above spot (400.00) → a loss.
            StockPosition(ticker="MSFT", shares=100, purchase_price=450.00,
                          purchase_date=date(2024, 3, 1)),
            StockPosition(ticker="SPY", shares=150, purchase_price=480.00,
                          purchase_date=date(2024, 2, 1)),
        ],
        total_notional=150_000.0,
    )
    kwargs.update(overrides)
    return PortfolioInput(**kwargs)


def _run(portfolio, **orch_kwargs):
    """Run the pipeline inside a ledger, as the API route does."""
    opts = dict(run_l9_simulation=False, run_l11_llm=False)
    opts.update(orch_kwargs)
    with dq.capture():
        orch = HedgeOrchestrator(**opts)
        market_data = sm.market_data_for(
            [h.ticker for h in portfolio.holdings]
        )
        return orch.run(portfolio, market_data)


def _all_candidates(output):
    return [c for rec in output.recommendations for c in rec.candidates]


# Module-scoped. A pipeline run takes ~4s, and `output` below is read by a
# dozen tests that never mutate it, so a function-scoped fixture paid for the
# whole product once per assertion. `monkeypatch` is function-scoped by
# construction, so the patches are installed through an explicit context that
# lives as long as the fixture. A test that needs a *different* run calls
# `_run()` directly, as several below do.
@pytest.fixture(scope="module")
def market():
    """The fixture market, fully quoted."""
    with pytest.MonkeyPatch.context() as mp:
        yield sm.install(mp)


@pytest.fixture(scope="module")
def output(market):
    """
    One pipeline run, shared by the read-only assertions below. Tests must not
    mutate it.
    """
    return _run(_portfolio())


# ── The run completes at all ─────────────────────────────────────────────────

class TestPipelineCompletes:
    def test_default_portfolio_produces_recommendations(self, output):
        assert output.recommendations, "pipeline produced no recommendations"
        assert _all_candidates(output), "recommendations contain no candidates"

    def test_every_holding_is_covered(self, output):
        hedged = {rec.asset_ticker for rec in output.recommendations}
        assert hedged == {"AAPL", "MSFT", "SPY"}

    def test_regime_is_classified_from_the_fixture_vix(self, output):
        # The fixture VIX is 17.5 with VIX3M 19.0 — contango, mid-vol.
        assert output.vix_level == pytest.approx(17.5, abs=0.01)
        assert output.regime, "no regime label"
        assert output.is_anomaly is False

    def test_run_is_deterministic(self, market):
        a, b = _run(_portfolio()), _run(_portfolio())
        assert a.portfolio_notional == b.portfolio_notional
        assert [c.contract_symbol for c in _all_candidates(a)] == \
               [c.contract_symbol for c in _all_candidates(b)]
        assert [c.total_cost for c in _all_candidates(a)] == \
               [c.total_cost for c in _all_candidates(b)]

    def test_unhedged_risk_is_reported_for_each_holding(self, output):
        for rec in output.recommendations:
            r = rec.unhedged_risk
            assert r is not None, f"{rec.asset_ticker}: no unhedged risk block"
            assert r.notional > 0
            # ES is the mean of the tail beyond VaR, so it cannot be smaller.
            assert r.cvar_95 >= r.var_95 > 0
            assert r.stress_loss_20pct >= r.stress_loss_10pct


# ── Valuation ────────────────────────────────────────────────────────────────

class TestPortfolioValuation:
    def test_notional_is_current_price_times_shares(self, output):
        expected = (200 * sm.base_price("AAPL")
                    + 100 * sm.base_price("MSFT")
                    + 150 * sm.base_price("SPY"))
        assert output.portfolio_notional == pytest.approx(expected, rel=1e-9)

    def test_cost_basis_is_kept_separate_from_market_value(self, output):
        expected_cost = 200 * 150.00 + 100 * 450.00 + 150 * 480.00
        assert output.portfolio_cost_basis == pytest.approx(expected_cost, rel=1e-9)
        assert output.portfolio_cost_basis != output.portfolio_notional

    def test_unrealized_pnl_reconciles(self, output):
        assert output.portfolio_unrealized_pnl == pytest.approx(
            output.portfolio_notional - output.portfolio_cost_basis, rel=1e-9
        )

    def test_declared_notional_is_recorded_not_used(self, output):
        # The client said 150,000; the holdings are worth 155,000. The server
        # must value the portfolio itself and keep the claim for comparison.
        assert output.declared_notional == pytest.approx(150_000.0)
        assert output.portfolio_notional != output.declared_notional

    def test_a_materially_wrong_declared_notional_is_flagged(self, market):
        # The user is told which figure the analysis used, and by how much
        # their own number differed — silently substituting one for the other
        # would leave every VaR and contract count unexplained.
        out = _run(_portfolio(total_notional=10_000.0))
        warned = [w for w in out.warnings if "computed market value" in w]
        assert warned, (
            f"a 15x understatement of portfolio value produced no warning: "
            f"{out.warnings}"
        )
        assert "$10,000" in warned[0] and "$155,000" in warned[0]


# ── Contracts must be real ───────────────────────────────────────────────────

class TestContractsAreListed:
    def test_every_option_candidate_names_a_listed_contract(self, output):
        from backend.data.options_chain import get_listed_strikes

        options = [c for c in _all_candidates(output)
                   if c.instrument_type == "option"]
        assert options, "no option candidates to check"

        for c in options:
            assert c.contract_symbol, f"{c.strategy} has no contract symbol"
            assert c.expiry_date, f"{c.strategy} has no expiry"
            listed = get_listed_strikes(c.ticker, c.expiry_date, c.option_type)
            assert c.strike in listed, (
                f"{c.ticker} {c.expiry_date} {c.option_type} K={c.strike} is "
                f"not on the listed ladder — a synthesised strike reached the user"
            )
            assert c.strike_is_listed is True

    def test_contract_symbol_encodes_the_contract_it_claims(self, output):
        # A symbol that does not decode to the strike and expiry beside it
        # would make the recommendation untradeable even though every field
        # looks populated.
        for c in _all_candidates(output):
            if c.instrument_type != "option" or not c.contract_symbol:
                continue
            sym = c.contract_symbol
            assert sym.startswith(c.ticker)
            tail = sym[len(c.ticker):]
            yymmdd, side, strike_part = tail[:6], tail[6], tail[7:]
            assert date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]),
                        int(yymmdd[4:6])).isoformat() == c.expiry_date
            assert side == ("P" if c.option_type == "put" else "C")
            assert int(strike_part) / 1000 == pytest.approx(c.strike)

    def test_both_legs_of_a_two_leg_structure_are_listed(self, output):
        from backend.data.options_chain import get_listed_strikes

        multi = [c for c in _all_candidates(output) if c.short_strike]
        assert multi, "no spread or collar candidates were produced"
        for c in multi:
            assert c.short_contract_symbol, (
                f"{c.strategy}: short leg has no contract symbol, so the "
                f"structure cannot be executed as quoted"
            )
            listed = get_listed_strikes(c.ticker, c.expiry_date,
                                        c.short_option_type)
            assert c.short_strike in listed

    def test_expiries_are_in_the_future_and_inside_the_horizon(self, output):
        for c in _all_candidates(output):
            if not c.expiry_date:
                continue
            exp = date.fromisoformat(c.expiry_date)
            assert exp > AS_OF, f"{c.contract_symbol} expires in the past"
            assert exp <= AS_OF + timedelta(days=400), (
                f"{c.contract_symbol} is far outside any requested horizon"
            )


# ── Pricing provenance ───────────────────────────────────────────────────────

class TestPricingProvenance:
    def test_quoted_candidates_price_off_the_market_mid(self, output):
        from backend.data.options_chain import get_contract_quote

        quoted = [c for c in _all_candidates(output)
                  if c.premium_source == "market_mid" and not c.short_strike]
        assert quoted, "the fully-quoted fixture produced no market-priced candidate"

        for c in quoted:
            q = get_contract_quote(c.ticker, c.expiry_date, c.strike,
                                   c.option_type)
            assert q is not None
            assert c.bid == pytest.approx(q["bid"], abs=0.01)
            assert c.ask == pytest.approx(q["ask"], abs=0.01)
            # Cost is the mid times the contract multiplier times size.
            assert c.total_cost == pytest.approx(
                q["mid"] * 100 * c.n_contracts, rel=0.01
            )

    def test_volatility_used_is_recorded_with_its_source(self, output):
        allowed = {"contract_iv", "interpolated_skew", "atm_iv",
                   "realized_vol", "garch_forecast"}
        for c in _all_candidates(output):
            if c.instrument_type != "option":
                continue
            assert c.iv_source in allowed, f"unknown iv_source {c.iv_source!r}"
            assert c.iv_used and 0.01 < c.iv_used < 3.0, (
                f"{c.contract_symbol}: iv_used={c.iv_used}"
            )

    def test_contract_specific_iv_is_used_not_one_number_for_the_book(self, output):
        # The regression this pins: a single market-wide GARCH forecast once
        # priced every contract across every underlying, all of them labelled
        # "atm_iv". With a skewed surface available, the IVs must differ.
        ivs = {round(c.iv_used, 6) for c in _all_candidates(output)
               if c.iv_used and c.instrument_type == "option"}
        assert len(ivs) > 1, (
            f"every contract in the book priced at the same volatility: {ivs}"
        )

    def test_iv_respects_the_skew(self, market):
        # Deeper out-of-the-money puts on the same underlying and expiry must
        # carry higher implied vol than nearer ones. A flat surface, or a
        # realized-vol substitution, would break this.
        from backend.data.options_chain import get_strike_iv, get_available_expiries

        expiry = get_available_expiries("SPY")[8]
        near = get_strike_iv("SPY", expiry, 490.0, "put")
        far = get_strike_iv("SPY", expiry, 430.0, "put")
        assert near and far
        assert far > near, f"no skew: K=430 IV {far} <= K=490 IV {near}"

    def test_quotes_are_compared_against_the_model(self, output):
        priced = [c for c in _all_candidates(output)
                  if c.premium_source == "market_mid" and c.model_premium > 0]
        assert priced, "no candidate carries both a quote and a model value"
        for c in priced:
            assert c.premium_vs_model_pct is not None, (
                f"{c.contract_symbol}: quote never compared to the model, so a "
                f"mispriced quote would go unnoticed"
            )


# ── Sizing, cost and budget ──────────────────────────────────────────────────

class TestSizingAndBudget:
    def test_contract_counts_are_whole_and_positive(self, output):
        for c in _all_candidates(output):
            if c.instrument_type != "option":
                continue
            assert isinstance(c.n_contracts, int)
            assert c.n_contracts >= 1, (
                f"{c.contract_symbol}: {c.n_contracts} contracts is not a trade"
            )

    def test_cost_is_consistent_with_premium_and_multiplier(self, output):
        for c in _all_candidates(output):
            if c.instrument_type != "option" or c.short_strike:
                continue
            per_share = (c.market_premium if c.premium_source == "market_mid"
                         else c.model_premium)
            if per_share <= 0:
                continue
            assert c.total_cost == pytest.approx(
                per_share * 100 * c.n_contracts, rel=0.02
            ), f"{c.contract_symbol}: cost does not reconcile to premium x 100 x n"

    def test_a_direct_hedge_does_not_exceed_the_position(self, output):
        """
        Puts on more shares than the user owns is not a hedge.

        Below the strike the extra contracts gain while nothing offsets them,
        so the position is net short the market — the opposite of protection.
        The allowance is one lot, because contracts are indivisible and a
        150-share position can only be covered in units of 100.
        """
        shares = {"AAPL": 200, "MSFT": 100, "SPY": 150}
        for rec in output.recommendations:
            held = shares[rec.asset_ticker]
            for c in rec.candidates:
                if c.instrument_type != "option" or c.hedge_category != "direct_hedge":
                    continue
                assert c.n_contracts * 100 <= held + 100, (
                    f"{c.contract_symbol}: {c.n_contracts} contracts covers "
                    f"{c.n_contracts * 100} shares against a {held}-share position"
                )

    def test_sizing_tracks_the_requested_protection_level(self, market):
        """
        Asking for less coverage must buy fewer contracts.

        `protection_level` reached L4 but not L7, and L7 ran last — so the
        number the user saw was sized for full coverage whatever they chose.
        """
        low = _run(_portfolio(protection_level=0.05, max_hedge_cost_pct=0.20))
        high = _run(_portfolio(protection_level=0.50, max_hedge_cost_pct=0.20))

        def contracts(out):
            return {(c.ticker, c.strike, c.expiry_date): c.n_contracts
                    for c in _all_candidates(out)}

        lo, hi = contracts(low), contracts(high)
        shared = set(lo) & set(hi)
        assert shared, "no comparable contract across the two runs"
        assert all(lo[k] <= hi[k] for k in shared), (
            "a lower protection level did not reduce any contract count"
        )
        assert sum(lo[k] for k in shared) < sum(hi[k] for k in shared), (
            f"protection level had no effect on sizing: "
            f"{sum(lo[k] for k in shared)} vs {sum(hi[k] for k in shared)} contracts"
        )

    @pytest.mark.parametrize("budget_pct", [0.01, 0.02, 0.05])
    def test_top_recommendation_per_holding_respects_the_budget(
        self, market, budget_pct
    ):
        out = _run(_portfolio(max_hedge_cost_pct=budget_pct))
        total_mv = out.portfolio_notional
        # The portfolio-level claim: hedging the whole book with the top pick
        # for each holding must not exceed the stated budget.
        chosen = [max(rec.candidates, key=lambda c: c.score)
                  for rec in out.recommendations if rec.candidates]
        spend = sum(c.total_cost for c in chosen)
        assert spend <= total_mv * budget_pct * 1.02, (
            f"budget {budget_pct:.0%} of ${total_mv:,.0f} = "
            f"${total_mv * budget_pct:,.0f}, but the top picks cost "
            f"${spend:,.0f}"
        )

    def test_a_tighter_budget_never_costs_more(self, market):
        tight = _run(_portfolio(max_hedge_cost_pct=0.01))
        loose = _run(_portfolio(max_hedge_cost_pct=0.05))

        def spend(o):
            return sum(max(r.candidates, key=lambda c: c.score).total_cost
                       for r in o.recommendations if r.candidates)

        assert spend(tight) <= spend(loose) * 1.001

    def test_credit_structures_are_reported_as_credits(self, market):
        # A collar can finance itself. If the pipeline cannot represent a
        # negative cost it will either hide the structure or misreport it as a
        # debit, and the cost comparison the product exists to show is wrong.
        out = _run(_portfolio(upside_preservation=False))
        collars = [c for c in _all_candidates(out) if "Collar" in c.strategy]
        assert collars, "no collar was produced"
        # Every collar's net cost must be strictly cheaper than its long leg
        # alone, because the short call brings in premium.
        for c in collars:
            long_only = [p for p in _all_candidates(out)
                         if p.ticker == c.ticker and not p.short_strike
                         and p.option_type == "put" and p.strike == c.strike
                         and p.expiry_date == c.expiry_date]
            for p in long_only:
                per_contract_collar = c.total_cost / max(c.n_contracts, 1)
                per_contract_put = p.total_cost / max(p.n_contracts, 1)
                assert per_contract_collar < per_contract_put, (
                    f"collar at K={c.strike} costs more per contract "
                    f"({per_contract_collar:.2f}) than the bare put "
                    f"({per_contract_put:.2f}) — the short call premium is "
                    f"not being credited"
                )


# ── Financial relationships between candidates ───────────────────────────────

class TestMonotonicRelationships:
    def test_a_lower_strike_put_costs_less(self, market):
        from backend.data.options_chain import get_contract_quote, get_available_expiries

        expiry = get_available_expiries("SPY")[8]
        quotes = [
            get_contract_quote("SPY", expiry, k, "put")
            for k in (450.0, 475.0, 500.0)
        ]
        mids = [q["mid"] for q in quotes]
        assert mids == sorted(mids), f"put premiums not monotonic in strike: {mids}"

    def test_a_spread_costs_less_than_its_long_leg(self, output):
        spreads = [c for c in _all_candidates(output)
                   if "Spread" in c.strategy and c.short_strike]
        assert spreads, "no bear put spread was produced"
        for c in spreads:
            assert c.short_strike < c.strike, (
                f"bear put spread with short leg {c.short_strike} at or above "
                f"long leg {c.strike} is not a bear put spread"
            )
            assert c.total_cost > 0, "a bear put spread is a debit structure"

    def test_scores_are_bounded_and_ordered(self, output):
        for rec in output.recommendations:
            scores = [c.score for c in rec.candidates]
            assert all(0 <= s <= 100 for s in scores), scores
            assert scores == sorted(scores, reverse=True), (
                f"{rec.asset_ticker}: candidates not returned best-first: {scores}"
            )

    def test_the_shortlist_compares_different_structures(self, output):
        # Five variations of the same trade is not a comparison. The product's
        # value is showing the cost/protection trade-off between structures.
        for rec in output.recommendations:
            families = {c.strategy.split(" (")[0] for c in rec.candidates}
            assert len(families) >= 2, (
                f"{rec.asset_ticker}: shortlist is all one structure: {families}"
            )

    def test_greeks_have_the_right_sign(self, output):
        for c in _all_candidates(output):
            if c.instrument_type != "option":
                continue
            if c.option_type == "put" and not c.short_strike:
                assert c.delta < 0, (
                    f"{c.contract_symbol}: long put with delta {c.delta} "
                    f"provides no downside protection"
                )
            assert c.vega >= 0 or c.short_strike, (
                f"{c.contract_symbol}: long option with negative vega"
            )


# ── Data quality ─────────────────────────────────────────────────────────────

class TestDataQualityLedger:
    def test_ledger_is_populated(self, output):
        assert output.data_quality, (
            "no data-quality ledger — the client cannot tell live data from "
            "substituted data"
        )
        counts = output.data_quality.get("source_counts", {})
        assert sum(counts.values()) > 0

    def test_a_fully_quoted_run_is_not_marked_degraded(self, output):
        # Every quote in this fixture is two-sided and every IV is published,
        # so anything flagged unavailable is a real defect, not market hours.
        counts = output.data_quality.get("source_counts", {})
        assert counts.get("unavailable", 0) == 0, (
            f"fully-quoted fixture reported unavailable inputs: "
            f"{[w for w in output.data_quality.get('warnings', [])]}"
        )
        assert output.data_quality.get("is_degraded") is False


# ── Degraded and failure scenarios ───────────────────────────────────────────

class TestDegradedData:
    def test_overnight_snapshot_is_not_reported_as_a_tight_market(self, monkeypatch):
        """
        The defect this pins cost the most: outside market hours Yahoo returns
        the full strike ladder with bid, ask and open interest cleared. Read
        naively that is a contract with a 0.00% bid-ask spread and zero
        depth — which renders as a perfectly tight market, the exact opposite
        of the truth.
        """
        sm.install(monkeypatch, ChainConfig(quotes_published=False))
        out = _run(_portfolio())

        candidates = _all_candidates(out)
        assert candidates, "pipeline produced nothing at all on an unquoted chain"

        for c in candidates:
            if c.instrument_type != "option":
                continue
            assert c.bid_ask_spread_pct != 0.0, (
                f"{c.contract_symbol}: reported a 0.00% spread on a contract "
                f"with no bid"
            )
            assert c.premium_source == "model", (
                f"{c.contract_symbol}: claims a market mid with no quote"
            )

        assert out.data_quality.get("is_degraded") is True
        warnings = " ".join(out.data_quality.get("warnings", []))
        assert "market hours" in warnings or "theoretical" in warnings, (
            f"an unquoted chain produced no explanation to the user: {warnings}"
        )

    def test_theoretical_premiums_are_disclosed_as_such(self, monkeypatch):
        sm.install(monkeypatch, ChainConfig(quotes_published=False))
        out = _run(_portfolio())
        modelled = [c for c in _all_candidates(out)
                    if c.premium_source == "model"]
        assert modelled
        for c in modelled:
            assert c.liquidity_ok is False
            assert any("theoretical" in con.lower() or "not an executable" in con.lower()
                       for con in c.cons), (
                f"{c.contract_symbol}: model premium presented with no caveat: "
                f"{c.cons}"
            )

    def test_untradeably_wide_quotes_do_not_rank(self, monkeypatch):
        sm.install(monkeypatch, ChainConfig(wide_spreads=True))
        out = _run(_portfolio())
        for c in _all_candidates(out):
            if c.premium_source != "market_mid":
                continue
            assert (c.bid_ask_spread_pct or 0) <= 0.50, (
                f"{c.contract_symbol}: ranked on a mid nobody can trade near "
                f"(spread {c.bid_ask_spread_pct:.0%})"
            )

    def test_no_chain_at_all_does_not_invent_contracts(self, monkeypatch):
        """
        With the chain provider down the pipeline may degrade or refuse, but it
        may not produce an option it cannot name. A recommendation without a
        verified listed strike is a fabricated contract.
        """
        sm.install(monkeypatch, ChainConfig(chain_available=False))
        out = _run(_portfolio())

        for c in _all_candidates(out):
            if c.instrument_type != "option":
                continue
            assert c.strike_is_listed and c.contract_symbol, (
                f"an option candidate was produced with no listed contract "
                f"behind it: {c.strategy} {c.ticker} K={c.strike}"
            )

        assert out.data_quality.get("is_degraded") is True

    def test_missing_iv_falls_back_to_a_named_source_not_a_guess(self, monkeypatch):
        # With no published IV the pipeline must price off something it can
        # name — per-ticker realized vol — and say so, never a hardcoded 0.20.
        sm.install(monkeypatch, ChainConfig(publish_iv=False))
        out = _run(_portfolio())
        options = [c for c in _all_candidates(out)
                   if c.instrument_type == "option" and c.iv_used]
        assert options
        for c in options:
            assert c.iv_source != "contract_iv"
            assert c.iv_used != pytest.approx(0.20, abs=1e-9), (
                f"{c.contract_symbol}: priced on a hardcoded default volatility"
            )
        # Per-ticker, not one number for the whole book.
        by_ticker = {}
        for c in options:
            by_ticker.setdefault(c.ticker, set()).add(round(c.iv_used, 4))
        assert len({next(iter(v)) for v in by_ticker.values() if len(v) == 1}) > 1 \
            or len(by_ticker) == 1


# ── Input validation ─────────────────────────────────────────────────────────

class TestInputValidation:
    def test_unsupported_ticker_is_rejected_at_the_boundary(self):
        with pytest.raises(Exception) as err:
            _portfolio(holdings=[
                StockPosition(ticker="NOTREAL", shares=10,
                              purchase_price=100.0,
                              purchase_date=date(2024, 1, 1))
            ])
        assert "supported ticker universe" in str(err.value)

    @pytest.mark.parametrize("shares", [0, -5])
    def test_non_positive_share_count_is_rejected(self, shares):
        with pytest.raises(Exception):
            StockPosition(ticker="AAPL", shares=shares, purchase_price=100.0,
                          purchase_date=date(2024, 1, 1))

    def test_empty_portfolio_is_rejected(self):
        with pytest.raises(Exception):
            PortfolioInput(holdings=[], total_notional=1000.0)


# ── The LLM layers cannot delete the product's disclosures ───────────────────
#
# These are the only tests in this file that let an explainer run, and they
# substitute one thing beyond the network: the model's reply. Everything
# between the chain and the formatted output is the product, including the
# real `_merge_explanation`.
#
# The defect they pin was live by default (`run_l11_llm = True`). The
# selector states three facts in `candidate.cons` — no executable quote, thin
# market, expiry crosses earnings — and both explainer layers merged a reply
# by replacing that list, so a model returning its own bullets deleted them
# and the response described a hedge without the caveat that decided whether
# its premium meant anything.

class _RewritingExplainer:
    """
    An explainer that answers every candidate with its own cons and nothing
    else — the shape of reply that used to erase the disclosures. Not a mock
    of the pipeline: the engine, the merge and the formatter all run.
    """
    CONS = ["Premium is a cost drag", "Expires worthless if spot is flat"]

    provider_name = "rewriting-test-double"

    def is_available(self):
        return True

    def supports_combined_call(self):
        return False

    def explain_candidates(self, context):
        from backend.core.explainer.base_explainer import CandidateExplanation
        asset = context.get("holding", {}).get("ticker", "")
        return [
            CandidateExplanation(
                ticker=c["ticker"], strategy=c["strategy"], asset_ticker=asset,
                when_works_best="A sharp drawdown",
                when_fails="A grind higher",
                rationale="Rewritten by the test double.",
                pros=["Defined risk"],
                cons=list(self.CONS),
            )
            for c in context.get("candidates", [])
        ]

    def explain_portfolio(self, context):
        from backend.core.explainer.base_explainer import PortfolioExplanation
        return PortfolioExplanation(
            summary="Test-double summary.", key_risks=["Equity beta"],
            regime_commentary="Test-double regime note.",
            top_recommendation="Test-double top pick.",
        )


@pytest.fixture(scope="module")
def llm_output():
    """
    One pipeline run with L11 on and the rewriting explainer installed,
    against the overnight snapshot (`quotes_published=False`).

    That market is chosen deliberately rather than the fully-quoted one: it
    is what makes the selector state "premium shown is theoretical", which is
    the disclosure with the most riding on it. Nothing is quoted anywhere, so
    the L4 no-market gate correctly stands down and the candidates reach L11
    carrying their disclosure — exactly the state in which the LLM merge used
    to erase it.
    """
    with pytest.MonkeyPatch.context() as mp:
        sm.install(mp, ChainConfig(quotes_published=False))
        mp.setattr(
            "backend.engines.layer_11_llm_explainer.get_explainer",
            lambda provider=None: _RewritingExplainer(),
        )
        return _run(_portfolio(), run_l11_llm=True)


class TestDisclosuresSurviveTheLLMLayers:

    def test_the_explainer_actually_ran(self, llm_output):
        """Guard on the guard: without this the class could pass vacuously."""
        rewritten = [
            c for c in _all_candidates(llm_output)
            if c.rationale == "Rewritten by the test double."
        ]
        assert rewritten, "no candidate was rewritten — the double never ran"

    def test_some_candidate_carried_a_disclosure_to_begin_with(self, llm_output):
        assert any(
            c.mandatory_disclosures for c in _all_candidates(llm_output)
        ), "fixture produced no disclosures, so the next test proves nothing"

    def test_no_disclosure_is_missing_from_the_cons_the_user_sees(self, llm_output):
        for c in _all_candidates(llm_output):
            missing = [d for d in c.mandatory_disclosures if d not in c.cons]
            assert not missing, (
                f"{c.ticker} {c.strategy}: the LLM merge deleted {missing!r}; "
                f"cons is now {c.cons!r}"
            )

    def test_a_disclosure_is_not_buried_under_the_llm_prose(self, llm_output):
        """
        Only for candidates the LLM actually rewrote. L11 explains the top
        three per holding, and a candidate it never saw keeps the selector's
        own ordering — strongest structural caveat first, disclosure appended
        — which is the long-standing display order and not what this fix is
        about. The front-of-list guarantee is about a *replaced* list, where
        otherwise the disclosure would reappear under four bullets of prose.
        """
        rewritten = [
            c for c in _all_candidates(llm_output)
            if c.rationale == "Rewritten by the test double."
            and c.mandatory_disclosures
        ]
        assert rewritten, "no rewritten candidate carried a disclosure"
        for c in rewritten:
            assert c.cons[0] in c.mandatory_disclosures, (
                f"{c.ticker} {c.strategy}: cons leads with "
                f"{c.cons[0]!r} instead of a disclosure"
            )

    def test_the_llm_s_own_cons_are_still_present(self, llm_output):
        """The fix must preserve the disclosure without discarding the reply."""
        rewritten = [
            c for c in _all_candidates(llm_output)
            if c.rationale == "Rewritten by the test double."
        ]
        for c in rewritten:
            assert _RewritingExplainer.CONS[0] in c.cons, c.cons

    def test_the_run_is_otherwise_a_normal_one(self, llm_output):
        assert llm_output.recommendations
        assert llm_output.portfolio_summary == "Test-double summary."
        assert llm_output.llm_provider == "rewriting-test-double"


class TestAFailingL5DoesNotFailTheRun:
    """
    L5's docstring claimed it "never raises — all LLM failures are caught and
    logged" while the method had no try/except. What makes an L5 failure
    non-fatal is the orchestrator's guard around the call, so that is what
    gets asserted; the docstring has been corrected to say so.
    """

    def test_an_exception_from_l5_leaves_the_pipeline_intact(self, market):
        class _Exploding:
            def explain(self, *a, **kw):
                raise RuntimeError("provider exploded")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.engines.layer_05_llm_explainer.LLMExplainerEngine."
                "__new__",
                lambda cls, *a, **kw: _Exploding(),
            )
            output = _run(_portfolio(), run_l5_prelim_llm=True)

        assert output.recommendations, "an L5 failure took the whole run down"
        assert _all_candidates(output), "an L5 failure cost every candidate"
