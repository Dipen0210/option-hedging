"""
Regression tests for a cost ratio that was published as a protection claim.

`PromptBuilder.build_combined_context` handed the model one field per holding
called `var_coverage_pct`, computed as:

    holding_var = var_by_ticker.get(rec.asset_ticker, 1)
    var_coverage_pct = round(cost / holding_var * 100, 1) if holding_var else 0

Three defects sat in those two lines, and all three reached the user:

  * The name said coverage; the value was `cost / VaR`. A live run against
    real option chains produced the headline "Buy the AAPL Direct Put (3% OTM)
    expiring 2027-02-19 for 1312 USD to cover 9.3% of portfolio VaR" — 9.3%
    being 1312 / 14051, the cost over AAPL's VaR. The put's maximum payoff
    covered 11523 of that 14051, i.e. 82%. The user was told a hedge covered
    9.3% of their risk when it covered 82% of that holding's.
  * The denominator was one holding's VaR, but the field was nested under
    `portfolio`, so the model attributed it to the portfolio. Portfolio VaR in
    the same run was 23756.
  * The fallbacks invented figures. A ticker absent from the lookup divided by
    `1`, turning a $1,312 hedge into "131250%"; a holding whose VaR was zero —
    which is exactly what `_fallback_profile` now sets for an unmeasurable
    holding — reported `0`, which reads as "this hedge covers nothing" rather
    than "VaR was not measured".

The fix splits the quantity in two, names the denominator in both field names,
and passes absence through as null for the prompt's null rule to handle.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pytest

from backend.core.explainer.prompt_builder import COMBINED_SYSTEM, PromptBuilder
from backend.models.hedge_models import (
    HedgeOutput, HedgeRecommendation, InstrumentCandidate,
)
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.risk_models import PortfolioRiskSummary, RegimeState, RiskProfile

NOTIONAL        = 40_000.0
HOLDING_VAR     = 3_200.0      # 8% of notional
PORTFOLIO_VAR   = 5_000.0      # deliberately != HOLDING_VAR, so a test can tell
COST            = 300.0
MAX_PROTECTION  = 2_400.0
# 300 / 3200 = 9.4%   vs   2400 / 3200 = 75.0%. The old single field reported
# the former under a name that made the model say the latter.
EXPECTED_COST_PCT     = 9.4
EXPECTED_COVERED_PCT  = 75.0


def _portfolio() -> PortfolioInput:
    return PortfolioInput(
        holdings=[StockPosition(
            ticker="AAPL", shares=200, purchase_price=150.0,
            purchase_date=date.today() - timedelta(days=365),
        )],
        total_notional=NOTIONAL,
    )


def _regime() -> RegimeState:
    return RegimeState(
        regime_label="mid_vol", regime_id=1, is_anomaly=False,
        anomaly_score=0.2, soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3, vix_level=17.5,
        vol_forecast_garch=0.18, realized_vol_20d=0.16,
    )


def _profile(*, var: float = HOLDING_VAR, available: bool = True) -> RiskProfile:
    return RiskProfile(
        ticker="AAPL", asset_class="equity", notional_value=NOTIONAL,
        beta_vs_spy=1.2, beta_available=True,
        var_5pct=var, cvar_5pct=var * 1.4,
        var_pct=var / NOTIONAL, cvar_pct=var * 1.4 / NOTIONAL,
        notional_at_risk=var,
        optimal_hedge_ratio=1.0, regression_hedge_ratio=1.2,
        tail_correlation=0.6,
        data_available=available,
        unavailable_reason="" if available else "no price history for AAPL",
    )


def _summary(profiles) -> PortfolioRiskSummary:
    return PortfolioRiskSummary(
        total_notional=NOTIONAL, portfolio_beta=1.2,
        portfolio_var_5pct=PORTFOLIO_VAR,
        portfolio_cvar_5pct=PORTFOLIO_VAR * 1.4,
        concentration_top1=1.0, risk_profiles=list(profiles),
    )


def _candidate(**over) -> InstrumentCandidate:
    kwargs = dict(
        instrument_type="option", asset_class="equity", ticker="AAPL",
        strategy="Direct Put (3% OTM)", n_contracts=2, strike=190.0,
        total_cost=COST, max_protection=MAX_PROTECTION,
    )
    kwargs.update(over)
    return InstrumentCandidate(**kwargs)


def _output(candidates, ticker: str = "AAPL") -> HedgeOutput:
    return HedgeOutput(
        portfolio_notional=NOTIONAL, regime="mid_vol", is_anomaly=False,
        vix_level=17.5, run_time_seconds=0.1,
        recommendations=[HedgeRecommendation(
            rank=1, asset_ticker=ticker, candidates=candidates,
        )],
    )


def _context(profiles=None, candidates=None, ticker: str = "AAPL") -> dict:
    """The dict actually handed to the model, which is what these tests assert on."""
    profiles = [_profile()] if profiles is None else profiles
    candidates = [_candidate()] if candidates is None else candidates
    summary = _summary(profiles)
    output = _output(candidates, ticker=ticker)
    return PromptBuilder().build_combined_context(
        positions=[(ticker, NOTIONAL, profiles[0], candidates)],
        portfolio=_portfolio(),
        risk_summary=summary,
        hedge_output=output,
        regime=_regime(),
    )


def _top(ctx: dict) -> dict:
    entries = ctx["portfolio"]["top_hedge_per_holding"]
    assert len(entries) == 1, entries
    return entries[0]


class TestCostAndProtectionAreSeparateFigures:

    def test_the_cost_ratio_is_named_as_cost(self):
        top = _top(_context())
        assert top["cost_pct_of_holding_var"] == EXPECTED_COST_PCT

    def test_the_protection_ratio_measures_the_payoff_not_the_premium(self):
        """
        The defect in one line: the only ratio offered was the premium's, and
        it was called coverage.
        """
        top = _top(_context())
        assert top["holding_var_covered_pct"] == EXPECTED_COVERED_PCT

    def test_the_two_are_not_the_same_number(self):
        """
        Guards the fixture itself. If cost and max_protection were ever set to
        the same value here, every other test in this class would pass while
        conflating the two.
        """
        top = _top(_context())
        assert top["cost_pct_of_holding_var"] != top["holding_var_covered_pct"]

    def test_no_field_is_called_coverage_without_saying_of_what(self):
        """
        `var_coverage_pct` is the name the model read as a portfolio-level
        protection claim. It must not come back, under that name or a bare
        `coverage`, because the name is what carried the false meaning.
        """
        top = _top(_context())
        assert "var_coverage_pct" not in top
        for key in top:
            if "cover" in key:
                assert key.startswith("holding_"), (
                    f"{key!r} does not name the risk it is measured against"
                )

    def test_both_denominators_are_published_so_the_ratios_can_be_checked(self):
        top = _top(_context())
        assert top["holding_var_usd"] == round(HOLDING_VAR)
        assert top["max_protection_usd"] == round(MAX_PROTECTION)
        assert top["cost_usd"] == round(COST)


class TestAPerHoldingFigureIsNotAttributedToThePortfolio:

    def test_the_holding_var_is_the_holdings_not_the_portfolios(self):
        top = _top(_context())
        assert top["holding_var_usd"] == round(HOLDING_VAR)
        assert top["holding_var_usd"] != round(PORTFOLIO_VAR)

    def test_the_portfolio_keeps_its_own_var_field(self):
        ctx = _context()
        assert ctx["portfolio"]["var_5pct_usd"] == round(PORTFOLIO_VAR)

    def test_the_prompt_forbids_calling_a_holding_figure_portfolio_var(self):
        lowered = COMBINED_SYSTEM.lower()
        assert "portfolio var" in lowered
        assert "holding_var_covered_pct" in lowered
        assert "cost_pct_of_holding_var" in lowered
        # The rule has to say both halves: the denominator, and that a cost
        # ratio is not a protection claim.
        assert "never describe them as covering" in lowered
        assert "never a statement about protection" in lowered


class TestAnUnmeasuredRiskProducesNoRatio:

    def test_a_missing_ticker_does_not_divide_by_one(self):
        """
        `.get(ticker, 1)` made a $300 hedge "30000%" of a $1 VaR. Null is the
        only honest answer, and the prompt's null rule drops the claim.
        """
        top = _top(_context(profiles=[_profile()], ticker="MSFT"))
        assert top["cost_pct_of_holding_var"] is None
        assert top["holding_var_covered_pct"] is None
        assert top["holding_var_usd"] is None

    def test_a_flagged_holding_reports_null_not_zero(self):
        """
        `_fallback_profile` sets `var_5pct = 0.0` with `data_available=False`.
        Zero here read as "this hedge covers nothing", which is a finding; the
        truth is that no finding was made.
        """
        top = _top(_context(profiles=[_profile(var=0.0, available=False)]))
        assert top["cost_pct_of_holding_var"] is None
        assert top["holding_var_covered_pct"] is None

    def test_a_zero_var_on_an_available_profile_is_also_declined(self):
        """Dividing by it would raise; substituting for it would invent."""
        top = _top(_context(profiles=[_profile(var=0.0, available=True)]))
        assert top["cost_pct_of_holding_var"] is None
        assert top["holding_var_covered_pct"] is None

    def test_a_candidate_without_a_max_protection_reports_no_coverage(self):
        """
        Coverage needs a payoff. `max_protection` is a non-optional float
        defaulting to 0.0, so "not computed" and "zero" are the same value on
        the model and both have to decline the ratio rather than publish 0.0%
        — a hedge that covers 0% of the risk is a finding, and this is not one.
        The cost ratio is still computable and must still be reported;
        declining both would lose a figure that was measured.
        """
        top = _top(_context(candidates=[_candidate(max_protection=0.0)]))
        assert top["holding_var_covered_pct"] is None
        assert top["max_protection_usd"] is None
        assert top["cost_pct_of_holding_var"] == EXPECTED_COST_PCT
