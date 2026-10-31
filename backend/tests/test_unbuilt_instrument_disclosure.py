"""
Regression tests for what the product says when it did not look.

Four of the five instrument selectors are declared stubs: `find_candidates`
returns [] without pricing anything. `POST /instruments/futures` therefore
cannot produce a hedge, and the pipeline explained that empty result with

    "No hedge for AAPL fits the cost budget and liquidity constraints;
     the position is shown unhedged."

which names a financial cause for a search that never ran. A user reading it
would conclude futures hedging is uneconomic for their portfolio when no
future had been examined, no budget consulted and no quote fetched. The text
came from `_drop_unsizeable`, whose job is to remove candidates L7 could not
size — it fired on holdings that arrived with no candidates at all, where
nothing had been dropped.

These tests pin the distinction between the reasons an empty candidate list
can have, because the value itself is identical in every case:

  * a selector evaluated the holding and found nothing worth recommending
  * the selector for that instrument class is not built
  * no instrument class applies to this asset class at all
  * a selector raised
  * candidates existed and sizing removed every one of them  <- and only
    this last one is a cost-and-liquidity finding
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pytest

from backend.engines.layer_04_instrument_selection import InstrumentSelectionEngine
from backend.engines.pipeline_orchestrator import HedgeOrchestrator
from backend.instruments.base import InstrumentSelector
from backend.instruments.forwards.selector import ForwardsSelector
from backend.instruments.futures.selector import FuturesSelector
from backend.instruments.inverse_etfs.selector import InverseETFSelector
from backend.instruments.options.selector import OptionsSelector
from backend.instruments.swaps.selector import SwapsSelector
from backend.models.hedge_models import (
    HedgeOutput,
    HedgeRecommendation,
    InstrumentCandidate,
)
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.risk_models import PortfolioRiskSummary, RegimeState, RiskProfile

NOTIONAL = 200 * 185.0


def _portfolio() -> PortfolioInput:
    # total_notional is set explicitly: the field is optional on the model and
    # the live pipeline fills it in L1, before selection ever runs.
    return PortfolioInput(
        holdings=[StockPosition(
            ticker="AAPL",
            shares=200,
            purchase_price=185.0,
            purchase_date=date.today() - timedelta(days=365),
        )],
        total_notional=NOTIONAL,
    )


def _regime() -> RegimeState:
    return RegimeState(
        regime_label="mid_vol",
        regime_id=1,
        is_anomaly=False,
        anomaly_score=0.2,
        soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3,
        vix_level=18.0,
        vol_forecast_garch=0.18,
        realized_vol_20d=0.16,
    )


def _profile() -> RiskProfile:
    """A measured profile: nothing here is a reason to decline a hedge."""
    return RiskProfile(
        ticker="AAPL",
        asset_class="equity",
        notional_value=NOTIONAL,
        beta_vs_spy=1.2,
        beta_available=True,
        var_5pct=NOTIONAL * 0.08,
        cvar_5pct=NOTIONAL * 0.11,
        var_pct=0.08,
        cvar_pct=0.11,
        notional_at_risk=NOTIONAL * 0.08,
        optimal_hedge_ratio=1.0,
        regression_hedge_ratio=1.2,
        tail_correlation=0.6,
    )


def _warnings_from(selectors) -> str:
    """Run L4 with the given selectors and return its warnings as one string."""
    out = InstrumentSelectionEngine(selectors=selectors).select(
        portfolio=_portfolio(),
        regime=_regime(),
        risk_summary=PortfolioRiskSummary(
            total_notional=NOTIONAL,
            portfolio_beta=1.2,
            portfolio_var_5pct=0.0,
            portfolio_cvar_5pct=0.0,
            concentration_top1=1.0,
            risk_profiles=[_profile()],
        ),
    )
    return " ".join(out.warnings)


# ── Test doubles, one per reason an empty list can have ───────────────────────

class _Unbuilt(InstrumentSelector):
    """The shape of all four real stubs."""
    label = "test futures"
    implemented = False

    def find_candidates(self, profile, portfolio, regime):
        raise AssertionError("an unbuilt selector must never be asked")


class _Searched(InstrumentSelector):
    """A built selector that looked and found nothing."""
    label = "test options"

    def find_candidates(self, profile, portfolio, regime):
        return []


class _Inapplicable(InstrumentSelector):
    label = "test swaps"

    def is_applicable(self, profile, regime):
        return False

    def find_candidates(self, profile, portfolio, regime):
        raise AssertionError("an inapplicable selector must never be asked")


class _Broken(InstrumentSelector):
    label = "test credit"

    def find_candidates(self, profile, portfolio, regime):
        raise RuntimeError("upstream chain parse failed: KeyError 'Close'")


class TestAnEmptyListIsExplainedByWhatActuallyHappened:

    def test_an_unbuilt_selector_says_so_and_is_never_asked(self):
        warnings = _warnings_from([_Unbuilt()])
        assert "not implemented" in warnings
        assert "test futures" in warnings
        assert "unhedged" in warnings.lower()
        # The claim that was wrong, in the words it was wrong in.
        assert "cost budget" not in warnings
        assert "liquidity" not in warnings

    def test_a_selector_that_looked_is_not_reported_as_unbuilt(self):
        warnings = _warnings_from([_Searched()])
        assert "no hedge candidate was found" in warnings
        assert "test options" in warnings
        assert "not implemented" not in warnings
        assert "cost budget" not in warnings

    def test_an_inapplicable_asset_class_says_that_instead(self):
        warnings = _warnings_from([_Inapplicable()])
        assert "no instrument type in this run applies" in warnings
        assert "equity" in warnings
        assert "not implemented" not in warnings
        assert "cost budget" not in warnings

    def test_a_selector_that_raised_is_disclosed_not_swallowed(self):
        warnings = _warnings_from([_Broken()])
        assert "failed" in warnings
        assert "test credit" in warnings
        # The failure reaches the user as a fact, never as its internals.
        assert "KeyError" not in warnings
        assert "Traceback" not in warnings
        assert "upstream chain parse failed" not in warnings

    def test_a_failure_is_disclosed_even_when_another_class_delivers(self):
        """
        A partial run must not read as a complete one. The empty result here
        has two causes and the error is the one the user is owed.
        """
        warnings = _warnings_from([_Broken(), _Unbuilt()])
        assert "failed" in warnings and "test credit" in warnings

    def test_a_selector_without_a_label_is_not_counted_as_a_failure(self):
        """
        A selector need only satisfy the two-method protocol. Reading `.label`
        where a missing attribute becomes a caught exception would report a
        working selector as broken — which is how the first version of this
        fix behaved.
        """
        class _NoLabel:
            def is_applicable(self, profile, regime):
                return True

            def find_candidates(self, profile, portfolio, regime):
                return []

        warnings = _warnings_from([_NoLabel()])
        assert "failed" not in warnings
        assert "no hedge candidate was found" in warnings


class TestCostAndLiquidityIsClaimedOnlyWhenItWasMeasured:

    @staticmethod
    def _output(candidates) -> HedgeOutput:
        return HedgeOutput(
            portfolio_notional=NOTIONAL,
            regime="mid_vol",
            is_anomaly=False,
            vix_level=18.0,
            run_time_seconds=0.1,
            recommendations=[HedgeRecommendation(
                rank=1, asset_ticker="AAPL", candidates=candidates,
            )],
        )

    @staticmethod
    def _candidate(n_contracts: int) -> InstrumentCandidate:
        return InstrumentCandidate(
            instrument_type="option",
            asset_class="equity",
            ticker="AAPL",
            n_contracts=n_contracts,
        )

    def test_a_holding_that_arrived_empty_gets_no_budget_claim(self):
        """The regression itself: nothing was dropped, so nothing was priced."""
        out = self._output([])
        HedgeOrchestrator._drop_unsizeable(out)
        assert out.warnings == []

    def test_a_holding_whose_candidates_were_all_unsizeable_still_gets_it(self):
        """
        The honest case has to survive the fix: these candidates were priced,
        and the budget is genuinely why none of them can be traded.
        """
        out = self._output([self._candidate(0), self._candidate(0)])
        HedgeOrchestrator._drop_unsizeable(out)
        assert len(out.warnings) == 1
        assert "cost budget and liquidity constraints" in out.warnings[0]
        assert "AAPL" in out.warnings[0]

    def test_a_sizeable_candidate_produces_no_warning(self):
        out = self._output([self._candidate(3)])
        HedgeOrchestrator._drop_unsizeable(out)
        assert out.warnings == []
        assert len(out.recommendations[0].candidates) == 1


class TestTheRealSelectorsDeclareThemselvesHonestly:
    """
    The disclosure is only as true as these flags. A stub that forgets
    `implemented = False` reports "no hedge candidate was found" — the
    searched-and-rejected story — for a search that never happened.
    """

    @pytest.mark.parametrize("selector,label", [
        (FuturesSelector(), "futures"),
        (ForwardsSelector(), "forwards"),
        (SwapsSelector(), "swaps"),
        (InverseETFSelector(), "inverse ETFs"),
    ])
    def test_the_four_stubs_declare_themselves_unbuilt(self, selector, label):
        assert selector.implemented is False
        assert selector.label == label
        # And the reason they are stubs: they return nothing, unconditionally.
        assert selector.find_candidates(_profile(), _portfolio(), _regime()) == []

    def test_options_is_the_one_that_is_built(self):
        assert OptionsSelector().implemented is True
        assert OptionsSelector().label == "options"
