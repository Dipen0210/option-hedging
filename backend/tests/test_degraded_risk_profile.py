"""
Regression tests for a holding whose risk could not be analysed.

The defect: `RiskAnalysisEngine._fallback_profile` was reached from a bare
`except Exception` around per-holding analysis and returned a *complete*
`RiskProfile` — `var_5pct = notional x 0.15`, `cvar_5pct = notional x 0.20`,
`notional_at_risk = notional x 0.15`, `tail_correlation = 0.7`,
`beta_vs_spy = 1.0`, both hedge ratios 1.0. None of those came from data. The
profile was structurally indistinguishable from a measured one, so:

  * Layer 4 selected, priced and ranked option contracts against a 15% VaR
    that had been assumed into existence, and the user received a contract
    count, a premium and a risk-reduction estimate for it;
  * `_compute_portfolio_beta` weighted the placeholder 1.0 by the holding's
    full notional, pulling the reported portfolio beta toward 1.0 on exactly
    the holdings where nothing was known;
  * nothing in the response said any of this had happened.

The fix is `data_available` / `unavailable_reason` on `RiskProfile`, zeroed
risk figures, `beta_available=False`, L4 declining to build recommendations
for a flagged holding, and a user-facing warning. These tests pin each part.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

import pytest

from backend.engines.layer_03_risk_analysis import RiskAnalysisEngine
from backend.engines.layer_04_instrument_selection import InstrumentSelectionEngine
from backend.models.portfolio_models import PortfolioInput, StockPosition
from backend.models.risk_models import PortfolioRiskSummary, RegimeState, RiskProfile


def _portfolio(*holdings) -> PortfolioInput:
    return PortfolioInput(holdings=list(holdings))


def _stock(ticker: str, shares: float, price: float) -> StockPosition:
    return StockPosition(
        ticker=ticker,
        shares=shares,
        purchase_price=price,
        purchase_date=date.today() - timedelta(days=365),
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


def _measured_profile(ticker: str, notional: float, beta: float) -> RiskProfile:
    return RiskProfile(
        ticker=ticker,
        asset_class="equity",
        notional_value=notional,
        beta_vs_spy=beta,
        beta_available=True,
        var_5pct=notional * 0.08,
        cvar_5pct=notional * 0.11,
        var_pct=0.08,
        cvar_pct=0.11,
        notional_at_risk=notional * 0.08,
        optimal_hedge_ratio=1.0,
        regression_hedge_ratio=beta,
        tail_correlation=0.6,
    )


@pytest.fixture
def engine():
    return RiskAnalysisEngine()


# ── The placeholder no longer looks like a measurement ────────────────────────

class TestFallbackProfileIsFlagged:

    def test_it_is_marked_unavailable_with_a_reason(self, engine):
        holding = _stock("AAPL", 200, 185.0)
        profile = engine._fallback_profile(
            holding, _portfolio(holding), reason="price history empty"
        )
        assert profile.data_available is False
        assert "price history empty" in profile.unavailable_reason

    def test_a_reason_is_recorded_even_when_the_caller_passes_none(self, engine):
        """An empty reason string would render as a blank explanation."""
        holding = _stock("AAPL", 200, 185.0)
        profile = engine._fallback_profile(holding, _portfolio(holding))
        assert profile.unavailable_reason.strip() != ""

    def test_no_invented_risk_figures(self, engine):
        """
        The specific numbers that used to be published as risk: 15% VaR, 20%
        CVaR, 0.7 tail correlation. Zero here is not a claim the holding is
        riskless — `data_available` is the field that carries meaning, and
        these tests exist so a future edit cannot quietly restore a plausible
        non-zero default in its place.
        """
        holding = _stock("AAPL", 200, 185.0)
        notional = 200 * 185.0
        profile = engine._fallback_profile(holding, _portfolio(holding))

        assert profile.var_5pct == 0.0
        assert profile.cvar_5pct == 0.0
        assert profile.var_pct == 0.0
        assert profile.cvar_pct == 0.0
        assert profile.notional_at_risk == 0.0
        assert profile.tail_correlation == 0.0
        # In particular, not the old formulas.
        assert profile.var_5pct != pytest.approx(notional * 0.15)
        assert profile.cvar_5pct != pytest.approx(notional * 0.20)

    def test_the_notional_is_still_reported(self, engine):
        """
        The holding stays in the portfolio — it still contributes to
        concentration, and dropping it would understate the user's exposure.
        """
        holding = _stock("AAPL", 200, 185.0)
        profile = engine._fallback_profile(holding, _portfolio(holding))
        assert profile.notional_value == pytest.approx(200 * 185.0)
        assert profile.ticker == "AAPL"

    def test_the_beta_is_flagged_unmeasured(self, engine):
        holding = _stock("AAPL", 200, 185.0)
        profile = engine._fallback_profile(holding, _portfolio(holding))
        assert profile.beta_available is False


# ── Portfolio beta excludes unmeasured betas ──────────────────────────────────

class TestPortfolioBeta:

    def test_a_placeholder_beta_does_not_drag_the_portfolio_beta(self, engine):
        """
        Two holdings of equal size, one measured at 1.6 and one unmeasurable.
        The old weighted average reported 1.3 — a number that describes
        neither holding and was half-composed of a default. The honest answer
        is the beta of the part that was measured.
        """
        measured = _measured_profile("NVDA", 500_000, 1.6)
        unmeasured = engine._fallback_profile(
            _stock("AAPL", 2000, 250.0), _portfolio(_stock("AAPL", 2000, 250.0))
        )
        portfolio = _portfolio(
            _stock("NVDA", 1000, 500.0), _stock("AAPL", 2000, 250.0)
        )
        portfolio.total_notional = 1_000_000

        beta = engine._compute_portfolio_beta([measured, unmeasured], portfolio)
        assert beta == pytest.approx(1.6, abs=0.001)
        assert beta != pytest.approx(1.3, abs=0.01)

    def test_measured_betas_are_weighted_among_themselves(self, engine):
        """
        Normalised over what was measured, not over the whole portfolio —
        otherwise excluding a holding would dilute the beta toward zero,
        which is just as wrong in the other direction.
        """
        a = _measured_profile("NVDA", 750_000, 2.0)
        b = _measured_profile("KO", 250_000, 0.4)
        portfolio = _portfolio(
            _stock("NVDA", 1500, 500.0), _stock("KO", 4000, 62.5)
        )
        portfolio.total_notional = 1_000_000

        # (0.75 x 2.0) + (0.25 x 0.4) = 1.6
        assert engine._compute_portfolio_beta([a, b], portfolio) == pytest.approx(
            1.6, abs=0.001
        )

    def test_no_measured_beta_anywhere_does_not_crash(self, engine):
        holding = _stock("AAPL", 200, 185.0)
        portfolio = _portfolio(holding)
        portfolio.total_notional = 37_000
        unmeasured = engine._fallback_profile(holding, portfolio)
        assert engine._compute_portfolio_beta([unmeasured], portfolio) == 1.0


# ── Layer 4 refuses to recommend on a placeholder ─────────────────────────────

class _RecordingSelector:
    """
    A stand-in for a real selector that records which holdings it was asked
    about and returns nothing.

    Using it keeps these tests offline and deterministic — the real selectors
    fetch a live option chain, so the same assertions would otherwise pass or
    fail depending on the time of day — and it makes the central claim
    directly observable: L4 must not even *ask* a selector about a holding
    whose risk is a placeholder.
    """

    def __init__(self):
        self.asked: list = []

    def is_applicable(self, profile, regime) -> bool:
        self.asked.append(profile.ticker)
        return False

    def find_candidates(self, profile, portfolio, regime) -> list:   # pragma: no cover
        return []


class TestSelectionDeclinesOnUnavailableProfile:

    def _run(self, profiles, portfolio, selectors=None):
        return InstrumentSelectionEngine(selectors=selectors or []).select(
            portfolio=portfolio,
            regime=_regime(),
            risk_summary=PortfolioRiskSummary(
                total_notional=portfolio.total_notional or 0.0,
                portfolio_beta=1.0,
                portfolio_var_5pct=0.0,
                portfolio_cvar_5pct=0.0,
                concentration_top1=1.0,
                risk_profiles=profiles,
            ),
        )

    def test_no_candidates_are_produced_for_a_flagged_holding(self):
        """
        The core of the defect: a full set of priced, ranked, tradeable
        contracts used to come back from a 15% VaR nobody computed.
        """
        holding = _stock("AAPL", 200, 185.0)
        portfolio = _portfolio(holding)
        portfolio.total_notional = 37_000
        flagged = RiskAnalysisEngine()._fallback_profile(
            holding, portfolio, reason="no price history for AAPL"
        )

        out = self._run([flagged], portfolio)
        assert len(out.recommendations) == 1
        assert out.recommendations[0].asset_ticker == "AAPL"
        assert out.recommendations[0].candidates == []

    def test_the_user_is_told_why_and_that_the_holding_is_unhedged(self):
        """
        Silence here is the dangerous reading: an empty candidate list looks
        identical to "no hedge was worth recommending".
        """
        holding = _stock("AAPL", 200, 185.0)
        portfolio = _portfolio(holding)
        portfolio.total_notional = 37_000
        flagged = RiskAnalysisEngine()._fallback_profile(
            holding, portfolio, reason="no price history for AAPL"
        )

        out = self._run([flagged], portfolio)
        joined = " ".join(out.warnings)
        assert "AAPL" in joined
        assert "no price history for AAPL" in joined
        assert "unhedged" in joined.lower()

    def test_a_flagged_holding_does_not_suppress_its_neighbours(self):
        """
        One unanalysable holding must not cost the user the rest of the
        analysis — the recommendation list keeps an entry per holding and the
        healthy one is still selected for.
        """
        bad = _stock("AAPL", 200, 185.0)
        good = _stock("SPY", 100, 550.0)
        portfolio = _portfolio(bad, good)
        portfolio.total_notional = 92_000
        flagged = RiskAnalysisEngine()._fallback_profile(
            bad, portfolio, reason="no price history for AAPL"
        )

        recorder = _RecordingSelector()
        out = self._run(
            [flagged, _measured_profile("SPY", 55_000, 1.0)],
            portfolio,
            selectors=[recorder],
        )
        assert [r.asset_ticker for r in out.recommendations] == ["AAPL", "SPY"]
        assert out.recommendations[0].candidates == []
        # Selection ran for the healthy holding and was never attempted for
        # the flagged one.
        assert recorder.asked == ["SPY"]

        # SPY must not be tarred with AAPL's failure. It is, separately,
        # entitled to a disclosure of its own: the recorder returns nothing,
        # and an empty candidate list is now always explained rather than
        # left to read as "no hedge was worth recommending". The two must not
        # be confused, which is what this asserts — the earlier form of this
        # test asserted SPY appeared in no warning at all, and that would
        # pass just as well if SPY's own empty result were silently dropped.
        spy_warnings = [w for w in out.warnings if "SPY" in w]
        assert len(spy_warnings) == 1, out.warnings
        assert "no price history" not in spy_warnings[0]
        assert "AAPL" not in spy_warnings[0]
        assert "unhedged" in spy_warnings[0].lower()

        aapl_warnings = [w for w in out.warnings if "AAPL" in w]
        assert len(aapl_warnings) == 1, out.warnings
        assert "no price history for AAPL" in aapl_warnings[0]
