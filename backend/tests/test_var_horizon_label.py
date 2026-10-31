"""
Regression tests for the VaR horizon label.

`RiskProfile.var_5pct` and `PortfolioRiskSummary.portfolio_var_5pct` are
scaled to the portfolio's hedge horizon — L3 passes
`horizon_days=portfolio.hedge_horizon_days` into the calculator, which scales
sigma by the square root of the horizon in *trading* days. Two consumers then
asserted a horizon of their own instead of reading the one the figure had:

  * `layer_11_llm_explainer._deterministic_narrative` opened the sentence
    "One-day 95% VaR is $42.2K". On the verified four-holding test portfolio
    with a 120-day hedge horizon, the true one-day figure is about $4.6K. The
    narrative presented a four-month tail loss as an overnight one, understating
    the stated horizon by a factor of roughly sqrt(83) ~ 9.

  * The UI tile was labelled "1mo VaR (95%)" and the `UnhedgedRisk.var_95`
    type comment read "(1-month)". The horizon defaults to 180 days and is
    user-chosen, so "1mo" was wrong both by default and for any other input.

A dollar VaR without its horizon is not a risk number — the same position
reads $4.6K over a day and $42.2K over four months. So the horizon now travels
with the figure (`var_horizon_calendar_days` / `var_horizon_trading_days` on
the risk models, `horizon_calendar_days` / `horizon_trading_days` on
`UnhedgedRisk`) and every label is derived from it. Where the horizon is
unknown the label names no horizon at all rather than defaulting to a wrong
one.
"""
import numpy as np
import pandas as pd
import pytest

from backend.core.risk.var_calculator import VaRCalculator, _trading_days
from backend.engines.layer_11_llm_explainer import (
    _deterministic_narrative,
    _var_horizon_phrase,
)
from backend.engines.layer_12_output_formatter import OutputFormatterEngine
from backend.models.hedge_models import (
    HedgeOutput,
    HedgeRecommendation,
    InstrumentCandidate,
)
from backend.models.risk_models import (
    PortfolioRiskSummary,
    RegimeState,
    RiskProfile,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _returns(n: int = 500, sigma_daily: float = 0.01, seed: int = 7) -> pd.Series:
    """A deterministic daily-return series with a known daily sigma."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-06-30", periods=n)
    return pd.Series(rng.normal(0.0, sigma_daily, n), index=idx)


def _profile(
    ticker: str = "AAPL",
    *,
    var: float = 42_200.0,
    horizon_calendar: int = 120,
    horizon_trading: float = 82.77,
) -> RiskProfile:
    return RiskProfile(
        ticker=ticker,
        asset_class="equity",
        notional_value=337_116.0,
        var_5pct=var,
        cvar_5pct=var * 1.63,
        var_pct=0.125,
        cvar_pct=0.204,
        notional_at_risk=var,
        var_horizon_calendar_days=horizon_calendar,
        var_horizon_trading_days=horizon_trading,
        optimal_hedge_ratio=1.0,
        regression_hedge_ratio=0.73,
        tail_correlation=0.6,
    )


def _summary(**kwargs) -> PortfolioRiskSummary:
    profile = kwargs.pop("profile", _profile())
    defaults = dict(
        total_notional=337_116.0,
        portfolio_beta=0.73,
        portfolio_var_5pct=42_200.0,
        portfolio_cvar_5pct=68_800.0,
        concentration_top1=0.28,
        risk_profiles=[profile],
        # Matches _profile()'s default so the fixture is self-consistent;
        # tests that care about an unknown horizon pass 0 explicitly.
        var_horizon_calendar_days=120,
        var_horizon_trading_days=82.77,
    )
    defaults.update(kwargs)
    return PortfolioRiskSummary(**defaults)


def _regime() -> RegimeState:
    return RegimeState(
        regime_label="mid_vol",
        regime_id=1,
        is_anomaly=False,
        anomaly_score=0.3,
        soft_membership=[0.2, 0.7, 0.1],
        n_discovered_regimes=3,
        vix_level=17.2,
        vol_forecast_garch=0.18,
        realized_vol_20d=0.10,
    )


def _hedge_output() -> HedgeOutput:
    return HedgeOutput(
        portfolio_notional=337_116.0,
        regime="mid_vol",
        is_anomaly=False,
        vix_level=17.2,
        run_time_seconds=0.0,
        recommendations=[
            HedgeRecommendation(rank=1, asset_ticker="AAPL", candidates=[]),
        ],
    )


# ── The figure really is horizon-scaled ──────────────────────────────────────

class TestTheFigureIsHorizonScaled:
    """
    The premise that makes the old labels wrong: these are not one-day numbers.
    If this class ever fails, the labels are not the thing to change.
    """

    def test_a_120_day_var_is_far_larger_than_a_one_day_var(self):
        calc = VaRCalculator()
        returns = _returns()
        one_day = calc.parametric_cf_var(
            returns, notional=1_000_000, horizon_days=1
        )
        horizon = calc.parametric_cf_var(
            returns, notional=1_000_000, horizon_days=120
        )

        ratio = horizon["var_dollar"] / one_day["var_dollar"]
        expected = _trading_days(120) ** 0.5

        # sqrt(83) ~ 9.1. Calling the 120-day figure "One-day" understated the
        # loss by this factor.
        assert ratio == pytest.approx(expected, rel=0.02)
        assert ratio > 8.5

    def test_the_calculator_reports_the_horizon_it_used(self):
        calc = VaRCalculator()
        out = calc.parametric_cf_var(
            _returns(), notional=1_000_000, horizon_days=120
        )
        assert out["horizon_trading_days"] == round(_trading_days(120), 2)

    def test_the_default_horizon_is_not_one_month(self):
        # The signature default is 180 calendar days, so neither "1mo" nor
        # "one-day" describes an unspecified call.
        calc = VaRCalculator()
        out = calc.parametric_cf_var(_returns(), notional=1_000_000)
        assert out["horizon_trading_days"] == round(_trading_days(180), 2)


# ── The horizon phrase ───────────────────────────────────────────────────────

class TestHorizonPhrase:

    def test_a_multi_day_horizon_is_named(self):
        assert _var_horizon_phrase(_summary(var_horizon_calendar_days=120)) == (
            "Over the 120-day hedge horizon, "
        )

    def test_the_user_chosen_horizon_is_the_one_reported(self):
        # Not a fixed month and not a fixed default — whatever was requested.
        for days in (30, 90, 180, 365):
            phrase = _var_horizon_phrase(
                _summary(var_horizon_calendar_days=days)
            )
            assert f"{days}-day" in phrase

    def test_a_one_day_horizon_is_still_called_one_day(self):
        assert _var_horizon_phrase(
            _summary(var_horizon_calendar_days=1)
        ) == "One-day "

    def test_an_unknown_horizon_names_no_horizon(self):
        # The fix must not swap one wrong default for another: with no horizon
        # recorded, the sentence claims none.
        phrase = _var_horizon_phrase(_summary(var_horizon_calendar_days=0))
        assert phrase == ""
        assert "day" not in phrase


# ── The narrative sentence ───────────────────────────────────────────────────

class TestNarrativeHorizon:

    def test_a_120_day_var_is_not_described_as_one_day(self):
        out = _hedge_output()
        _deterministic_narrative(
            out, _summary(var_horizon_calendar_days=120), _regime()
        )
        summary = out.portfolio_summary

        assert "One-day" not in summary
        assert "one-day" not in summary
        assert "120-day hedge horizon" in summary
        assert "95% VaR" in summary

    def test_the_sentence_is_well_formed(self):
        out = _hedge_output()
        _deterministic_narrative(
            out, _summary(var_horizon_calendar_days=90), _regime()
        )
        assert (
            "Over the 90-day hedge horizon, 95% VaR is"
            in out.portfolio_summary
        )

    def test_an_unknown_horizon_still_reports_the_figure(self):
        # Refusing to name a horizon must not drop the number or leave a
        # dangling phrase.
        out = _hedge_output()
        _deterministic_narrative(
            out, _summary(var_horizon_calendar_days=0), _regime()
        )
        summary = out.portfolio_summary
        assert "95% VaR is" in summary
        assert "One-day" not in summary
        assert "hedge horizon," not in summary

    def test_no_horizon_is_named_when_there_is_no_var(self):
        out = _hedge_output()
        _deterministic_narrative(
            out,
            _summary(portfolio_var_5pct=0.0, var_horizon_calendar_days=120),
            _regime(),
        )
        assert "VaR" not in out.portfolio_summary


# ── The horizon reaches the API response ─────────────────────────────────────

class TestUnhedgedRiskCarriesTheHorizon:

    def _run_formatter(self, profile: RiskProfile) -> HedgeOutput:
        out = HedgeOutput(
            portfolio_notional=profile.notional_value,
            regime="mid_vol",
            is_anomaly=False,
            vix_level=17.2,
            run_time_seconds=0.0,
            recommendations=[
                HedgeRecommendation(
                    rank=1, asset_ticker=profile.ticker, candidates=[]
                ),
            ],
        )
        return OutputFormatterEngine().run(
            out,
            pipeline_start_time=0.0,
            risk_summary=_summary(profile=profile),
        )

    def test_the_horizon_travels_with_the_var(self):
        profile = _profile(horizon_calendar=120, horizon_trading=82.77)
        result = self._run_formatter(profile)

        ur = result.recommendations[0].unhedged_risk
        assert ur is not None
        assert ur.var_95 == pytest.approx(profile.var_5pct)
        assert ur.horizon_calendar_days == 120
        assert ur.horizon_trading_days == pytest.approx(82.77)

    def test_the_horizon_is_not_hardcoded_to_a_month(self):
        result = self._run_formatter(
            _profile(horizon_calendar=365, horizon_trading=251.8)
        )
        ur = result.recommendations[0].unhedged_risk
        assert ur.horizon_calendar_days == 365
        assert ur.horizon_calendar_days != 30

    def test_an_unknown_horizon_is_reported_as_unknown_not_guessed(self):
        result = self._run_formatter(
            _profile(horizon_calendar=0, horizon_trading=0.0)
        )
        ur = result.recommendations[0].unhedged_risk
        # Zero is the signal the UI uses to omit the horizon from the label.
        assert ur.horizon_calendar_days == 0


# ── L3 records the horizon it priced against ─────────────────────────────────

class TestRiskModelsDefaults:

    def test_an_unset_horizon_defaults_to_unknown_not_to_one_day(self):
        # A profile built without a horizon must not imply one day, which is
        # what the old narrative assumed of every profile.
        profile = RiskProfile(
            ticker="X",
            asset_class="equity",
            notional_value=1000.0,
            var_5pct=10.0,
            cvar_5pct=15.0,
            var_pct=0.01,
            cvar_pct=0.015,
            notional_at_risk=10.0,
            optimal_hedge_ratio=1.0,
            regression_hedge_ratio=1.0,
            tail_correlation=0.5,
        )
        assert profile.var_horizon_calendar_days == 0
        assert profile.var_horizon_trading_days == 0.0
        assert _var_horizon_phrase(
            _summary(profile=profile, var_horizon_calendar_days=0)
        ) != "One-day "


# ── A deterministic narrative is always disclosed as one ─────────────────────

class TestNarrativeSubstitutionIsDisclosed:
    """
    An end-to-end run found the provider reachable and raising nothing, yet
    returning blank narrative fields. The backstop filled all four, and the
    response went out with `llm_provider="huggingface"` and an empty
    `llm_notice` — template text presented as a model's analysis. The three
    paths that already set a notice (provider unavailable, rate limit,
    exception) were covered; a successful-but-empty response was not.
    """

    def test_the_narrative_reports_which_fields_it_filled(self):
        out = _hedge_output()
        filled = _deterministic_narrative(out, _summary(), _regime())

        assert set(filled) == {
            "portfolio_summary",
            "key_risks",
            "regime_commentary",
            "top_recommendation",
        }

    def test_fields_the_model_wrote_are_not_reported_as_filled(self):
        out = _hedge_output()
        out.portfolio_summary = "A model wrote this."
        out.top_recommendation = "And this."

        filled = _deterministic_narrative(out, _summary(), _regime())

        assert "portfolio_summary" not in filled
        assert "top_recommendation" not in filled
        # And the model's text is left alone.
        assert out.portfolio_summary == "A model wrote this."

    def test_a_fully_written_narrative_reports_nothing_filled(self):
        # The only case in which no notice is warranted.
        out = _hedge_output()
        out.portfolio_summary = "s"
        out.key_risks = ["r"]
        out.regime_commentary = "c"
        out.top_recommendation = "t"

        assert _deterministic_narrative(out, _summary(), _regime()) == []


class TestEngineDisclosesSubstitution:
    """
    The engine-level defect: a provider that is available, raises nothing, and
    returns blanks produced a response with a provider name and no notice.
    """

    @staticmethod
    def _portfolio():
        from datetime import date
        from backend.models.portfolio_models import PortfolioInput, StockPosition

        return PortfolioInput(
            holdings=[
                StockPosition(
                    ticker="AAPL",
                    shares=400,
                    purchase_price=185.0,
                    purchase_date=date(2025, 1, 15),
                    asset_class="equity",
                ),
            ],
            # Set explicitly: L1 computes this upstream, and leaving it None
            # makes the engine fail inside the prompt builder, which would let
            # these tests pass via the exception path instead of the one under
            # test.
            total_notional=337_116.0,
            hedge_horizon_days=120,
            max_hedge_cost_pct=0.03,
        )

    class _BlankExplainer:
        """Available, raises nothing, says nothing — the observed case."""
        provider_name = "stub-provider"

        def is_available(self) -> bool:
            return True

        def supports_combined_call(self) -> bool:
            return False

        def explain_candidates(self, ctx):
            return []

        def explain_portfolio(self, ctx):
            from backend.core.explainer.base_explainer import PortfolioExplanation
            return PortfolioExplanation(
                summary="",
                key_risks=[],
                regime_commentary="",
                top_recommendation="",
            )

    @staticmethod
    def _output_with_a_candidate() -> HedgeOutput:
        """
        One candidate, because this class is about what the engine discloses
        when the *model* writes nothing.

        The shared `_hedge_output()` carries no candidates, and L11 no longer
        calls a provider for a run where no holding has one — there is nothing
        to explain, and asking a model to recommend a hedge that does not
        exist is what produced a fabricated position size. With that fixture
        these tests still passed, but through the skip path: the explainer was
        never invoked and the notice they asserted on was the
        no-candidate one, which also mentions "computed figures". They would
        have gone on passing with the substitution disclosure entirely
        removed.
        """
        out = _hedge_output()
        out.recommendations[0].candidates = [InstrumentCandidate(
            instrument_type="option",
            asset_class="equity",
            ticker="AAPL",
            strategy="protective_put",
            n_contracts=4,
        )]
        return out

    def _run(self, monkeypatch, explainer):
        import backend.engines.layer_11_llm_explainer as l11

        monkeypatch.setattr(l11, "get_explainer", lambda provider=None: explainer)
        return l11.FinalLLMExplainerEngine().run(
            hedge_output=self._output_with_a_candidate(),
            portfolio=self._portfolio(),
            risk_summary=_summary(profile=_profile(ticker="AAPL")),
            regime=_regime(),
        )

    def test_a_blank_llm_response_is_disclosed(self, monkeypatch):
        out = self._run(monkeypatch, self._BlankExplainer())

        # The text is there...
        assert out.portfolio_summary.strip()
        assert out.top_recommendation.strip()
        # ...and so is the statement that the model did not write it. This
        # assertion is the regression: it used to be the empty string.
        assert out.llm_notice.strip()
        assert "computed figures" in out.llm_notice

    def test_the_narrative_still_carries_the_right_horizon(self, monkeypatch):
        out = self._run(monkeypatch, self._BlankExplainer())
        assert "120-day hedge horizon" in out.portfolio_summary
        assert "one-day" not in out.portfolio_summary.lower()

    def test_a_partial_response_names_the_substituted_sections(self, monkeypatch):
        class _PartialExplainer(TestEngineDisclosesSubstitution._BlankExplainer):
            def explain_portfolio(self, ctx):
                from backend.core.explainer.base_explainer import PortfolioExplanation
                return PortfolioExplanation(
                    summary="The model wrote a real summary.",
                    key_risks=["A real risk."],
                    regime_commentary="",
                    top_recommendation="",
                )

        out = self._run(monkeypatch, _PartialExplainer())

        assert out.portfolio_summary == "The model wrote a real summary."
        assert out.llm_notice.strip()
        assert "regime commentary" in out.llm_notice
        assert "top recommendation" in out.llm_notice
        # The sections the model did write are not claimed as substituted.
        assert "portfolio summary" not in out.llm_notice

    def test_a_complete_response_gets_no_notice(self, monkeypatch):
        class _GoodExplainer(TestEngineDisclosesSubstitution._BlankExplainer):
            def explain_portfolio(self, ctx):
                from backend.core.explainer.base_explainer import PortfolioExplanation
                return PortfolioExplanation(
                    summary="Model summary.",
                    key_risks=["Model risk."],
                    regime_commentary="Model regime view.",
                    top_recommendation="Model pick.",
                )

        out = self._run(monkeypatch, _GoodExplainer())

        # No substitution happened, so there is nothing to disclose. A notice
        # here would be a false warning.
        assert out.llm_notice == ""
        assert out.portfolio_summary == "Model summary."
