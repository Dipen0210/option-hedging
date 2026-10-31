"""
Regression tests: the product's disclosures survive an LLM rewrite.

The defect these cover was live in the default configuration. Three facts
that the selector asserts about a candidate — no executable quote, thin
market, expiry crosses earnings — were kept in `candidate.cons`, and both
explainer layers merge an LLM response by replacing that list outright:

    if explanation.cons:
        candidate.cons = explanation.cons

With `run_l11_llm = True` by default, any model that returned its own bullets
deleted "premium shown is theoretical" on the way past, and the API answered
with a model-derived premium and nothing marking it as one. Every test below
asserts against the real merge functions the pipeline calls, not a stand-in.
"""

from backend.core import disclosures
from backend.models.hedge_models import InstrumentCandidate
from backend.engines.layer_11_llm_explainer import _merge_explanation
from backend.engines.layer_05_llm_explainer import _apply_explanation
from backend.core.explainer.base_explainer import CandidateExplanation


NO_QUOTE = ("No live two-sided quote — premium shown is theoretical, "
            "not an executable price")


def _candidate(**kw) -> InstrumentCandidate:
    base = dict(
        instrument_type="option", asset_class="equity", ticker="SPY",
        strategy="protective_put", hedge_category="direct_hedge",
    )
    base.update(kw)
    return InstrumentCandidate(**base)


def _llm(**kw) -> CandidateExplanation:
    base = dict(ticker="SPY", strategy="protective_put", asset_ticker="SPY")
    base.update(kw)
    return CandidateExplanation(**base)


class TestDiscloseRecordsInBothPlaces:

    def test_disclosure_shows_in_cons_so_the_ui_is_unchanged(self):
        c = _candidate(cons=["Theta decay erodes value daily"])
        disclosures.disclose(c, NO_QUOTE)
        assert NO_QUOTE in c.cons
        # and it did not displace what was already there
        assert "Theta decay erodes value daily" in c.cons

    def test_disclosure_is_also_recorded_out_of_the_llm_s_reach(self):
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        assert c.mandatory_disclosures == [NO_QUOTE]

    def test_ordinary_cons_are_not_recorded_as_disclosures(self):
        """A caveat is not a disclosure; the LLM may rewrite caveats freely."""
        c = _candidate(cons=["Theta decay erodes value daily"])
        assert c.mandatory_disclosures == []

    def test_disclosing_twice_does_not_say_it_twice(self):
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        disclosures.disclose(c, NO_QUOTE)
        assert c.cons.count(NO_QUOTE) == 1
        assert c.mandatory_disclosures.count(NO_QUOTE) == 1


class TestL11CannotDeleteADisclosure:
    """L11 runs by default — this is the path that shipped broken."""

    def test_an_llm_rewrite_of_cons_does_not_remove_the_quote_disclosure(self):
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _merge_explanation(c, _llm(cons=["Costs money", "Expires worthless if flat"]))
        assert NO_QUOTE in c.cons

    def test_the_llm_s_own_cons_are_still_kept(self):
        """The fix must not throw the explanation away to protect the disclosure."""
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _merge_explanation(c, _llm(cons=["Costs money"]))
        assert "Costs money" in c.cons

    def test_the_disclosure_leads_rather_than_trailing_the_prose(self):
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _merge_explanation(c, _llm(cons=["a", "b", "c", "d"]))
        assert c.cons[0] == NO_QUOTE

    def test_every_disclosure_survives_not_just_the_first(self):
        c = _candidate()
        earnings = "Expiry crosses earnings (2026-11-05) — IV crush risk"
        thin = "Low liquidity: OI 12 — verify before trading"
        for d in (NO_QUOTE, earnings, thin):
            disclosures.disclose(c, d)
        _merge_explanation(c, _llm(cons=["Rewritten"]))
        assert set(c.mandatory_disclosures) <= set(c.cons)

    def test_an_llm_that_echoes_a_disclosure_does_not_double_it(self):
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _merge_explanation(c, _llm(cons=[NO_QUOTE, "Costs money"]))
        assert c.cons.count(NO_QUOTE) == 1

    def test_a_candidate_with_no_disclosures_is_left_exactly_as_the_llm_wrote_it(self):
        c = _candidate(cons=["Theta decay"])
        _merge_explanation(c, _llm(cons=["Costs money"]))
        assert c.cons == ["Costs money"]

    def test_an_empty_llm_cons_leaves_the_disclosure_in_place(self):
        """`if explanation.cons` skips the write; nothing should change."""
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _merge_explanation(c, _llm(cons=[]))
        assert c.cons == [NO_QUOTE]


class TestL5CannotDeleteADisclosure:
    """Off by default (`run_l5_prelim_llm=False`) but reachable by config."""

    def test_an_llm_rewrite_of_cons_does_not_remove_the_quote_disclosure(self):
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _apply_explanation(c, _llm(cons=["Costs money"]))
        assert NO_QUOTE in c.cons
        assert "Costs money" in c.cons

    def test_both_layers_in_sequence_still_keep_it(self):
        """L5 then L11, which is what `run_l5_prelim_llm=True` produces."""
        c = _candidate()
        disclosures.disclose(c, NO_QUOTE)
        _apply_explanation(c, _llm(cons=["First pass"]))
        _merge_explanation(c, _llm(cons=["Second pass"]))
        assert NO_QUOTE in c.cons
        assert "Second pass" in c.cons


class TestTheSelectorStillProducesTheDisclosures:
    """
    Guard on the guard: the restore is worthless if the producers stopped
    recording. These assert the real selector paths, not `disclose` directly.
    """

    def test_the_unquoted_path_records_the_disclosure_as_mandatory(self):
        from backend.instruments.options.selector import OptionsSelector
        c = _candidate(premium=4.2, premium_source="theoretical")
        out = OptionsSelector._filter_executable([c], ticker="SPY")
        assert out, "the unquoted path must keep the candidate, not drop it"
        assert any("theoretical" in d for d in out[0].mandatory_disclosures)

    def test_that_disclosure_then_survives_the_default_llm_path(self):
        """End to end over the two real functions: selector → L11 merge."""
        from backend.instruments.options.selector import OptionsSelector
        c = _candidate(premium=4.2, premium_source="theoretical")
        c, = OptionsSelector._filter_executable([c], ticker="SPY")
        _merge_explanation(c, _llm(cons=["Premium is modest", "Theta bleeds"]))
        assert any("theoretical" in x for x in c.cons), c.cons


class TestTheExplainerFactoryDoesNotSubstituteSilently:
    """
    A second silent substitution found alongside the one above.

    `get_explainer` returned ClaudeExplainer for every provider name it did
    not recognise — and it recognised only "huggingface", so
    `llm_provider=ollama` sent the portfolio to Anthropic with nothing in the
    logs to say the configured provider had been ignored. OllamaExplainer was
    unreachable through the factory, and would have raised AttributeError if
    reached, because `settings.ollama_model` and `settings.ollama_base_url`
    were referenced but never declared.
    """

    def test_the_ollama_provider_is_reachable(self):
        from backend.core.explainer import get_explainer
        from backend.core.explainer.ollama_explainer import OllamaExplainer
        assert isinstance(get_explainer("ollama"), OllamaExplainer)

    def test_constructing_it_does_not_raise_on_missing_settings(self):
        from backend.core.explainer.ollama_explainer import OllamaExplainer
        e = OllamaExplainer()
        assert e._base_url and e._model

    def test_an_unknown_provider_name_is_logged_not_swallowed(self, caplog):
        from backend.core.explainer import get_explainer
        from backend.core.explainer.claude_explainer import ClaudeExplainer
        with caplog.at_level("ERROR", logger="backend.core.explainer"):
            e = get_explainer("gpt-4")
        assert isinstance(e, ClaudeExplainer), "fallback itself is still correct"
        assert any("gpt-4" in r.getMessage() for r in caplog.records), caplog.text

    def test_a_known_provider_logs_nothing(self, caplog):
        from backend.core.explainer import get_explainer
        with caplog.at_level("ERROR", logger="backend.core.explainer"):
            get_explainer("claude")
            get_explainer("huggingface")
        assert not caplog.records, caplog.text
