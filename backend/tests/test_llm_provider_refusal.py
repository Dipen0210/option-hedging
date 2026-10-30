"""
A provider that refuses permanently must not be reported as an empty run.

The defect: HuggingFace answered every explanation call with HTTP 402 "You have
no remaining credits". `_is_rate_limit` only looked for "429", "rate limit",
"too many requests" and "quota" in the message, matched none of them, so the
explainer logged the error and returned an empty result. Layer 11 saw a
successful call that produced nothing, filled the narrative from the computed
figures, and attached:

    "The language model returned no analysis for this run, so the narrative
     below was generated directly from the computed figures instead."

Every word of which is wrong about the cause. The account had no credits; it
would fail identically on the next run and every run after it, and only an
operator topping up an account or changing a key could fix it. "for this run"
told the user to retry, and a retry could never work.

Classification is by HTTP status, and the message is consulted only when there
is no status — because these SDKs append a request ID and a documentation URL
to the message, so digits like "429" appear in the text of a 402.

No network in this file: the exceptions are built to the shape the real SDKs
produce (confirmed against a live HuggingFace 402: `HfHubHTTPError` with
`.response.status_code == 402`), and the clients are stubbed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backend.core.explainer.base_explainer import (
    LLMProviderRejectedError,
    LLMRateLimitError,
    raise_if_provider_refused,
)
from backend.core.explainer.claude_explainer import ClaudeExplainer
from backend.core.explainer.huggingface_explainer import (
    HFRateLimitError,
    HuggingFaceExplainer,
)
from backend.engines import layer_11_llm_explainer as l11
from backend.engines.layer_11_llm_explainer import (
    FinalLLMExplainerEngine,
    _RATE_LIMIT_NOTICE,
    _REJECTED_NOTICE,
    _SUBSTITUTED_NOTICE,
)
from backend.tests.test_explainer import (
    _make_hedge_output,
    _make_portfolio,
    _make_portfolio_risk_summary,
    _make_regime,
)


class _Response:
    """The `.response` attribute huggingface_hub hangs off its HTTP errors."""

    def __init__(self, status_code):
        self.status_code = status_code


def _hf_error(status, message=""):
    """An exception shaped like huggingface_hub's, status on `.response`."""
    exc = OSError(message)
    exc.response = _Response(status)
    return exc


def _anthropic_error(status, message=""):
    """An exception shaped like the anthropic SDK's, status on the exception."""
    exc = RuntimeError(message)
    exc.status_code = status
    return exc


# The real body, verbatim apart from the credit figures, including the request
# ID and the MDN link the SDK appends. This is the input the old classifier saw.
REAL_402_MESSAGE = (
    "Client error '402 Payment Required' for url "
    "'https://router.huggingface.co/v1/chat/completions' "
    "(Request ID: Root=1-6ae50969-1258ffd3769c9558594aac62;"
    "b98f0dfc-c49f-409f-8b10-f6372fc836b5)\n"
    "For more information check: "
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/402\n\n"
    "You have no remaining credits."
)


class TestTheStatusDecides:

    @pytest.mark.parametrize("status, fragment", [
        (401, "credentials were refused"),
        (402, "no remaining credits"),
        (403, "not permitted to use this model"),
    ])
    def test_a_permanent_status_raises_a_rejection(self, status, fragment):
        with pytest.raises(LLMProviderRejectedError) as caught:
            raise_if_provider_refused(_hf_error(status), "huggingface")
        assert fragment in caught.value.reason
        assert caught.value.provider == "huggingface"

    def test_429_raises_a_rate_limit_not_a_rejection(self):
        with pytest.raises(LLMRateLimitError) as caught:
            raise_if_provider_refused(_hf_error(429), "huggingface")
        assert not isinstance(caught.value, LLMProviderRejectedError)

    def test_the_real_402_is_a_rejection_despite_its_message(self):
        """
        The regression. This message contains "402" and — in the request ID —
        the digits "429" are not present by luck, so the next test forces them.
        What matters here is that the old keyword list found nothing in it and
        therefore classified a dead account as an ordinary failure.
        """
        with pytest.raises(LLMProviderRejectedError):
            raise_if_provider_refused(_hf_error(402, REAL_402_MESSAGE), "huggingface")

    def test_a_402_whose_text_reads_like_a_rate_limit_is_still_permanent(self):
        """
        A request ID is opaque hex and may contain "429"; provider bodies also
        use the word "quota" for billing. Either would have made the old
        substring matcher call this transient and tell the user to retry.
        """
        misleading = (
            "Client error '402 Payment Required' (Request ID: Root=1-429abc) "
            "quota exhausted: you have no remaining credits"
        )
        with pytest.raises(LLMProviderRejectedError):
            raise_if_provider_refused(_hf_error(402, misleading), "huggingface")

    def test_an_ordinary_server_error_is_left_alone(self):
        """A 500 is a real failure, not a refusal: the caller degrades."""
        raise_if_provider_refused(_hf_error(500, "internal error"), "huggingface")
        raise_if_provider_refused(_hf_error(503, "unavailable"), "huggingface")

    def test_a_status_on_the_exception_itself_is_found(self):
        """The anthropic SDK does not wrap a response object."""
        with pytest.raises(LLMProviderRejectedError):
            raise_if_provider_refused(_anthropic_error(401, "bad key"), "claude")

    def test_wording_is_used_only_when_there_is_no_status(self):
        with pytest.raises(LLMProviderRejectedError):
            raise_if_provider_refused(Exception("You have no remaining credits"), "x")
        with pytest.raises(LLMRateLimitError):
            raise_if_provider_refused(Exception("rate limit exceeded"), "x")
        # Nothing recognisable, no status: an ordinary failure.
        raise_if_provider_refused(Exception("connection reset"), "x")

    def test_a_bare_429_in_text_is_not_enough_to_claim_a_rate_limit(self):
        """
        Without a status there is no evidence a stray "429" is a status code
        rather than part of an ID, so it must not drive the classification.
        """
        raise_if_provider_refused(Exception("Request ID: Root=1-429abcdef"), "x")

    def test_the_reason_shown_to_users_carries_no_internal_detail(self):
        """
        Per the repair spec: no internal exception detail in public responses.
        `reason` is built from the status, so none of the request ID, the
        internal router URL or the MDN link can reach the response body.
        """
        with pytest.raises(LLMProviderRejectedError) as caught:
            raise_if_provider_refused(_hf_error(402, REAL_402_MESSAGE), "huggingface")
        reason = caught.value.reason
        for leak in ("Request ID", "Root=", "router.huggingface.co",
                     "developer.mozilla.org", "402", "Traceback"):
            assert leak not in reason, f"{leak!r} leaked into a user-facing reason"


class TestTheProvidersRaiseRatherThanReturnEmpty:
    """
    The classifier is only useful if the call sites consult it. Before this
    fix all three HuggingFace methods and both Claude methods returned an
    empty result for a 402.
    """

    class _Raises:
        def __init__(self, exc):
            self._exc = exc

        def chat_completion(self, **kw):      # huggingface_hub
            raise self._exc

        @property
        def messages(self):                   # anthropic
            return self

        def create(self, **kw):
            raise self._exc

    def _hf(self, monkeypatch, exc):
        ex = HuggingFaceExplainer(api_token="hf_stub")
        monkeypatch.setattr(ex, "_get_client", lambda: self._Raises(exc))
        return ex

    def _claude(self, monkeypatch, exc):
        ex = ClaudeExplainer(api_key="sk-stub")
        monkeypatch.setattr(ex, "_get_client", lambda: self._Raises(exc))
        return ex

    @pytest.mark.parametrize("method, arg", [
        ("explain_candidates", {}),
        ("explain_portfolio", {}),
        ("explain_all", {}),
    ])
    def test_huggingface_raises_on_a_402(self, monkeypatch, method, arg):
        ex = self._hf(monkeypatch, _hf_error(402, REAL_402_MESSAGE))
        with pytest.raises(LLMProviderRejectedError):
            getattr(ex, method)(arg)

    @pytest.mark.parametrize("method", ["explain_candidates", "explain_portfolio"])
    def test_claude_raises_on_a_refused_key(self, monkeypatch, method):
        ex = self._claude(monkeypatch, _anthropic_error(401, "invalid x-api-key"))
        with pytest.raises(LLMProviderRejectedError):
            getattr(ex, method)({})

    def test_an_ordinary_failure_still_degrades_quietly(self, monkeypatch):
        """The fix must not turn every provider hiccup into a failed hedge run."""
        ex = self._hf(monkeypatch, _hf_error(500, "internal error"))
        assert ex.explain_candidates({}) == []
        assert ex.explain_portfolio({}).summary == ""

    def test_a_429_still_surfaces_as_a_rate_limit(self, monkeypatch):
        ex = self._hf(monkeypatch, _hf_error(429, "too many requests"))
        with pytest.raises(LLMRateLimitError):
            ex.explain_all({})

    def test_the_legacy_hf_error_still_satisfies_callers_that_catch_it(self):
        assert issubclass(HFRateLimitError, LLMRateLimitError)


class TestWhatLayer11TellsTheUser:

    class _Stub:
        provider_name = "huggingface"

        def __init__(self, exc):
            self._exc = exc

        def is_available(self):
            return True

        def supports_combined_call(self):
            return True

        def explain_all(self, ctx):
            raise self._exc

    def _run(self, monkeypatch, exc):
        monkeypatch.setattr(l11, "get_explainer", lambda provider=None: self._Stub(exc))
        return FinalLLMExplainerEngine().run(
            hedge_output=_make_hedge_output(),
            portfolio=_make_portfolio(),
            risk_summary=_make_portfolio_risk_summary(),
            regime=_make_regime(),
        )

    def _rejected(self, monkeypatch):
        return self._run(
            monkeypatch,
            LLMProviderRejectedError("huggingface", "the provider account has no remaining credits"),
        )

    def test_the_notice_is_not_the_per_run_wording(self, monkeypatch):
        """The whole point: this is what the user used to be told."""
        out = self._rejected(monkeypatch)
        assert out.llm_notice != _SUBSTITUTED_NOTICE
        assert "returned no analysis for this run" not in out.llm_notice
        assert out.llm_notice != _RATE_LIMIT_NOTICE

    def test_the_notice_says_it_is_permanent_and_who_refused(self, monkeypatch):
        out = self._rejected(monkeypatch)
        assert "not a temporary problem" in out.llm_notice
        assert "huggingface" in out.llm_notice
        assert "no remaining credits" in out.llm_notice
        assert out.llm_notice == _REJECTED_NOTICE.format(
            provider="huggingface",
            reason="the provider account has no remaining credits",
        )

    def test_the_hedge_run_still_completes_with_a_full_narrative(self, monkeypatch):
        """A dead LLM account must not cost the user their hedge analysis."""
        out = self._rejected(monkeypatch)
        assert out.portfolio_summary
        assert out.key_risks
        assert out.regime_commentary
        assert out.top_recommendation
        assert out.recommendations

    def test_the_notice_leaks_no_internals(self, monkeypatch):
        out = self._rejected(monkeypatch)
        for leak in ("Traceback", ".py", "Root=", "http://", "https://"):
            assert leak not in out.llm_notice

    def test_a_real_402_from_the_client_reaches_the_user_as_permanent(self, monkeypatch):
        """
        End to end through the real explainer, not an injected exception: the
        only stub is the HTTP client. This is the path that actually ran in
        production and produced the wrong notice, so it is the one that has to
        be pinned — the tests above would still pass if nothing connected the
        classifier to Layer 11.
        """
        explainer = HuggingFaceExplainer(api_token="hf_stub")
        monkeypatch.setattr(
            explainer, "_get_client",
            lambda: TestTheProvidersRaiseRatherThanReturnEmpty._Raises(
                _hf_error(402, REAL_402_MESSAGE)
            ),
        )
        monkeypatch.setattr(l11, "get_explainer", lambda provider=None: explainer)
        out = FinalLLMExplainerEngine().run(
            hedge_output=_make_hedge_output(),
            portfolio=_make_portfolio(),
            risk_summary=_make_portfolio_risk_summary(),
            regime=_make_regime(),
        )
        assert "not a temporary problem" in out.llm_notice
        assert "no remaining credits" in out.llm_notice
        assert "returned no analysis for this run" not in out.llm_notice
        assert out.portfolio_summary, "the hedge analysis must still be delivered"

    def test_a_rate_limit_is_still_reported_as_transient(self, monkeypatch):
        """The two must stay distinguishable in both directions."""
        out = self._run(monkeypatch, HFRateLimitError("slow down"))
        assert out.llm_notice == _RATE_LIMIT_NOTICE
        assert "not a temporary problem" not in out.llm_notice
