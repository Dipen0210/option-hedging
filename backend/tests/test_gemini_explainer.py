"""
The Gemini provider: refusals, capacity failures, and the empty-answer traps.

Written alongside the provider, because a second provider is where the defect
behind `test_llm_provider_refusal.py` gets a second chance to happen. That
defect was a permanent failure reported to the user as "the language model
returned no analysis for this run". Gemini offers three fresh ways to repeat
it, and each one is pinned here:

  1. An invalid key answers **400 INVALID_ARGUMENT / API_KEY_INVALID**, not
     401. A classifier keyed on {401, 402, 403} would call that an ordinary
     failure and tell the user to retry forever.
  2. A retired model answers **404**. `gemini-2.5-flash` really does — see
     `RETIRED_MODEL_MESSAGE`, captured from this project's first live call.
  3. `thinkingConfig.thinkingBudget` left unset lets reasoning tokens consume
     `maxOutputTokens`, so the API returns `finishReason: MAX_TOKENS` with no
     text — a successful HTTP 200 carrying nothing.

A 400 is also the shape of a malformed request of our own making, and that is
*not* a refusal: reporting our own bug as "the provider refused your
credentials" would be a different lie. So the ambiguous statuses are decided
by wording, and the "our own bug" case is asserted in both directions.

No network in this file: the `requests.Session` is stubbed, which also lets
the request itself be inspected — the API key must travel in a header and
never in a URL, because `requests` copies the full URL into the text of every
error it raises.
"""
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backend.core.explainer.base_explainer import (
    LLMProviderRejectedError,
    LLMRateLimitError,
)
from backend.core.explainer import gemini_explainer as gx
from backend.core.explainer.gemini_explainer import GeminiExplainer
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

# Deliberately too short to match a credential-shaped scanner pattern
# (`AIza` + 30 chars), so a stub in a test cannot become a false positive.
STUB_KEY = "AIza_STUB_not_a_key"

# Verbatim from the first live call this project made, with the default model.
RETIRED_MODEL_MESSAGE = (
    "This model models/gemini-2.5-flash is no longer available to new users. "
    "Please update your code to use models/gemini-3.8-flash for the latest "
    "features and improvements. We recommend you to use the Interactions API "
    "(https://ai.google.dev/gemini-api/docs/get-started)."
)

# Google's wording for an invalid key, which arrives as a 400.
BAD_KEY_MESSAGE = "API key not valid. Please pass a valid API key."

# A malformed request of *our* making. Also a 400; not a refusal.
OUR_BUG_MESSAGE = (
    'Invalid JSON payload received. Unknown name "generatonConfig" at '
    "'generaton_config': Cannot find field."
)


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    """Replays a scripted list of responses and records every request."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append(
            {"url": url, "headers": headers or {}, "body": json, "timeout": timeout}
        )
        if not self._responses:
            raise AssertionError("more requests than the test scripted")
        return self._responses.pop(0)


def _error_resp(status, api_message):
    return _Resp(status, {"error": {"code": status, "message": api_message,
                                    "status": "INVALID_ARGUMENT"}})


def _ok_resp(text, finish="STOP"):
    return _Resp(200, {"candidates": [
        {"finishReason": finish, "content": {"parts": [{"text": text}]}}
    ]})


def _explainer(*responses, model="gemini-3.5-flash"):
    ex = GeminiExplainer(model=model, api_key=STUB_KEY)
    session = _Session(*responses)
    ex._session = session
    return ex, session


# A response in the shape COMBINED_SYSTEM asks for.
def _combined_body(ticker="AAPL", strategy="Direct Put (3% OTM)"):
    return json.dumps({
        "positions": {ticker: {strategy: {
            "when_works_best": "A sharp drawdown before expiry.",
            "when_fails": "A flat tape, where theta decay dominates.",
            "rationale": "Long put at the 335 strike, -36.4 delta per contract.",
            "pros": ["zero basis risk"], "cons": ["daily theta decay"],
        }}},
        "portfolio": {
            "summary": "Notional $333,727 with portfolio beta 0.731.",
            "key_risks": ["GLD carries a 29.3% VaR"],
            "regime_commentary": "VIX 19.4, high_vol regime.",
            "top_recommendation": "Buy the GLD protective put.",
        },
    })


# ── The request itself ────────────────────────────────────────────────────────

class TestTheRequestWeSend:

    def test_the_api_key_never_appears_in_the_url(self):
        """
        `requests` puts the full URL into every HTTPError it raises, so a key
        in a `?key=` query string would be copied into exception text, logs
        and anything that serialises them.
        """
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        req = session.requests[0]
        assert STUB_KEY not in req["url"]
        assert "key=" not in req["url"]
        assert req["headers"]["x-goog-api-key"] == STUB_KEY
        assert STUB_KEY not in json.dumps(req["body"])

    def test_reasoning_tokens_cannot_eat_the_output_budget(self):
        """
        Gemini 2.5+ charges its own reasoning against maxOutputTokens, so an
        unset budget can return finishReason=MAX_TOKENS with no text at all —
        a 200 that reaches the user as "the model returned no analysis".
        """
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        cfg = session.requests[0]["body"]["generationConfig"]
        assert cfg["thinkingConfig"]["thinkingBudget"] == 0

    def test_the_api_is_asked_to_enforce_json(self):
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        cfg = session.requests[0]["body"]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["maxOutputTokens"] == gx.COMBINED_MAX_TOKENS

    def test_the_call_is_bounded_by_a_timeout(self):
        """An optional narrative must not be what holds a hedge request open."""
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        assert session.requests[0]["timeout"] == gx.REQUEST_TIMEOUT_SECONDS

    def test_the_model_is_named_in_the_path_and_the_prompt_is_a_system_instruction(self):
        ex, session = _explainer(_ok_resp(_combined_body()), model="gemini-3.5-flash")
        ex.explain_all({"positions": [], "portfolio": {}})
        req = session.requests[0]
        assert req["url"].endswith("/models/gemini-3.5-flash:generateContent")
        assert req["body"]["systemInstruction"]["parts"][0]["text"]

    def test_a_good_response_is_parsed_into_explanations(self):
        ctx = {"positions": [{"asset_ticker": "AAPL",
                              "candidates": [{"strategy": "Direct Put (3% OTM)"}]}],
               "portfolio": {}}
        ex, _ = _explainer(_ok_resp(_combined_body()))
        result = ex.explain_all(ctx)
        exp = result.candidates["AAPL"]["Direct Put (3% OTM)"]
        assert exp.rationale and exp.when_fails and exp.pros and exp.cons
        assert result.portfolio.summary and result.portfolio.top_recommendation


# ── Refusals ──────────────────────────────────────────────────────────────────

class TestARefusalIsNotAnEmptyRun:

    METHODS = [
        ("explain_candidates", {"positions": []}),
        ("explain_portfolio", {"portfolio": {}}),
        ("explain_all", {"positions": [], "portfolio": {}}),
    ]

    @pytest.mark.parametrize("method,arg", METHODS)
    def test_a_400_invalid_key_is_a_permanent_refusal(self, method, arg):
        """Not 401. This is the whole reason 400 is examined at all."""
        ex, _ = _explainer(_error_resp(400, BAD_KEY_MESSAGE))
        with pytest.raises(LLMProviderRejectedError) as e:
            getattr(ex, method)(arg)
        assert "credentials were refused" in e.value.reason

    @pytest.mark.parametrize("method,arg", METHODS)
    def test_a_retired_model_is_a_permanent_refusal(self, method, arg):
        ex, _ = _explainer(_error_resp(404, RETIRED_MODEL_MESSAGE))
        with pytest.raises(LLMProviderRejectedError) as e:
            getattr(ex, method)(arg)
        assert "not available to this account" in e.value.reason

    @pytest.mark.parametrize("method,arg", METHODS)
    def test_our_own_malformed_request_is_not_called_a_refusal(self, method, arg):
        """
        The other direction, and it matters just as much: a bug of ours must
        degrade quietly, not accuse the user's credentials.
        """
        ex, _ = _explainer(_error_resp(400, OUR_BUG_MESSAGE))
        result = getattr(ex, method)(arg)
        assert not result.candidates if method == "explain_all" else True

    @pytest.mark.parametrize("method,arg", METHODS)
    def test_a_404_from_our_own_wrong_url_is_not_blamed_on_the_model(self, method, arg):
        """
        The mirror of the test above, and the reason 404 is decided by wording
        rather than by status alone: "that model is retired for your account"
        and "we called the wrong endpoint" arrive as the same status, and only
        the first is a refusal. Telling a user their model is unavailable when
        the real fault is our URL sends them to fix the wrong thing.
        """
        ex, _ = _explainer(_error_resp(
            404, "Requested entity was not available at v1beta/modles/typo:generateContent"))
        result = getattr(ex, method)(arg)
        assert not result.candidates if method == "explain_all" else True

    def test_a_429_is_a_rate_limit_not_a_rejection(self):
        ex, _ = _explainer(_error_resp(429, "You exceeded your current quota"))
        with pytest.raises(LLMRateLimitError):
            ex.explain_all({"positions": [], "portfolio": {}})

    def test_a_403_is_a_permission_refusal(self):
        ex, _ = _explainer(_error_resp(
            403, "Generative Language API has not been used in project 123 before or it is disabled"))
        with pytest.raises(LLMProviderRejectedError) as e:
            ex.explain_all({"positions": [], "portfolio": {}})
        assert "not permitted" in e.value.reason

    def test_the_reason_shown_to_users_carries_no_provider_text(self):
        """
        The live 404 body names an internal model path and a documentation
        URL. The reason is built from the status, so none of it can travel.
        """
        ex, _ = _explainer(_error_resp(404, RETIRED_MODEL_MESSAGE))
        with pytest.raises(LLMProviderRejectedError) as e:
            ex.explain_all({"positions": [], "portfolio": {}})
        for leak in ("https://", "models/", "Interactions API", "gemini-"):
            assert leak not in e.value.reason

    def test_no_key_configured_is_not_a_refusal(self):
        ex = GeminiExplainer(model="gemini-3.5-flash", api_key="")
        assert not ex.is_available()
        assert ex.explain_candidates({"positions": []}) == []


# ── Capacity ──────────────────────────────────────────────────────────────────

class TestCapacityFailuresAreRetried:
    """
    503 "the model is currently overloaded" is what the free tier returns
    under load — five of eight calls during model selection. Without a retry,
    half of all hedge runs would lose their narrative to a condition that
    clears in a second.
    """

    @pytest.fixture(autouse=True)
    def _no_waiting(self, monkeypatch):
        self.slept = []
        monkeypatch.setattr(gx.time, "sleep", lambda s: self.slept.append(s))

    def test_a_503_is_retried_and_then_succeeds(self):
        ex, session = _explainer(
            _error_resp(503, "The model is currently overloaded"),
            _ok_resp(_combined_body()),
        )
        result = ex.explain_all({"positions": [], "portfolio": {}})
        assert result.portfolio.summary
        assert len(session.requests) == 2
        assert self.slept == [gx._BACKOFF_SECONDS[0]]

    def test_retries_are_bounded(self):
        """Three attempts, then degrade — not an unbounded hold on a request."""
        ex, session = _explainer(*[
            _error_resp(503, "The model is currently overloaded")
            for _ in range(gx._MAX_ATTEMPTS)
        ])
        result = ex.explain_all({"positions": [], "portfolio": {}})
        assert not result.candidates, "it degrades rather than raising"
        assert len(session.requests) == gx._MAX_ATTEMPTS
        assert sum(self.slept) <= 10

    def test_a_refusal_is_never_retried(self):
        """Retrying a dead key wastes the request budget to get the same answer."""
        ex, session = _explainer(_error_resp(400, BAD_KEY_MESSAGE))
        with pytest.raises(LLMProviderRejectedError):
            ex.explain_all({"positions": [], "portfolio": {}})
        assert len(session.requests) == 1
        assert self.slept == []

    def test_a_rate_limit_is_never_retried(self):
        ex, session = _explainer(_error_resp(429, "quota exceeded"))
        with pytest.raises(LLMRateLimitError):
            ex.explain_all({"positions": [], "portfolio": {}})
        assert len(session.requests) == 1


# ── A 200 that carries nothing ────────────────────────────────────────────────

class TestAnEmptyAnswerIsNotAnError:
    """
    These are blank results, not failures: the engine fills the narrative from
    the computed figures and discloses that it did. They must not raise, and
    they must not be mistaken for refusals.
    """

    def test_a_truncated_response_returns_empty(self):
        ex, _ = _explainer(_ok_resp("", finish="MAX_TOKENS"))
        assert ex.explain_all({"positions": [], "portfolio": {}}).candidates == {}

    def test_a_safety_block_returns_empty(self):
        ex, _ = _explainer(_Resp(200, {"promptFeedback": {"blockReason": "SAFETY"}}))
        assert ex.explain_candidates({"positions": []}) == []

    def test_no_candidates_returns_empty(self):
        ex, _ = _explainer(_Resp(200, {"candidates": []}))
        assert ex.explain_portfolio({"portfolio": {}}).summary == ""

    def test_unparseable_text_returns_empty_without_raising(self):
        ex, _ = _explainer(_ok_resp("I'm sorry, I can't help with that."))
        assert ex.explain_all({"positions": [], "portfolio": {}}).candidates == {}


# ── Through Layer 11, to the user ─────────────────────────────────────────────

class TestWhatTheUserIsTold:
    """
    End to end through the real explainer with only the HTTP session stubbed.
    The tests above would all still pass if nothing connected the classifier to
    Layer 11, which is exactly the gap defect 38 lived in.
    """

    def _run(self, monkeypatch, *responses):
        ex, _ = _explainer(*responses)
        monkeypatch.setattr(l11, "get_explainer", lambda provider=None: ex)
        return FinalLLMExplainerEngine().run(
            hedge_output=_make_hedge_output(),
            portfolio=_make_portfolio(),
            risk_summary=_make_portfolio_risk_summary(),
            regime=_make_regime(),
        )

    def test_a_retired_model_reaches_the_user_as_permanent(self, monkeypatch):
        out = self._run(monkeypatch, _error_resp(404, RETIRED_MODEL_MESSAGE))
        assert "not a temporary problem" in out.llm_notice
        assert "not available to this account" in out.llm_notice
        assert "returned no analysis for this run" not in out.llm_notice
        assert out.llm_notice == _REJECTED_NOTICE.format(
            provider="gemini", reason="the configured model is not available to this account")

    def test_an_invalid_key_reaches_the_user_as_permanent(self, monkeypatch):
        out = self._run(monkeypatch, _error_resp(400, BAD_KEY_MESSAGE))
        assert "not a temporary problem" in out.llm_notice
        assert "credentials were refused" in out.llm_notice

    def test_a_rate_limit_stays_transient(self, monkeypatch):
        out = self._run(monkeypatch, _error_resp(429, "quota exceeded"))
        assert out.llm_notice == _RATE_LIMIT_NOTICE
        assert "not a temporary problem" not in out.llm_notice

    def test_our_own_bug_is_not_blamed_on_the_credentials(self, monkeypatch):
        out = self._run(monkeypatch, _error_resp(400, OUR_BUG_MESSAGE))
        assert "credentials were refused" not in out.llm_notice
        assert "not a temporary problem" not in out.llm_notice

    def test_a_truncated_answer_is_reported_as_an_empty_run(self, monkeypatch):
        """This one really *is* per-run wording — the honest use of it."""
        out = self._run(monkeypatch, _ok_resp("", finish="MAX_TOKENS"))
        assert out.llm_notice == _SUBSTITUTED_NOTICE

    @pytest.mark.parametrize("responses,label", [
        ((_error_resp(404, RETIRED_MODEL_MESSAGE),), "a retired model"),
        ((_error_resp(400, BAD_KEY_MESSAGE),), "an invalid key"),
        ((_ok_resp("", finish="MAX_TOKENS"),), "an empty answer"),
    ])
    def test_the_hedge_analysis_survives_every_llm_failure(self, monkeypatch, responses, label):
        out = self._run(monkeypatch, *responses)
        assert out.portfolio_summary, f"{label} must not cost the user their analysis"
        assert out.key_risks and out.regime_commentary and out.top_recommendation
        assert out.recommendations

    def test_the_notice_leaks_no_internals(self, monkeypatch):
        out = self._run(monkeypatch, _error_resp(404, RETIRED_MODEL_MESSAGE))
        for leak in ("Traceback", ".py", "https://", "models/", STUB_KEY):
            assert leak not in out.llm_notice
