"""
The Vertex AI provider: the endpoint, the credential, and the 404 that is ours.

Vertex is the configured provider, and it is a subclass rather than a second
copy of `GeminiExplainer` — so what needs pinning here is the pair of things
it changes, plus proof that the inherited defences still run on this path.

Three traps are specific to Vertex and each one is asserted:

  1. **The location is not free.** `gemini-3.5-flash` is served from `global`
     only; every regional endpoint answers it `404 NOT_FOUND`. The host for
     `global` is also spelled differently — unprefixed — so a naive
     `f"{loc}-aiplatform..."` produces a URL that resolves to nothing.
  2. **Two different 404s.** An unavailable model and a URL we malformed
     ourselves share a status. Vertex's model error says "Publisher model
     ... was not found or your project does not have access to it"; a bad URL
     gets Google's generic frontend page instead. The first is a refusal, the
     second is our bug, and calling our bug a credential problem is its own
     misreport. Both directions are asserted.
  3. **A missing credential is not an empty run.** There is no API key to find
     absent — ADC either resolves or raises. If it raises, the user must be
     told the provider was refused, not that "the model returned no analysis".

The stubs are imported from the Gemini tests rather than rewritten: these two
providers must not drift apart in what a test considers a well-formed
response.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from backend.core.explainer import get_explainer
from backend.core.explainer.base_explainer import (
    LLMProviderRejectedError,
    LLMRateLimitError,
)
from backend.core.explainer import gemini_explainer as gx
from backend.core.explainer import vertex_explainer as vx
from backend.core.explainer.vertex_explainer import VertexExplainer
from backend.tests.test_gemini_explainer import (
    _Resp,
    _Session,
    _combined_body,
    _error_resp,
    _ok_resp,
)

PROJECT = "optionq-test-project"

# Verbatim from this project's live call to a regional endpoint. The wording is
# load-bearing: it is what the classifier keys on.
UNAVAILABLE_MODEL_MESSAGE = (
    "Publisher model `projects/optionq-test-project/locations/us-central1/"
    "publishers/google/models/gemini-3.5-flash` was not found or your project "
    "does not have access to it. Ensure you are using a valid model name and "
    "that the model is available in the specified region."
)

# What Vertex actually returns for a URL we built wrong — Google's generic
# frontend 404, as HTML, with no JSON body at all. Captured live by sending a
# malformed path. Not a refusal.
OUR_BAD_URL_PAGE = (
    "<!DOCTYPE html><html lang=en><title>Error 404 (Not Found)!!1</title>"
    "<p><b>404.</b> <ins>That’s an error.</ins><p>The requested URL "
    "<code>/v1/projects/optionq-test-project/locations/global/publishers/"
    "google/models/5-flashnerateContent</code> was not found on this server."
)

# Vertex's wording when the caller's service account lacks the role.
PERMISSION_MESSAGE = (
    "Permission 'aiplatform.endpoints.predict' denied on resource "
    "'//aiplatform.googleapis.com/projects/optionq-test-project/locations/"
    "global/publishers/google/models/gemini-3.5-flash'."
)


def _explainer(*responses, model="gemini-3.5-flash", location="global",
               project=PROJECT):
    ex = VertexExplainer(model=model, project=project, location=location)
    session = _Session(*responses)
    ex._session = session
    return ex, session


# ── The endpoint ──────────────────────────────────────────────────────────────

class TestTheEndpointWeCall:

    def test_the_global_location_uses_the_unprefixed_host(self):
        """
        Measured, not guessed: the 3.x Gemini publisher models are served from
        `global`, whose host has no region prefix. `global-aiplatform...`
        does not resolve.
        """
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        url = session.requests[0]["url"]
        assert url == (
            "https://aiplatform.googleapis.com/v1/projects/optionq-test-project"
            "/locations/global/publishers/google/models/"
            "gemini-3.5-flash:generateContent"
        )

    def test_a_regional_location_uses_the_prefixed_host(self):
        ex, session = _explainer(_ok_resp(_combined_body()),
                                 location="us-central1")
        ex.explain_all({"positions": [], "portfolio": {}})
        assert session.requests[0]["url"] == (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/"
            "optionq-test-project/locations/us-central1/publishers/google/"
            "models/gemini-3.5-flash:generateContent"
        )

    def test_the_model_name_survives_into_the_path_intact(self):
        """
        The model name contains a dot and is followed by `:generateContent`.
        Built in a shell, `$model:generateContent` is read as zsh history
        modifiers (`:g` + `:e`) and silently becomes `5-flashnerateContent` —
        which is how four rounds of probing this API produced nothing but
        generic 404s. Python has no such trap, but the assertion costs nothing
        and names the failure if the URL is ever assembled elsewhere.
        """
        ex, session = _explainer(_ok_resp(_combined_body()),
                                 model="gemini-3.5-flash")
        ex.explain_all({"positions": [], "portfolio": {}})
        assert session.requests[0]["url"].endswith(
            "/models/gemini-3.5-flash:generateContent")

    def test_no_api_key_header_is_sent(self):
        """
        Vertex authenticates with IAM. An `x-goog-api-key` header here would
        mean a key had been carried over from the other provider's
        configuration — the whole point of this path is that there is no
        long-lived secret.
        """
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        headers = session.requests[0]["headers"]
        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_the_project_is_never_put_in_a_query_string(self):
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        assert "?" not in session.requests[0]["url"]


# ── The inherited defences must still run on this path ────────────────────────

class TestTheInheritedDefencesStillApply:
    """
    The reason Vertex is a subclass. Each of these was a defect fixed once on
    the Generative Language path; a copied class would let the fix apply to
    one surface and silently miss the other.
    """

    def test_the_thinking_budget_is_still_zero(self):
        """
        Unset, reasoning tokens are charged against maxOutputTokens and the
        call returns `finishReason: MAX_TOKENS` with no text — a successful
        HTTP 200 carrying nothing, which reaches the user as "no analysis".
        """
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        cfg = session.requests[0]["body"]["generationConfig"]
        assert cfg["thinkingConfig"]["thinkingBudget"] == 0
        assert cfg["responseMimeType"] == "application/json"

    def test_a_503_is_still_retried(self, monkeypatch):
        slept = []
        monkeypatch.setattr(gx.time, "sleep", lambda s: slept.append(s))
        ex, session = _explainer(
            _error_resp(503, "The model is currently overloaded"),
            _ok_resp(_combined_body()),
        )
        result = ex.explain_all({"positions": [], "portfolio": {}})
        assert result.portfolio.summary
        assert len(session.requests) == 2
        assert slept == [gx._BACKOFF_SECONDS[0]]

    def test_a_429_is_a_rate_limit_not_a_refusal(self):
        ex, _ = _explainer(_error_resp(429, "Quota exceeded"))
        with pytest.raises(LLMRateLimitError):
            ex.explain_all({"positions": [], "portfolio": {}})

    def test_an_empty_but_successful_response_is_not_an_error(self):
        ex, _ = _explainer(_Resp(200, {"candidates": [
            {"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}))
        result = ex.explain_all({"positions": [], "portfolio": {}})
        assert result.portfolio.summary == ""

    def test_the_timeout_is_set(self):
        ex, session = _explainer(_ok_resp(_combined_body()))
        ex.explain_all({"positions": [], "portfolio": {}})
        assert session.requests[0]["timeout"] == gx.REQUEST_TIMEOUT_SECONDS


# ── The two different 404s ────────────────────────────────────────────────────

class TestARefusalIsNotAnEmptyRun:

    def test_an_unavailable_publisher_model_is_a_refusal(self):
        """
        The case that actually happened: `gemini-3.5-flash` on us-central1.
        Permanent — no retry clears it, an operator has to change the location
        or the model — so it must not surface as an empty run.
        """
        ex, _ = _explainer(_error_resp(404, UNAVAILABLE_MODEL_MESSAGE))
        with pytest.raises(LLMProviderRejectedError) as e:
            ex.explain_all({"positions": [], "portfolio": {}})
        assert "model is not available" in e.value.reason

    def test_a_404_from_a_url_we_malformed_is_not_blamed_on_the_model(self):
        """
        The other direction, and the one that is easy to get wrong. Vertex
        answers a bad path with Google's generic HTML page, which has no JSON
        body and none of the publisher-model wording. That is our bug, and
        reporting it as "the configured model is not available to this
        account" would send an operator to fix something that is not broken.
        """
        ex, _ = _explainer(_Resp(404, None, text=OUR_BAD_URL_PAGE))
        result = ex.explain_all({"positions": [], "portfolio": {}})
        assert result.portfolio.summary == ""

    def test_the_html_404_body_never_reaches_the_user(self):
        """
        It is caught and logged, not raised — so nothing in it can be
        serialised into a response. Asserted because the page echoes the
        request path, which carries the project id.
        """
        ex, _ = _explainer(_Resp(404, None, text=OUR_BAD_URL_PAGE))
        result = ex.explain_candidates({"positions": []})
        assert result == []

    def test_a_permission_denied_is_a_refusal(self):
        ex, _ = _explainer(_error_resp(403, PERMISSION_MESSAGE))
        with pytest.raises(LLMProviderRejectedError) as e:
            ex.explain_all({"positions": [], "portfolio": {}})
        assert "not permitted" in e.value.reason


# ── The credential ────────────────────────────────────────────────────────────

class TestTheCredential:

    def test_absent_credentials_are_reported_as_a_refusal(self, monkeypatch):
        """
        Not an empty narrative. There is no key to check for emptiness on this
        path, so an unconfigured environment shows up only here — and the
        operator needs to be told, because `gcloud auth
        application-default login` is the fix and nothing else will do it.
        """
        def _boom(scopes=None):
            raise DefaultCredentialsError("Could not automatically determine credentials")
        monkeypatch.setattr(vx.google.auth, "default", _boom)

        ex = VertexExplainer(project=PROJECT)
        with pytest.raises(LLMProviderRejectedError) as e:
            ex.explain_all({"positions": [], "portfolio": {}})
        assert "no Google Cloud credentials" in e.value.reason

    def test_a_credential_that_cannot_be_signed_is_a_refusal(self, monkeypatch):
        def _boom(scopes=None):
            raise RefreshError("invalid_grant: token has been expired or revoked")
        monkeypatch.setattr(vx.google.auth, "default", _boom)

        ex = VertexExplainer(project=PROJECT)
        with pytest.raises(LLMProviderRejectedError) as e:
            ex.explain_candidates({"positions": []})
        assert "refused" in e.value.reason

    def test_quota_is_attributed_to_the_configured_project(self, monkeypatch):
        """
        Application Default Credentials carry whichever `quota_project_id` the
        developer last configured — on the machine this was built on, a
        different project entirely. `AuthorizedSession` turns that into an
        `x-goog-user-project` header, so without this override the calls would
        bill and count against a project unrelated to OptionQ.
        """
        class _Creds:
            def __init__(self):
                self.quota_project = None

            def with_quota_project(self, project):
                self.quota_project = project
                return self

        creds = _Creds()
        monkeypatch.setattr(vx.google.auth, "default", lambda scopes=None: (creds, "ambient-project"))
        monkeypatch.setattr(vx, "AuthorizedSession", lambda c: _Session())

        ex = VertexExplainer(project=PROJECT)
        ex._get_session()
        assert creds.quota_project == PROJECT

    def test_the_cloud_platform_scope_is_requested(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(vx.google.auth, "default",
                            lambda scopes=None: (seen.setdefault("scopes", scopes), None))
        monkeypatch.setattr(vx, "AuthorizedSession", lambda c: _Session())
        VertexExplainer(project=PROJECT)._get_session()
        assert seen["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]

    def test_the_session_is_built_once(self, monkeypatch):
        """A token mint per hedge run, not per call — three calls, one session."""
        built = []
        monkeypatch.setattr(vx.google.auth, "default", lambda scopes=None: (None, None))
        monkeypatch.setattr(vx, "AuthorizedSession",
                            lambda c: built.append(c) or _Session())
        ex = VertexExplainer(project=PROJECT)
        ex._get_session(); ex._get_session(); ex._get_session()
        assert len(built) == 1


# ── Configuration ─────────────────────────────────────────────────────────────

class TestConfiguration:

    def test_no_project_configured_is_not_a_refusal_and_calls_nothing(self):
        """
        An unconfigured provider is a disabled provider: the pipeline returns
        its deterministic narrative and discloses that it did. It is not a
        rejection, and it must not reach the network.
        """
        ex = VertexExplainer(project="", location="global")
        session = _Session()           # scripted with nothing; any call raises
        ex._session = session
        assert not ex.is_available()
        assert ex.explain_candidates({"positions": []}) == []
        assert ex.explain_all({"positions": [], "portfolio": {}}).portfolio.summary == ""
        assert session.requests == []

    def test_an_explicitly_empty_project_does_not_fall_back_to_the_configured_one(
        self, monkeypatch
    ):
        """
        The sentinel rule. With a falsiness test, a caller that deliberately
        passes no project silently gets the ambient configured one — which on
        a developer machine is a live, billed project the caller never named.

        The configured value is installed here rather than read from the
        environment. The first version of this test asserted that a project
        was already configured and told the reader to put one in
        backend/.env — which passed on the machine it was written on and
        failed in CI, where no .env exists. A test that needs a developer's
        local config to be meaningful is testing that machine, not this code.
        """
        from backend.config.settings import get_settings

        monkeypatch.setattr(get_settings(), "vertex_project", "configured-project")
        assert VertexExplainer(project="")._project == ""
        assert VertexExplainer()._project == "configured-project"

    def test_the_factory_returns_the_vertex_provider(self):
        ex = get_explainer("vertex")
        assert isinstance(ex, VertexExplainer)
        assert ex.provider_name == "vertex"

    def test_the_provider_name_reaches_the_response_as_vertex(self):
        """
        `llm_provider` is shown to the user. Inheriting "gemini" from the
        parent would tell them the wrong surface produced the narrative.
        """
        assert VertexExplainer(project=PROJECT).provider_name == "vertex"
        assert VertexExplainer(project=PROJECT).display_name == "Vertex AI"

    def test_a_combined_call_is_supported(self):
        assert VertexExplainer(project=PROJECT).supports_combined_call()
