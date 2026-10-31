"""
Google Gemini explainer, over Vertex AI.

Vertex speaks the same `generateContent` protocol as the Generative Language
API, so this is `GeminiExplainer` with two things changed and nothing else:

  * **The endpoint.** The project and the region live in the URL path, and the
    host is regional. `global` is its own case, served by the unprefixed host.
  * **The credential.** Vertex has no API key. Every call carries an OAuth
    bearer token, which is why this is the one provider that needs a library:
    `google-auth` resolves Application Default Credentials and signs the
    token. See backend/requirements.txt for why the heavier SDKs were rejected.

Everything that was hard to get right — the retry policy, the `thinkingBudget`
defence against an empty 200, the empty-response reporting, the refusal
classification — is inherited, not copied, so a fix to any of it applies to
both Google surfaces at once.

**Why Vertex rather than the API key.** The same credential path works in both
places it has to: a developer's `gcloud auth application-default login`
locally, and the service account attached to the container in Cloud Run. There
is no long-lived secret to put in an environment variable, mount into an
image, or leak through a log — the two failure modes an API key has and this
does not. Access is revoked in IAM instead of by rotating a string.

**Measured availability, not assumed.** `gemini-3.5-flash` is served from
`global` only. The regional endpoints (`us-central1` among them) answer the
3.x models with `404 NOT_FOUND`, and `gemini-2.5-*` is retired for projects
created after its deprecation — on Vertex exactly as on the Generative
Language API. The default location is therefore `global`, which is also what
Google's own quickstarts now use for the Gemini publisher models.

**Quota is attributed to this project explicitly.** Application Default
Credentials carry a `quota_project_id` from whichever project the developer
last configured — on this machine, a different one. `AuthorizedSession` turns
that into an `x-goog-user-project` header, so without the override below the
calls would bill and count against a project that has nothing to do with
OptionQ. The quota project is set from `VERTEX_PROJECT` instead, which is the
same project named in the URL path.
"""
import logging
from typing import Dict, Optional

import google.auth
import requests
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError
from google.auth.transport.requests import AuthorizedSession

from backend.core.explainer.base_explainer import LLMProviderRejectedError
from backend.core.explainer.gemini_explainer import GeminiExplainer

logger = logging.getLogger(__name__)

# Only reached if the setting is present but empty; the real defaults, and the
# measurements behind them, live in backend/config/settings.py.
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_LOCATION = "global"

# Vertex accepts nothing narrower. The token it mints is used for exactly one
# call, `publishers/google/models/*:generateContent`, which IAM governs through
# the service account's roles rather than through the scope.
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class VertexExplainer(GeminiExplainer):

    display_name = "Vertex AI"

    @property
    def provider_name(self) -> str:
        return "vertex"

    def __init__(
        self,
        model: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        `None` means "not specified, read the configuration"; an empty string
        means "explicitly none". Same sentinel rule as `GeminiExplainer`, for
        the same reason: with a falsiness test, a caller that deliberately
        passes no project would silently get the ambient configured one and
        bill a live account it never named.
        """
        from backend.config.settings import get_settings
        settings = get_settings()

        self._model = model if model is not None else (
            settings.vertex_model or DEFAULT_MODEL)
        self._project = project if project is not None else settings.vertex_project
        self._location = location if location is not None else (
            settings.vertex_location or DEFAULT_LOCATION)
        self._session: Optional[requests.Session] = None

        # Inherited methods still read `_api_key` through `is_available` on the
        # parent; Vertex has none, and the empty string is the honest value.
        self._api_key = ""

    def is_available(self) -> bool:
        """
        Whether a call is worth attempting.

        Only the project is checked. Whether the credentials actually resolve
        is deliberately *not* probed here: `is_available()` is called on every
        hedge run, credential resolution touches the filesystem and the
        metadata server, and — more importantly — a missing credential is an
        operator-visible refusal, not a reason to quietly skip the narrative.
        It is raised as one from `_get_session`.
        """
        return bool(self._project)

    # ── The two things that differ from the Generative Language API ───────────

    def _endpoint(self) -> str:
        host = ("https://aiplatform.googleapis.com" if self._location == "global"
                else f"https://{self._location}-aiplatform.googleapis.com")
        return (f"{host}/v1/projects/{self._project}/locations/{self._location}"
                f"/publishers/google/models/{self._model}:generateContent")

    def _auth_headers(self) -> Dict[str, str]:
        """
        None. `AuthorizedSession` sets `Authorization` itself, on every request
        and after every refresh — which is the point of using it rather than
        minting a token here and holding it until it expires mid-run.
        """
        return {}

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = AuthorizedSession(self._credentials())
        return self._session

    def _credentials(self):
        """
        Application Default Credentials, with quota attributed to this project.

        A failure here is permanent in the sense `LLMProviderRejectedError`
        means: no credentials are configured, or the ones that are cannot be
        signed. Retrying will not fix it and an operator has to act, so it is
        reported as a refusal with a categorical reason rather than degrading
        to "the model returned no analysis" — the misreport that
        `LLMProviderRejectedError` exists to prevent.
        """
        try:
            credentials, _ = google.auth.default(scopes=list(_SCOPES))
        except DefaultCredentialsError as e:
            logger.error("Vertex AI credentials could not be resolved: %s", e)
            raise LLMProviderRejectedError(
                self.provider_name,
                "no Google Cloud credentials are configured for this environment",
            ) from e
        except GoogleAuthError as e:
            logger.error("Vertex AI credential error: %s", e)
            raise LLMProviderRejectedError(
                self.provider_name,
                "the configured Google Cloud credentials were refused",
            ) from e

        if hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(self._project)
        return credentials
