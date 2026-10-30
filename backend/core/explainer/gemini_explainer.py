"""
Google Gemini explainer, over the Generative Language REST API.

Deliberately no SDK. `google-genai` pulls google-auth, protobuf and grpcio into
the runtime closure; `requests==2.33.1` is already pinned there and already
used by `backend/data/market_data.py`, so this provider adds **no new
dependency**. That matters here beyond tidiness: the deployment contract
verifies that every distribution in the image resolves to a `cp313` linux
x86-64 wheel with zero source builds, and grpcio is a frequent exception to
that.

Two things this provider does that the others cannot:

  * `responseMimeType: application/json` makes the API itself reject non-JSON
    output, so a malformed body is a provider-side failure rather than
    something `extract_json` has to rescue. The shared parser still runs, as
    belt and braces.
  * `thinkingBudget: 0`. Gemini 2.5+ models reason before answering and that
    reasoning is charged against `maxOutputTokens`. Left on, a request with a
    small budget can spend the entire allowance thinking and return
    `finishReason: MAX_TOKENS` with **no text at all** — which would reach the
    user as "the model returned no analysis", the exact misreport defects 38
    and 39 were about. These prompts ask for structured restatement of figures
    that are already computed, not for reasoning, so the budget is zero and
    the failure mode is designed out rather than handled.

The API key travels in the `x-goog-api-key` header, never as a `?key=` query
parameter: `requests` puts the full URL into the text of every `HTTPError` it
raises, so a key in the query string would be copied into exception messages,
logs and anything that serialises them.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from backend.core.explainer.base_explainer import (
    BaseLLMExplainer,
    CandidateExplanation,
    CombinedExplanation,
    LLMProviderRejectedError,
    LLMRateLimitError,
    PortfolioExplanation,
    raise_if_provider_refused,
)
from backend.core.explainer.prompt_builder import (
    CANDIDATES_SYSTEM,
    COMBINED_SYSTEM,
    PORTFOLIO_SYSTEM,
)
from backend.core.explainer.response_parsers import (
    parse_candidates,
    parse_combined,
    parse_portfolio,
)

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Only reached if GEMINI_MODEL is set but empty; the real default, and the
# reasoning behind it, live in backend/config/settings.py.
DEFAULT_MODEL = "gemini-3.5-flash"

CANDIDATE_MAX_TOKENS = 1500
PORTFOLIO_MAX_TOKENS = 400
COMBINED_MAX_TOKENS = 3500

# A hedge run has its own server-side deadline. An LLM narrative is the one
# part of the pipeline the product can do without, so it must not be what
# holds the request open.
REQUEST_TIMEOUT_SECONDS = 60

# Capacity failures, retried. Measured, not assumed: across four models and
# eight calls of the real combined prompt, five came back
# `503 - The model is currently overloaded`, which is the free tier shedding
# load. Without a retry, roughly half of all hedge runs would silently lose
# their narrative and show a "could not be reached" notice for a condition
# that clears in a second.
#
# Bounded on purpose. Three attempts with 1s then 3s of backoff adds at most
# ~4s of waiting to a request that already takes ~8s for the model itself, and
# a run that cannot get through in three tries degrades to the deterministic
# narrative, which is a real result and is disclosed as one. Retrying only
# 503/504 matters: a 429 is a rate limit, where retrying makes the problem
# worse, and the refusals (400/401/402/403/404) will give the same answer
# every time.
_RETRY_STATUSES = frozenset({503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 3.0)


class GeminiHTTPError(Exception):
    """
    A non-2xx from the Generative Language API, carrying what the classifier needs.

    `requests.HTTPError` stringifies to "400 Client Error: Bad Request for url
    ..." and drops the response body — but the body is where Google puts
    `API_KEY_INVALID`, and Google answers an invalid key with **400**, not 401.
    Without the body text, `raise_if_provider_refused` could not tell a refused
    key from a malformed request of our own. This carries both.
    """

    def __init__(self, status_code: int, api_message: str):
        super().__init__(f"Gemini API returned {status_code}: {api_message}")
        self.status_code = status_code
        self.api_message = api_message


class GeminiExplainer(BaseLLMExplainer):

    @property
    def provider_name(self) -> str:
        return "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        `None` means "not specified, read the configuration"; an empty string
        means "explicitly none". The distinction is not pedantry — with a
        falsiness test, a caller that deliberately passes no key gets the
        ambient configured one instead, which is a silent substitution of a
        live credential for an absent one.
        """
        from backend.config.settings import get_settings
        settings = get_settings()
        self._model = model if model is not None else (
            settings.gemini_model or DEFAULT_MODEL)
        self._api_key = api_key if api_key is not None else settings.gemini_api_key
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def is_available(self) -> bool:
        return bool(self._api_key)

    def supports_combined_call(self) -> bool:
        return True

    # ── The one network call ──────────────────────────────────────────────────

    def _generate(self, system: str, user_payload: Dict[str, Any], max_tokens: int) -> str:
        """
        One `generateContent` call. Returns the model's text, possibly empty.

        Retries a capacity failure up to `_MAX_ATTEMPTS` times, then raises
        `GeminiHTTPError` on a non-2xx so the caller can classify it.
        Never raises for an empty-but-successful response: that is a blank
        result, which the engine's backstop already handles and discloses.
        """
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{
                "role": "user",
                "parts": [{"text": json.dumps(user_payload, indent=2)}],
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                # See the module docstring: reasoning tokens are charged against
                # maxOutputTokens and can consume the whole budget.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            resp = self._get_session().post(
                f"{API_ROOT}/{self._model}:generateContent",
                headers={
                    "x-goog-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if resp.status_code < 400:
                return _first_text(resp.json(), self._model)

            err = GeminiHTTPError(resp.status_code, _api_error_message(resp))
            if resp.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
                raise err

            delay = _BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "Gemini (%s) returned %d on attempt %d/%d — retrying in %.0fs",
                self._model, resp.status_code, attempt, _MAX_ATTEMPTS, delay,
            )
            time.sleep(delay)

        raise AssertionError("unreachable: the loop either returns or raises")

    # ── Public interface ──────────────────────────────────────────────────────

    def explain_candidates(self, context: Dict[str, Any]) -> List[CandidateExplanation]:
        if not self.is_available():
            logger.warning("Gemini API key not configured — skipping explanation")
            return []
        try:
            raw = self._generate(CANDIDATES_SYSTEM, context, CANDIDATE_MAX_TOKENS)
            return parse_candidates(raw, context, "Gemini")
        except (LLMProviderRejectedError, LLMRateLimitError):
            raise
        except Exception as e:
            raise_if_provider_refused(e, self.provider_name)
            logger.error("Gemini explain_candidates failed: %s", e)
            return []

    def explain_portfolio(self, context: Dict[str, Any]) -> PortfolioExplanation:
        if not self.is_available():
            return PortfolioExplanation()
        try:
            raw = self._generate(PORTFOLIO_SYSTEM, context, PORTFOLIO_MAX_TOKENS)
            return parse_portfolio(raw, "Gemini")
        except (LLMProviderRejectedError, LLMRateLimitError):
            raise
        except Exception as e:
            raise_if_provider_refused(e, self.provider_name)
            logger.error("Gemini explain_portfolio failed: %s", e)
            return PortfolioExplanation()

    def explain_all(self, combined_context: Dict[str, Any]) -> CombinedExplanation:
        """Single call covering every position plus the portfolio narrative."""
        if not self.is_available():
            return CombinedExplanation()
        try:
            raw = self._generate(COMBINED_SYSTEM, combined_context, COMBINED_MAX_TOKENS)
            return parse_combined(raw, combined_context, "Gemini")
        except (LLMProviderRejectedError, LLMRateLimitError):
            raise
        except Exception as e:
            raise_if_provider_refused(e, self.provider_name)
            logger.error("Gemini explain_all failed: %s", e)
            return CombinedExplanation()


# ── Response helpers ──────────────────────────────────────────────────────────

def _api_error_message(resp: "requests.Response") -> str:
    """
    Google's own description of the failure, for classification only.

    Never shown to a user: `LLMProviderRejectedError.reason` is built from the
    status instead, precisely so provider text cannot carry request IDs or
    internal URLs into a response body.
    """
    try:
        return str(((resp.json() or {}).get("error") or {}).get("message") or "")[:300]
    except ValueError:
        return (resp.text or "")[:300]


def _first_text(payload: Dict[str, Any], model: str) -> str:
    """
    The text of the first candidate, or "" with a log line saying why not.

    An empty return is a blank result, and the engine discloses that it filled
    in. The log line matters because the causes are different and all of them
    are silent otherwise: a prompt refused by a safety filter, a response
    truncated at the token budget, or a candidate with no text part at all.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        reason = ((payload.get("promptFeedback") or {}).get("blockReason")) or "no candidates returned"
        logger.warning("Gemini (%s) returned nothing: %s", model, reason)
        return ""

    first = candidates[0] or {}
    finish = first.get("finishReason") or ""
    parts = ((first.get("content") or {}).get("parts")) or []
    text = "".join(p.get("text") or "" for p in parts if isinstance(p, dict)).strip()

    if not text:
        logger.warning(
            "Gemini (%s) returned an empty response (finishReason=%s). "
            "MAX_TOKENS here means the output budget was exhausted; SAFETY "
            "means the prompt was refused.", model, finish or "unspecified",
        )
    elif finish not in ("STOP", ""):
        logger.warning(
            "Gemini (%s) stopped early (finishReason=%s) — the JSON may be "
            "truncated and parse to a partial result.", model, finish,
        )
    return text
