"""
Abstract base for all LLM explainers.

Subclasses implement `explain_candidates` and `explain_portfolio`.
The engine always calls through this interface — the LLM provider
(Claude, Ollama, OpenAI, etc.) is swapped by config without touching
any other code.

Contract:
    explain_candidates(context) -> List[CandidateExplanation]
    explain_portfolio(context)  -> PortfolioExplanation
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LLMRateLimitError(Exception):
    """
    The provider refused this call but would accept a later one.

    Transient by definition: a retry, or waiting for a window to reset, is the
    correct response. Raised rather than swallowed so the engine can tell the
    user the analysis is coming back, instead of reporting a blank result.
    """


class LLMProviderRejectedError(Exception):
    """
    The provider refused this call and will refuse the next one too.

    Bad credentials, an exhausted balance or a model the account may not use.
    Nothing inside OptionQ can clear it and no amount of retrying will: an
    operator has to change a key or top up an account. It is a distinct type
    from `LLMRateLimitError` because the two demand opposite responses, and
    reporting one as the other is what this class exists to prevent — a
    permanent billing rejection used to surface as "the language model
    returned no analysis for this run", which reads as a one-off glitch.

    `reason` is a short, categorical phrase safe to show a user. It is derived
    from the HTTP status, never copied from the provider's response body: those
    bodies carry request IDs, internal URLs and account identifiers.
    """

    def __init__(self, provider: str, reason: str):
        super().__init__(f"{provider} rejected the request: {reason}")
        self.provider = provider
        self.reason = reason


# Statuses that mean "do not come back without changing something", mapped to
# the phrase shown to the user.
_REJECTION_REASONS = {
    401: "the configured API credentials were refused",
    402: "the provider account has no remaining credits",
    403: "the configured credentials are not permitted to use this model",
    404: "the configured model is not available to this account",
}


# Statuses that settle the question by themselves. A provider does not answer
# 401/402/403 to a request we malformed, so no wording check is needed or
# wanted: the real 402 body contains the digits "429".
_UNCONDITIONAL_STATUSES = frozenset({401, 402, 403})

# Statuses that do not settle it, because each is equally the shape of a bad
# request of our own. Decided by wording below. 404 must NOT be unconditional:
# "that model is retired for your account" and "we called the wrong URL" share
# a status, and reporting our own bug as a credential problem is its own lie.
_AMBIGUOUS_STATUSES = frozenset({400, 404})

_CREDENTIAL_PHRASES = (
    "api key not valid", "api_key_invalid", "invalid api key",
    "api key expired", "invalid authentication", "unauthorized",
)

# A model the account cannot reach. Found live rather than reasoned about:
# `gemini-2.5-flash` answers a new project with "404 — This model
# models/gemini-2.5-flash is no longer available to new users", and a retired
# model will keep being retired on every retry.
# "publisher model" is Vertex AI's wording and is the discriminator on purpose,
# not a convenience. Vertex answers an unavailable model with
#   404: Publisher model `projects/<p>/locations/<l>/publishers/google/
#        models/<m>` was not found or your project does not have access to it.
# while a 404 caused by *our own* malformed URL is Google's generic frontend
# page — "The requested URL ... was not found on this server" — which contains
# none of these phrases and so stays an ordinary failure. Matching on the
# shared words ("was not found", "does not have access") would have collapsed
# that distinction and reported a bug of ours as a model the account may not
# use.
_MODEL_PHRASES = (
    "no longer available", "is not found", "not found for api version",
    "model not found", "does not exist", "publisher model",
)

_PERMISSION_PHRASES = (
    "permission_denied", "permission denied",
    "has not been used in project", "is disabled",
)


def _http_status(exc: BaseException) -> Optional[int]:
    """
    The HTTP status behind an SDK exception, or None if it carries none.

    huggingface_hub hangs a requests/httpx response off `.response`; the
    anthropic SDK puts the code on the exception itself. Both are checked
    because neither is guaranteed and a missing one is not an error.
    """
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def raise_if_provider_refused(exc: BaseException, provider: str) -> None:
    """
    Translate a provider SDK exception into a typed refusal, or return.

    Returning means "this is an ordinary failure" — the caller logs it and
    degrades. Raising means the caller must not pretend the run merely came
    back empty.

    The status code is authoritative and the message is consulted **only** when
    there is no status. That ordering is the point of this function, not an
    implementation detail: a real HuggingFace 402 body reads "You have no
    remaining credits" and arrives alongside a request ID such as
    `Root=1-6ae50969-...`, so substring-matching an error string for "429" or
    "quota" can classify a permanent billing failure as a transient rate limit.
    Once a status is in hand, the text cannot overrule it.
    """
    status = _http_status(exc)

    if status is not None:
        if status in _UNCONDITIONAL_STATUSES:
            raise LLMProviderRejectedError(provider, _REJECTION_REASONS[status]) from exc
        if status == 429:
            raise LLMRateLimitError(f"{provider} rate limit reached") from exc
        if status in _AMBIGUOUS_STATUSES:
            # 400 and 404 are the two statuses that do not settle the question,
            # because each is equally the shape of a bad request of our own —
            # and a bug of ours is not a refusal and must not be reported as
            # one. So here, and only here, the wording decides.
            #
            # Both must be examined rather than ignored. Google's Generative
            # Language API answers an invalid key with `400 INVALID_ARGUMENT /
            # API_KEY_INVALID`, not 401, and a retired model with a 404 — so
            # letting either fall through as an ordinary failure would
            # reproduce the exact defect this function exists to prevent, on a
            # different provider.
            msg = str(exc).lower()
            if any(k in msg for k in _CREDENTIAL_PHRASES):
                raise LLMProviderRejectedError(provider, _REJECTION_REASONS[401]) from exc
            if any(k in msg for k in _PERMISSION_PHRASES):
                raise LLMProviderRejectedError(provider, _REJECTION_REASONS[403]) from exc
            if any(k in msg for k in _MODEL_PHRASES):
                raise LLMProviderRejectedError(provider, _REJECTION_REASONS[404]) from exc
        return

    # No status to go on — fall back to the wording. Deliberately no bare "402"
    # or "429" substring test: those digits appear in request IDs and in the
    # documentation URLs these SDKs append to their messages.
    msg = str(exc).lower()
    if any(k in msg for k in ("no remaining credits", "payment required", "insufficient credit")):
        raise LLMProviderRejectedError(provider, _REJECTION_REASONS[402]) from exc
    if any(k in msg for k in ("invalid api key", "authentication", "unauthorized")):
        raise LLMProviderRejectedError(provider, _REJECTION_REASONS[401]) from exc
    if any(k in msg for k in ("rate limit", "too many requests", "quota")):
        raise LLMRateLimitError(f"{provider} rate limit reached") from exc


@dataclass
class CandidateExplanation:
    """
    LLM-generated explanation for one InstrumentCandidate.
    Matched back to candidates by (ticker, strategy) key.
    """
    ticker: str
    strategy: str
    asset_ticker: str               # the holding being hedged
    when_works_best: str = ""
    when_fails: str = ""
    rationale: str = ""             # if LLM improves on the rule-based rationale
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


@dataclass
class PortfolioExplanation:
    """
    LLM-generated portfolio-level narrative.
    """
    summary: str = ""               # 2–4 sentence plain-English overview
    key_risks: List[str] = field(default_factory=list)   # top 3–5 risks
    regime_commentary: str = ""     # what the current regime means for this portfolio
    top_recommendation: str = ""    # one sentence on the #1 hedge pick


class BaseLLMExplainer(ABC):
    """
    Interface all LLM explainer implementations must satisfy.

    Implementors should:
      - Accept a structured context dict (built by PromptBuilder)
      - Return CandidateExplanation and PortfolioExplanation objects
      - Return empty/fallback objects when a call simply fails, so a provider
        hiccup cannot take down a hedge run
      - Raise `LLMProviderRejectedError` or `LLMRateLimitError` when the
        provider refuses for a reason the caller must report differently.
        This is the one exception to the rule above, and it is deliberate:
        "never raise" previously meant a permanently misconfigured provider
        was indistinguishable from a model that answered with nothing, and the
        user was told the model "returned no analysis for this run".
    """

    @property
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'claude', 'ollama')."""
        return self.__class__.__name__.replace("Explainer", "").lower()

    @abstractmethod
    def explain_candidates(
        self,
        context: Dict[str, Any],
    ) -> List[CandidateExplanation]:
        """
        Generate per-candidate explanations.

        Args:
            context: dict built by PromptBuilder.build_candidates_context()

        Returns:
            List of CandidateExplanation — one per (asset_ticker, strategy) pair.
            Order not guaranteed; engine matches by key.
        """
        ...

    @abstractmethod
    def explain_portfolio(
        self,
        context: Dict[str, Any],
    ) -> PortfolioExplanation:
        """
        Generate portfolio-level narrative.

        Args:
            context: dict built by PromptBuilder.build_portfolio_context()

        Returns:
            PortfolioExplanation with summary, key_risks, regime_commentary.
        """
        ...

    def is_available(self) -> bool:
        """
        Quick health-check — returns True if the provider is reachable.
        Default: True (subclasses should override if they make network calls).
        """
        return True

    def supports_combined_call(self) -> bool:
        """Return True if this provider implements explain_all() as a single API call."""
        return False

    def explain_all(
        self,
        combined_context: Dict[str, Any],
    ) -> "CombinedExplanation":
        """
        Single-call alternative: explain every candidate across every position
        AND the portfolio narrative in one prompt.

        Output is keyed by asset_ticker → strategy → CandidateExplanation,
        plus a PortfolioExplanation, so nothing can be misattributed.

        Only called when supports_combined_call() returns True.
        """
        raise NotImplementedError


@dataclass
class CombinedExplanation:
    """Result of a single explain_all() call covering the whole portfolio."""
    candidates: Dict[str, Dict[str, CandidateExplanation]] = field(default_factory=dict)
    portfolio: PortfolioExplanation = field(default_factory=PortfolioExplanation)
