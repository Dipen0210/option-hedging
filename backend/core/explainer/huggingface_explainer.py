"""
HuggingFace Inference API explainer.

Uses huggingface_hub.InferenceClient with the serverless Inference API.
Default model: meta-llama/Llama-3.3-70B-Instruct

Requires:
    settings.hf_api_token  — HuggingFace API token (hf_...)
    settings.hf_model      — model ID (default: meta-llama/Llama-3.3-70B-Instruct)

No local GPU, no Ollama. All inference runs on HuggingFace servers.
huggingface_hub ships with the transformers package already in requirements.txt.
"""
import json
import logging
from typing import Any, Dict, List

from backend.core.explainer.base_explainer import (
    BaseLLMExplainer,
    CandidateExplanation,
    CombinedExplanation,
    LLMProviderRejectedError,
    LLMRateLimitError,
    PortfolioExplanation,
    raise_if_provider_refused,
)
from backend.core.explainer.prompt_builder import CANDIDATES_SYSTEM, COMBINED_SYSTEM, PORTFOLIO_SYSTEM
from backend.core.explainer.response_parsers import (
    parse_candidates,
    parse_combined,
    parse_portfolio,
)

logger = logging.getLogger(__name__)

CANDIDATE_MAX_TOKENS = 1500
PORTFOLIO_MAX_TOKENS = 400


class HFRateLimitError(LLMRateLimitError):
    """
    The HuggingFace rate limit was hit (HTTP 429).

    Kept as a provider-specific subclass so existing callers that catch it
    still work, while the engine can catch the provider-agnostic
    `LLMRateLimitError` instead.
    """


class HuggingFaceExplainer(BaseLLMExplainer):

    def __init__(self, model: str = None, api_token: str = None):
        from backend.config.settings import get_settings
        settings = get_settings()
        self._model     = model     or settings.hf_model     or "meta-llama/Llama-3.3-70B-Instruct"
        self._api_token = api_token or settings.hf_api_token
        self._client    = None  # lazy init

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                model=self._model,
                token=self._api_token,
            )
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_token)

    def supports_combined_call(self) -> bool:
        return True

    # ── Public interface ──────────────────────────────────────────────────────

    def explain_candidates(
        self,
        context: Dict[str, Any],
    ) -> List[CandidateExplanation]:
        if not self.is_available():
            logger.warning("HuggingFace API token not configured — skipping explanation")
            return []

        try:
            response = self._get_client().chat_completion(
                messages=[
                    {"role": "system", "content": CANDIDATES_SYSTEM},
                    {"role": "user",   "content": json.dumps(context, indent=2)},
                ],
                max_tokens=CANDIDATE_MAX_TOKENS,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_candidates(raw, context)

        except (LLMProviderRejectedError, LLMRateLimitError):
            raise
        except Exception as e:
            # A refusal the caller has to report differently is re-raised; an
            # ordinary failure degrades to an empty result as before.
            raise_if_provider_refused(e, self.provider_name)
            logger.error(f"HuggingFace explain_candidates failed: {e}")
            return []

    def explain_portfolio(
        self,
        context: Dict[str, Any],
    ) -> PortfolioExplanation:
        if not self.is_available():
            return PortfolioExplanation()

        try:
            response = self._get_client().chat_completion(
                messages=[
                    {"role": "system", "content": PORTFOLIO_SYSTEM},
                    {"role": "user",   "content": json.dumps(context, indent=2)},
                ],
                max_tokens=PORTFOLIO_MAX_TOKENS,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_portfolio(raw)

        except (LLMProviderRejectedError, LLMRateLimitError):
            raise
        except Exception as e:
            # A refusal the caller has to report differently is re-raised; an
            # ordinary failure degrades to an empty result as before.
            raise_if_provider_refused(e, self.provider_name)
            logger.error(f"HuggingFace explain_portfolio failed: {e}")
            return PortfolioExplanation()

    def explain_all(self, combined_context: Dict[str, Any]) -> CombinedExplanation:
        """Single API call covering all positions + portfolio narrative."""
        if not self.is_available():
            return CombinedExplanation()

        try:
            response = self._get_client().chat_completion(
                messages=[
                    {"role": "system", "content": COMBINED_SYSTEM},
                    {"role": "user",   "content": json.dumps(combined_context, indent=2)},
                ],
                max_tokens=3500,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_combined(raw, combined_context)

        except (LLMProviderRejectedError, LLMRateLimitError):
            raise
        except Exception as e:
            # A refusal the caller has to report differently is re-raised; an
            # ordinary failure degrades to an empty result as before.
            raise_if_provider_refused(e, self.provider_name)
            logger.error(f"HuggingFace explain_all failed: {e}")
            return CombinedExplanation()

    # ── Parsers ───────────────────────────────────────────────────────────────
    #
    # Delegated. Every provider is asked for the same JSON shape by the same
    # system prompts, so the parsing belongs to the prompt contract rather than
    # to a provider; keeping a copy per provider is how two of them end up
    # disagreeing about the same response. These remain methods because the
    # class's own call sites and tests use them.

    def _parse_candidates(self, raw, context):
        return parse_candidates(raw, context, "HuggingFace")

    def _parse_portfolio(self, raw):
        return parse_portfolio(raw, "HuggingFace")

    def _parse_combined(self, raw, context):
        return parse_combined(raw, context, "HuggingFace")


