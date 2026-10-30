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
from backend.core.explainer.claude_explainer import _extract_json

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

    def _parse_candidates(
        self,
        raw: str,
        context: Dict[str, Any],
    ) -> List[CandidateExplanation]:
        try:
            data = json.loads(_extract_json(raw))
            if not isinstance(data, list):
                data = [data]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"HuggingFace candidates JSON parse failed: {e}\nRaw: {raw[:300]}")
            return []

        asset_ticker = context.get("holding", {}).get("ticker", "")
        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            results.append(CandidateExplanation(
                ticker=item.get("ticker", ""),
                strategy=item.get("strategy", ""),
                asset_ticker=asset_ticker,
                when_works_best=item.get("when_works_best", ""),
                when_fails=item.get("when_fails", ""),
                rationale=item.get("rationale", ""),
                pros=item.get("pros", []),
                cons=item.get("cons", []),
            ))
        return results

    def _parse_portfolio(self, raw: str) -> PortfolioExplanation:  # noqa: E303
        try:
            data = json.loads(_extract_json(raw))
            if not isinstance(data, dict):
                return PortfolioExplanation()
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"HuggingFace portfolio JSON parse failed: {e}")
            return PortfolioExplanation()

        return PortfolioExplanation(
            summary=data.get("summary", ""),
            key_risks=data.get("key_risks", []),
            regime_commentary=data.get("regime_commentary", ""),
            top_recommendation=data.get("top_recommendation", ""),
        )


    def _parse_combined(self, raw: str, context: Dict[str, Any]) -> CombinedExplanation:
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"HuggingFace combined JSON parse failed: {e}\nRaw: {raw[:300]}")
            return CombinedExplanation()

        result = CombinedExplanation()

        # ── Candidate explanations keyed by asset_ticker → strategy ──────────
        positions_data = data.get("positions", {})
        if not isinstance(positions_data, dict):
            logger.warning("HuggingFace combined: 'positions' is not a dict")
            return result

        for asset_ticker, strategies in positions_data.items():
            if not isinstance(strategies, dict):
                continue
            result.candidates[asset_ticker] = {}
            for strategy, fields in strategies.items():
                if not isinstance(fields, dict):
                    continue
                result.candidates[asset_ticker][strategy] = CandidateExplanation(
                    ticker="",          # filled by caller from candidate data
                    strategy=strategy,
                    asset_ticker=asset_ticker,
                    when_works_best=fields.get("when_works_best", ""),
                    when_fails=fields.get("when_fails", ""),
                    rationale=fields.get("rationale", ""),
                    pros=fields.get("pros", []),
                    cons=fields.get("cons", []),
                )

        # ── Portfolio narrative ───────────────────────────────────────────────
        port = data.get("portfolio", {})
        if isinstance(port, dict):
            result.portfolio = PortfolioExplanation(
                summary=port.get("summary", ""),
                key_risks=port.get("key_risks", []),
                regime_commentary=port.get("regime_commentary", ""),
                top_recommendation=port.get("top_recommendation", ""),
            )

        return result


def _is_rate_limit(exc: Exception) -> bool:
    """
    Deprecated — kept only because it was public.

    It matched the bare substrings "429" and "quota" anywhere in an error
    message, which classified HuggingFace's HTTP 402 "You have no remaining
    credits" as a transient rate limit whenever those digits turned up in the
    appended request ID. Use `raise_if_provider_refused`, which reads the HTTP
    status and only falls back to wording when there is none.
    """
    try:
        raise_if_provider_refused(exc, "huggingface")
    except LLMRateLimitError:
        return True
    except LLMProviderRejectedError:
        return False
    return False
