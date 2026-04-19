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
    PortfolioExplanation,
)
from backend.core.explainer.prompt_builder import CANDIDATES_SYSTEM, PORTFOLIO_SYSTEM
from backend.core.explainer.claude_explainer import _extract_json

logger = logging.getLogger(__name__)

CANDIDATE_MAX_TOKENS = 1200
PORTFOLIO_MAX_TOKENS = 600


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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"HuggingFace explain_portfolio failed: {e}")
            return PortfolioExplanation()

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

    def _parse_portfolio(self, raw: str) -> PortfolioExplanation:
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
