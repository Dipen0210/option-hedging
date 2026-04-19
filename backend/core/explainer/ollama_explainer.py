"""
Ollama LLM explainer — local model fallback (Llama 3, Mistral, etc).

Uses Ollama's REST API (POST /api/generate or /api/chat).
Works entirely offline — no API key required.

Default model: llama3 (override via settings.ollama_model or constructor).
Requires Ollama running at settings.ollama_base_url (default: http://localhost:11434).

Same JSON-in / JSON-out contract as ClaudeExplainer.
"""
import json
import logging
import requests
from typing import Any, Dict, List

from backend.core.explainer.base_explainer import (
    BaseLLMExplainer,
    CandidateExplanation,
    PortfolioExplanation,
)
from backend.core.explainer.prompt_builder import CANDIDATES_SYSTEM, PORTFOLIO_SYSTEM
from backend.core.explainer.claude_explainer import _extract_json   # reuse parser util

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 120    # local models can be slow; generous timeout
CANDIDATE_MAX_TOKENS = 1200
PORTFOLIO_MAX_TOKENS = 600


class OllamaExplainer(BaseLLMExplainer):

    def __init__(self, model: str = None, base_url: str = None):
        from backend.config.settings import get_settings
        settings = get_settings()
        self._model    = model    or settings.ollama_model    or "llama3"
        self._base_url = base_url or settings.ollama_base_url or "http://localhost:11434"

    def is_available(self) -> bool:
        """Ping Ollama health endpoint."""
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    # ── Public interface ──────────────────────────────────────────────────────

    def explain_candidates(
        self,
        context: Dict[str, Any],
    ) -> List[CandidateExplanation]:
        if not self.is_available():
            logger.warning("Ollama not reachable — skipping explanation")
            return []

        try:
            prompt = _build_chat_prompt(CANDIDATES_SYSTEM, json.dumps(context, indent=2))
            raw = self._generate(prompt, max_tokens=CANDIDATE_MAX_TOKENS)
            return self._parse_candidates(raw, context)
        except Exception as e:
            logger.error(f"Ollama explain_candidates failed: {e}")
            return []

    def explain_portfolio(
        self,
        context: Dict[str, Any],
    ) -> PortfolioExplanation:
        if not self.is_available():
            return PortfolioExplanation()

        try:
            prompt = _build_chat_prompt(PORTFOLIO_SYSTEM, json.dumps(context, indent=2))
            raw = self._generate(prompt, max_tokens=PORTFOLIO_MAX_TOKENS)
            return self._parse_portfolio(raw)
        except Exception as e:
            logger.error(f"Ollama explain_portfolio failed: {e}")
            return PortfolioExplanation()

    # ── Ollama HTTP call ──────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_tokens: int = 800) -> str:
        """
        POST to /api/generate (non-streaming).
        Ollama's /api/chat also works but /api/generate is simpler for
        system+user prompt injection.
        """
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,      # low temp for structured output
                "top_p": 0.9,
            },
        }
        response = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    # ── Parsers (identical logic to Claude) ──────────────────────────────────

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
            logger.warning(f"Ollama candidates JSON parse failed: {e}\nRaw: {raw[:300]}")
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
            logger.warning(f"Ollama portfolio JSON parse failed: {e}")
            return PortfolioExplanation()

        return PortfolioExplanation(
            summary=data.get("summary", ""),
            key_risks=data.get("key_risks", []),
            regime_commentary=data.get("regime_commentary", ""),
            top_recommendation=data.get("top_recommendation", ""),
        )


def _build_chat_prompt(system: str, user: str) -> str:
    """
    Combine system + user into a single prompt string.
    Ollama's /api/generate accepts a single `prompt` field, so we
    embed the system instruction in a [INST]...[/INST] template
    (works for Llama-based models; other models treat it as prefix).
    """
    return (
        f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"{user} [/INST]"
    )
