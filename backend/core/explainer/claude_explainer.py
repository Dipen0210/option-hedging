"""
Claude (Anthropic) LLM explainer.

Uses claude-sonnet-4-6 by default (configurable via settings.llm_model).
Sends one structured JSON prompt per holding for candidates, then one
portfolio-level prompt.

Structured output strategy:
  - System prompt demands valid JSON only
  - Response parsed with json.loads; falls back to rule-based text on failure
  - Uses Anthropic messages API (not streaming) for deterministic output
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

logger = logging.getLogger(__name__)

# Max tokens to request — candidates fit in ~800, portfolio fits in ~400
CANDIDATE_MAX_TOKENS = 1200
PORTFOLIO_MAX_TOKENS = 600


class ClaudeExplainer(BaseLLMExplainer):

    def __init__(self, model: str = None, api_key: str = None):
        from backend.config.settings import get_settings
        settings = get_settings()
        self._model    = model or settings.llm_model or "claude-sonnet-4-6"
        self._api_key  = api_key or settings.anthropic_api_key
        self._client   = None  # lazy init

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    # ── Public interface ──────────────────────────────────────────────────────

    def explain_candidates(
        self,
        context: Dict[str, Any],
    ) -> List[CandidateExplanation]:
        """
        Send candidates context to Claude, parse structured JSON response.
        Returns empty list on any error (engine handles fallback).
        """
        if not self.is_available():
            logger.warning("Claude API key not configured — skipping explanation")
            return []

        try:
            user_msg = json.dumps(context, indent=2)
            response = self._get_client().messages.create(
                model=self._model,
                max_tokens=CANDIDATE_MAX_TOKENS,
                system=CANDIDATES_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            return self._parse_candidates(raw, context)

        except Exception as e:
            logger.error(f"Claude explain_candidates failed: {e}")
            return []

    def explain_portfolio(
        self,
        context: Dict[str, Any],
    ) -> PortfolioExplanation:
        """
        Send portfolio context to Claude, parse structured JSON response.
        Returns empty PortfolioExplanation on any error.
        """
        if not self.is_available():
            return PortfolioExplanation()

        try:
            user_msg = json.dumps(context, indent=2)
            response = self._get_client().messages.create(
                model=self._model,
                max_tokens=PORTFOLIO_MAX_TOKENS,
                system=PORTFOLIO_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            return self._parse_portfolio(raw)

        except Exception as e:
            logger.error(f"Claude explain_portfolio failed: {e}")
            return PortfolioExplanation()

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_candidates(
        self,
        raw: str,
        context: Dict[str, Any],
    ) -> List[CandidateExplanation]:
        """Parse JSON array from Claude into CandidateExplanation objects."""
        try:
            data = json.loads(_extract_json(raw))
            if not isinstance(data, list):
                data = [data]   # model sometimes wraps in object
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Claude candidates JSON parse failed: {e}")
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
        """Parse JSON object from Claude into PortfolioExplanation."""
        try:
            data = json.loads(_extract_json(raw))
            if not isinstance(data, dict):
                return PortfolioExplanation()
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Claude portfolio JSON parse failed: {e}")
            return PortfolioExplanation()

        return PortfolioExplanation(
            summary=data.get("summary", ""),
            key_risks=data.get("key_risks", []),
            regime_commentary=data.get("regime_commentary", ""),
            top_recommendation=data.get("top_recommendation", ""),
        )


def _extract_json(text: str) -> str:
    """
    Extract the first complete JSON array or object from LLM output.
    Handles markdown code fences, trailing commas, and extra text that
    Llama/HF models append after the closing bracket.
    """
    import re
    text = text.strip()

    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()

    # Remove trailing commas before ] or } (common Llama quirk)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Extract the first complete balanced [ ... ] or { ... } block,
    # discarding anything Llama appended after the closing bracket.
    start = -1
    opener = closer = None
    for i, ch in enumerate(text):
        if ch in ("[", "{"):
            start = i
            opener = ch
            closer = "]" if ch == "[" else "}"
            break
    if start == -1:
        return text  # no JSON found — return as-is, let caller handle

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # Truncated — return what we have; trailing comma already cleaned above
    return text[start:]
