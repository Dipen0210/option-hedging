"""
Shared parsers for LLM explanation responses.

Every provider is asked for the same JSON shape by the same system prompts in
`prompt_builder`, so the code that turns that JSON into
`CandidateExplanation` / `PortfolioExplanation` / `CombinedExplanation` is a
property of the prompt contract, not of any one provider. It lives here so a
second provider does not arrive with a second copy that drifts from the first —
`test_llm_fallback.py` already says of its own fixtures that "duplicating them
here is how they drift", and the same applies with more force to parsing.

Every function is total: malformed or partially-shaped JSON yields an empty or
partially-filled result and a log line, never an exception. A provider refusal
is a different matter and is raised — see `raise_if_provider_refused`.
"""
import json
import logging
from typing import Any, Dict, List

from backend.core.explainer.base_explainer import (
    CandidateExplanation,
    CombinedExplanation,
    PortfolioExplanation,
)

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
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


def parse_candidates(
    raw: str,
    context: Dict[str, Any],
    provider: str,
) -> List[CandidateExplanation]:
    """Per-holding candidate explanations, from a JSON list (or single object)."""
    try:
        data = json.loads(extract_json(raw))
        if not isinstance(data, list):
            data = [data]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("%s candidates JSON parse failed: %s\nRaw: %s", provider, e, raw[:300])
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


def parse_portfolio(raw: str, provider: str) -> PortfolioExplanation:
    """The portfolio-level narrative, from a single JSON object."""
    try:
        data = json.loads(extract_json(raw))
        if not isinstance(data, dict):
            return PortfolioExplanation()
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("%s portfolio JSON parse failed: %s", provider, e)
        return PortfolioExplanation()

    return PortfolioExplanation(
        summary=data.get("summary", ""),
        key_risks=data.get("key_risks", []),
        regime_commentary=data.get("regime_commentary", ""),
        top_recommendation=data.get("top_recommendation", ""),
    )


def parse_combined(raw: str, context: Dict[str, Any], provider: str) -> CombinedExplanation:
    """
    One call covering every position plus the portfolio narrative.

    `CandidateExplanation.ticker` is deliberately left empty: the model is
    keyed on asset_ticker → strategy, and the caller fills the contract ticker
    from the candidate it matched. Taking the model's word for the contract
    symbol would let a hallucinated ticker into a recommendation.
    """
    try:
        data = json.loads(extract_json(raw))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("%s combined JSON parse failed: %s\nRaw: %s", provider, e, raw[:300])
        return CombinedExplanation()

    result = CombinedExplanation()
    if not isinstance(data, dict):
        logger.warning("%s combined: response is not a JSON object", provider)
        return result

    positions_data = data.get("positions", {})
    if not isinstance(positions_data, dict):
        logger.warning("%s combined: 'positions' is not a dict", provider)
        return result

    for asset_ticker, strategies in positions_data.items():
        if not isinstance(strategies, dict):
            continue
        result.candidates[asset_ticker] = {}
        for strategy, fields in strategies.items():
            if not isinstance(fields, dict):
                continue
            result.candidates[asset_ticker][strategy] = CandidateExplanation(
                ticker="",
                strategy=strategy,
                asset_ticker=asset_ticker,
                when_works_best=fields.get("when_works_best", ""),
                when_fails=fields.get("when_fails", ""),
                rationale=fields.get("rationale", ""),
                pros=fields.get("pros", []),
                cons=fields.get("cons", []),
            )

    port = data.get("portfolio", {})
    if isinstance(port, dict):
        result.portfolio = PortfolioExplanation(
            summary=port.get("summary", ""),
            key_risks=port.get("key_risks", []),
            regime_commentary=port.get("regime_commentary", ""),
            top_recommendation=port.get("top_recommendation", ""),
        )

    return result
