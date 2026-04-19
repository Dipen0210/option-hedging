"""
LLM Explainer subsystem.

Usage:
    from backend.core.explainer import get_explainer

    explainer = get_explainer()   # reads settings.llm_provider
    explanations = explainer.explain_candidates(context)
    portfolio_exp = explainer.explain_portfolio(context)

Available providers:
    "claude"        — Anthropic claude-sonnet-4-6 (settings.anthropic_api_key)
    "huggingface"   — meta-llama/Llama-3.3-70B-Instruct via HF Inference API
                      (settings.hf_api_token, settings.hf_model)
"""
from backend.core.explainer.base_explainer import BaseLLMExplainer, CandidateExplanation, PortfolioExplanation
from backend.core.explainer.claude_explainer import ClaudeExplainer
from backend.core.explainer.huggingface_explainer import HuggingFaceExplainer
from backend.core.explainer.prompt_builder import PromptBuilder


def get_explainer(provider: str = None) -> BaseLLMExplainer:
    """
    Factory — returns the right explainer for the configured provider.

    Priority:
      1. `provider` argument (for testing or explicit override)
      2. settings.llm_provider
      3. Falls back to ClaudeExplainer (no-ops if no API key)
    """
    from backend.config.settings import get_settings
    settings = get_settings()
    p = (provider or settings.llm_provider or "claude").lower()

    if p == "huggingface":
        return HuggingFaceExplainer()
    return ClaudeExplainer()


__all__ = [
    "BaseLLMExplainer",
    "CandidateExplanation",
    "PortfolioExplanation",
    "ClaudeExplainer",
    "HuggingFaceExplainer",
    "PromptBuilder",
    "get_explainer",
]
