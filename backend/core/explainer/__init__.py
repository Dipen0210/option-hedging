"""
LLM Explainer subsystem.

Usage:
    from backend.core.explainer import get_explainer

    explainer = get_explainer()   # reads settings.llm_provider
    explanations = explainer.explain_candidates(context)
    portfolio_exp = explainer.explain_portfolio(context)

Available providers:
    "claude"        — Anthropic claude-sonnet-4-6 (settings.anthropic_api_key)
    "vertex"        — Google Gemini over Vertex AI, authenticated with IAM
                      (settings.vertex_project, _location, _model)
    "gemini"        — the same models over the Generative Language REST API,
                      authenticated with an API key
                      (settings.gemini_api_key, settings.gemini_model)
    "huggingface"   — meta-llama/Llama-3.3-70B-Instruct via HF Inference API
                      (settings.hf_api_token, settings.hf_model)
    "ollama"        — a local Ollama server
                      (settings.ollama_base_url, settings.ollama_model)
"""
import logging
from backend.core.explainer.base_explainer import BaseLLMExplainer, CandidateExplanation, PortfolioExplanation
from backend.core.explainer.claude_explainer import ClaudeExplainer
from backend.core.explainer.gemini_explainer import GeminiExplainer
from backend.core.explainer.huggingface_explainer import HuggingFaceExplainer
from backend.core.explainer.ollama_explainer import OllamaExplainer
from backend.core.explainer.prompt_builder import PromptBuilder
from backend.core.explainer.vertex_explainer import VertexExplainer


logger = logging.getLogger(__name__)


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

    if p == "vertex":
        return VertexExplainer()
    if p == "gemini":
        return GeminiExplainer()
    if p == "huggingface":
        return HuggingFaceExplainer()
    if p == "ollama":
        return OllamaExplainer()
    if p != "claude":
        # Falling back silently is how `llm_provider=ollama` ran against
        # Anthropic for as long as that value has been accepted: the name was
        # unrecognised, this function returned Claude, and nothing said so.
        # Still a fallback rather than an error — an unreadable provider name
        # must not take the whole pipeline down — but a loud one.
        logger.error(
            "Unknown llm_provider %r — falling back to the Claude explainer. "
            "Explanations will come from Anthropic, not from %r.", p, p,
        )
    return ClaudeExplainer()


__all__ = [
    "BaseLLMExplainer",
    "CandidateExplanation",
    "PortfolioExplanation",
    "ClaudeExplainer",
    "GeminiExplainer",
    "VertexExplainer",
    "HuggingFaceExplainer",
    "OllamaExplainer",
    "PromptBuilder",
    "get_explainer",
]
