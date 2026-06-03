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
from typing import List, Dict, Any


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
      - Never raise exceptions to the caller — return empty/fallback objects on error
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
