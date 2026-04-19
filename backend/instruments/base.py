"""
Abstract base for all hedge instrument selectors.

Each asset class (options, futures, forwards, swaps, inverse ETFs) subclasses
InstrumentSelector and implements `find_candidates`.
"""
from abc import ABC, abstractmethod
from typing import List
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import InstrumentCandidate


class InstrumentSelector(ABC):
    """
    Base class for all instrument selectors.

    Subclasses must implement `find_candidates` and optionally
    override `price_candidate` for custom pricing logic.
    """

    @abstractmethod
    def find_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        """
        Given a per-stock risk profile, portfolio constraints, and current
        regime, return a ranked list of InstrumentCandidates for this
        asset class.
        """
        ...

    def is_applicable(
        self,
        profile: RiskProfile,
        regime: RegimeState,
    ) -> bool:
        """
        Quick gate: return False to skip this asset class for a given
        holding/regime combination. Subclasses may override.
        Default: always applicable.
        """
        return True
