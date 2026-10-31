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

    #: What to call this instrument class in a user-facing disclosure.
    label: str = "instrument"

    #: False for a declared stub whose `find_candidates` returns [] without
    #: evaluating anything.
    #:
    #: This is not documentation. An empty candidate list from a stub and an
    #: empty one from a selector that priced every listed contract and
    #: rejected them are the same value, and the pipeline used to report both
    #: as "no hedge fits the cost budget and liquidity constraints" — naming a
    #: financial cause for a search that never ran. A caller of
    #: POST /instruments/futures was told futures hedging was too expensive
    #: for their portfolio when no future had been looked at. L4 reads this
    #: flag so it can say which of the two actually happened.
    implemented: bool = True

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
