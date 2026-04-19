"""
Swaps instrument selector — stub for Phase 2.

Will implement:
  - Interest rate swaps (pay-fixed / receive-float) for bond duration management
  - Total return swaps for synthetic short exposure
  - Credit default swaps (CDS) for credit risk hedging
  - Pricing: fixed-leg PV vs floating-leg PV
"""
import logging
from typing import List

from backend.instruments.base import InstrumentSelector
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import InstrumentCandidate

logger = logging.getLogger(__name__)


class SwapsSelector(InstrumentSelector):

    def is_applicable(self, profile: RiskProfile, regime: RegimeState) -> bool:
        return profile.asset_class in ("bond", "credit")

    def find_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        logger.debug("SwapsSelector: not yet implemented (Phase 2)")
        return []
