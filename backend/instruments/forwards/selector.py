"""
Forwards instrument selector — stub for Phase 2.

Will implement:
  - FX forwards for currency-exposed equity holdings
  - Commodity forwards (OTC) for energy/agriculture exposure
  - Pricing: forward price F = S × e^((r-q)T)
  - Sizing: notional match (forward notional = holding notional × fx_sensitivity)
"""
import logging
from typing import List

from backend.instruments.base import InstrumentSelector
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import InstrumentCandidate

logger = logging.getLogger(__name__)


class ForwardsSelector(InstrumentSelector):

    def is_applicable(self, profile: RiskProfile, regime: RegimeState) -> bool:
        return profile.asset_class in ("fx", "commodity")

    def find_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        logger.debug("ForwardsSelector: not yet implemented (Phase 2)")
        return []
