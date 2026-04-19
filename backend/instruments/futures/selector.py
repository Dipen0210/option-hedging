"""
Futures instrument selector — stub for Phase 2.

Will implement:
  - Index futures (ES, NQ, RTY) for equity portfolio hedging
  - Treasury futures (ZN, ZB) for bond duration hedging
  - Commodity futures (CL, GC, NG) for commodity exposure hedging
  - Pricing: cost-of-carry model (F = S × e^((r-q)T))
  - Sizing: min-variance futures (h* × V / (contract_size × F))
            beta-weighted (βp × V / contract_value)
            DV01 match for rate hedges
"""
import logging
from typing import List

from backend.instruments.base import InstrumentSelector
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import InstrumentCandidate

logger = logging.getLogger(__name__)


class FuturesSelector(InstrumentSelector):

    def is_applicable(self, profile: RiskProfile, regime: RegimeState) -> bool:
        # Futures are applicable to equity (index futures), bond, and commodity
        return profile.asset_class in ("equity", "bond", "commodity")

    def find_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        # Phase 2: implement index futures, treasury futures, commodity futures
        logger.debug("FuturesSelector: not yet implemented (Phase 2)")
        return []
