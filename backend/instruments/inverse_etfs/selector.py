"""
Inverse ETF instrument selector — stub for Phase 2.

Will implement:
  - Single-inverse ETFs: SH (S&P), PSQ (Nasdaq), RWM (Russell 2000)
  - Leveraged-inverse: SDS (-2× S&P), SQQQ (-3× Nasdaq)
  - Sector inverse: SEF (Financials), ERY (Energy)
  - Sizing: notional match adjusted for leverage multiplier
  - Caveats: daily rebalancing drag, path-dependency warning
"""
import logging
from typing import List

from backend.instruments.base import InstrumentSelector
from backend.models.risk_models import RiskProfile, RegimeState
from backend.models.portfolio_models import PortfolioInput
from backend.models.hedge_models import InstrumentCandidate

logger = logging.getLogger(__name__)


class InverseETFSelector(InstrumentSelector):

    def is_applicable(self, profile: RiskProfile, regime: RegimeState) -> bool:
        # Only apply for short-term hedges (≤ 30 days) to avoid rebalancing drag
        return profile.asset_class == "equity"

    def find_candidates(
        self,
        profile: RiskProfile,
        portfolio: PortfolioInput,
        regime: RegimeState,
    ) -> List[InstrumentCandidate]:
        logger.debug("InverseETFSelector: not yet implemented (Phase 2)")
        return []
