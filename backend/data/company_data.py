"""
Loader and lookup utilities for cleaned_companies.csv.
Provides company metadata (name, sector, market cap) for any ticker.
Used by risk analysis and instrument selector to apply sector-specific hedge logic.
"""
import csv
import os
from typing import Optional, Dict
from functools import lru_cache
from dataclasses import dataclass

CSV_PATH = os.path.join(os.path.dirname(__file__), "cleaned_companies.csv")


@dataclass
class CompanyInfo:
    ticker: str
    name: str
    market_cap: float
    sector: str

    @property
    def market_cap_category(self) -> str:
        """Classify by market cap (standard US definitions)."""
        if self.market_cap >= 200_000_000_000:
            return "mega_cap"
        elif self.market_cap >= 10_000_000_000:
            return "large_cap"
        elif self.market_cap >= 2_000_000_000:
            return "mid_cap"
        elif self.market_cap >= 300_000_000:
            return "small_cap"
        else:
            return "micro_cap"

    @property
    def is_commodity_exposed(self) -> bool:
        return self.sector in ("Energy", "Basic Materials")

    @property
    def is_rate_sensitive(self) -> bool:
        return self.sector in ("Real Estate", "Utilities", "Financial Services")

    @property
    def is_defensive(self) -> bool:
        return self.sector in ("Healthcare", "Consumer Defensive", "Utilities")


@lru_cache(maxsize=1)
def _load_all() -> Dict[str, CompanyInfo]:
    companies: Dict[str, CompanyInfo] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"].strip().upper()
            try:
                mc = float(row["market cap"])
            except (ValueError, KeyError):
                mc = 0.0
            companies[ticker] = CompanyInfo(
                ticker=ticker,
                name=row["company name"].strip(),
                market_cap=mc,
                sector=row["sector"].strip(),
            )
    return companies


def get_company(ticker: str) -> Optional[CompanyInfo]:
    """Returns CompanyInfo for a ticker, or None if not found."""
    return _load_all().get(ticker.upper().strip())


def get_sector(ticker: str) -> str:
    """Returns sector string, or 'Unknown' if not in database."""
    info = get_company(ticker)
    return info.sector if info else "Unknown"


def ticker_exists(ticker: str) -> bool:
    return ticker.upper().strip() in _load_all()
