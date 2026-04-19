"""
Supported ticker universe — equity options Phase 1.

Only tickers in SUPPORTED_TICKERS are accepted by the API.
These are chosen for:
  - Liquid options markets (tight spreads, high open interest)
  - Reliable yfinance data coverage
  - Active institutional hedging use

To extend: add tickers to the appropriate section below.
Phase 2 will expand to commodities, FX, and fixed income.
"""

# ── Broad market indices ───────────────────────────────────────────────────────
INDEX_ETFS = {
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "VTI",   # Total market
}

# ── Sector ETFs ───────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
}

# ── Volatility & hedging products ─────────────────────────────────────────────
VOLATILITY = {
    "GLD",   # Gold ETF
    "TLT",   # 20yr Treasury
    "HYG",   # High Yield Bond
    "LQD",   # Investment Grade Bond
    "UUP",   # USD index
}

# ── Mega-cap equities with deep options markets ────────────────────────────────
MEGA_CAP_EQUITIES = {
    # Technology
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "GOOGL", # Alphabet
    "META",  # Meta
    "AMZN",  # Amazon
    "AMD",   # AMD
    "TSLA",  # Tesla
    "NFLX",  # Netflix
    "CRM",   # Salesforce
    # Financials
    "JPM",   # JPMorgan
    "BAC",   # Bank of America
    "GS",    # Goldman Sachs
    "MS",    # Morgan Stanley
    "V",     # Visa
    "MA",    # Mastercard
    # Healthcare
    "JNJ",   # Johnson & Johnson
    "UNH",   # UnitedHealth
    "PFE",   # Pfizer
    "ABBV",  # AbbVie
    # Energy
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    # Other
    "BRKB",  # Berkshire Hathaway B
    "WMT",   # Walmart
    "KO",    # Coca-Cola
}

# ── Master set ─────────────────────────────────────────────────────────────────
SUPPORTED_TICKERS: set[str] = (
    INDEX_ETFS
    | SECTOR_ETFS
    | VOLATILITY
    | MEGA_CAP_EQUITIES
)


def is_supported(ticker: str) -> bool:
    return ticker.upper().strip() in SUPPORTED_TICKERS


def supported_tickers_list() -> list[str]:
    return sorted(SUPPORTED_TICKERS)
