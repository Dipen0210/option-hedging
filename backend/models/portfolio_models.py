from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated, ClassVar, List, Literal, Optional, Union
from datetime import date


# ── Shared ticker validator ────────────────────────────────────────────────────

def _validate_ticker(v: str) -> str:
    from backend.config.instrument_universe import is_supported
    ticker = v.upper().strip()
    if not is_supported(ticker):
        from backend.config.instrument_universe import supported_tickers_list
        raise ValueError(
            f"'{ticker}' is not in the supported ticker universe. "
            f"Supported tickers: {supported_tickers_list()}"
        )
    return ticker


# ── Mode A: Stock position ─────────────────────────────────────────────────────

class StockPosition(BaseModel):
    """
    A stock (equity) holding the user wants to hedge.

    Example: user owns 200 shares of AAPL at $185 and wants downside protection.
    """
    position_type: Literal["stock"] = "stock"

    ticker: str
    shares: float = Field(..., gt=0)
    purchase_price: float = Field(..., gt=0)
    purchase_date: date
    asset_class: Literal["equity", "commodity", "bond", "fx", "crypto"] = "equity"

    @field_validator("ticker")
    @classmethod
    def ticker_supported(cls, v: str) -> str:
        return _validate_ticker(v)

    @field_validator("shares", "purchase_price")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v


# ── Mode B: Options position ───────────────────────────────────────────────────

class OptionsPosition(BaseModel):
    """
    An existing options position the user wants to hedge.

    Example: user is long 10 AAPL $200 calls expiring 2025-06-20
    and wants to reduce delta/gamma/vega exposure using other options.

    Phase 2: pipeline routing for this mode is a stub — returns
    Greek-neutralisation recommendations once implemented.
    """
    position_type: Literal["option"] = "option"

    ticker: str
    option_type: Literal["call", "put"]
    strike: float = Field(..., gt=0)
    expiry: date
    contracts: int = Field(..., gt=0)           # number of contracts (1 contract = 100 shares)
    direction: Literal["long", "short"] = "long"
    premium_paid: Optional[float] = None        # per-share premium (optional, for P&L tracking)
    asset_class: Literal["equity"] = "equity"   # Phase 1: equity only

    @field_validator("ticker")
    @classmethod
    def ticker_supported(cls, v: str) -> str:
        return _validate_ticker(v)

    @field_validator("strike", "contracts")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @model_validator(mode="after")
    def expiry_must_be_future(self) -> "OptionsPosition":
        # Allow same-day (intraday positions); reject already-expired contracts
        if self.expiry < date.today():
            raise ValueError(
                f"expiry {self.expiry} is in the past (today is {date.today()})"
            )
        return self


# ── Union type ─────────────────────────────────────────────────────────────────

PositionInput = Annotated[
    Union[StockPosition, OptionsPosition],
    Field(discriminator="position_type"),
]


# ── Portfolio input ────────────────────────────────────────────────────────────

class PortfolioInput(BaseModel):
    """
    Top-level input to the hedge pipeline.

    Supports mixed positions: a user can have both stock holdings
    and existing options positions in the same portfolio.

    Phase 1: StockPosition → full 12-layer pipeline
    Phase 2: OptionsPosition → Greek-neutralisation pipeline (stub)
    """
    holdings: List[PositionInput]

    # Client-declared portfolio value. OPTIONAL and NOT authoritative: the
    # server recomputes market value from live prices x quantities (see
    # backend.data.portfolio_valuation) and overwrites this before the pipeline
    # runs. A client that derives this from purchase price is reporting cost
    # basis, which diverged from market value by 52% on the default portfolio
    # and silently mis-scaled every VaR, stress loss and contract count.
    total_notional: Optional[float] = Field(default=None, gt=0)

    hedge_horizon_days: int = Field(default=180, ge=1, le=730)
    protection_level: float = Field(default=0.15, ge=0.01, le=0.50)
    max_hedge_cost_pct: float = Field(default=0.05, ge=0.005, le=0.20)
    upside_preservation: bool = True
    execution_mode: Literal["analyze", "paper", "live"] = "analyze"

    # Max positions accepted in one request — bounds fan-out of market-data calls.
    MAX_HOLDINGS: ClassVar[int] = 25

    @model_validator(mode="after")
    def validate_holdings(self) -> "PortfolioInput":
        if not self.holdings:
            raise ValueError("portfolio must have at least one position")
        if len(self.holdings) > self.MAX_HOLDINGS:
            raise ValueError(
                f"portfolio has {len(self.holdings)} positions; "
                f"maximum supported is {self.MAX_HOLDINGS}"
            )

        # Reject duplicate stock tickers: two entries for the same underlying
        # would be hedged twice and double-count the budget.
        seen: set = set()
        for h in self.holdings:
            key = (
                h.ticker if h.position_type == "stock"
                else (h.ticker, h.option_type, h.strike, h.expiry)
            )
            if key in seen:
                raise ValueError(f"duplicate position for {h.ticker}; combine them into one entry")
            seen.add(key)
        return self

    @property
    def cost_basis(self) -> float:
        """Sum of what the user paid — not current exposure."""
        total = 0.0
        for h in self.holdings:
            if isinstance(h, StockPosition):
                total += h.shares * h.purchase_price
            else:
                premium = h.premium_paid if h.premium_paid is not None else 0.0
                total += h.contracts * 100 * premium
        return total

    @property
    def stock_positions(self) -> List[StockPosition]:
        return [h for h in self.holdings if isinstance(h, StockPosition)]

    @property
    def options_positions(self) -> List[OptionsPosition]:
        return [h for h in self.holdings if isinstance(h, OptionsPosition)]


# ── Backwards-compat alias ─────────────────────────────────────────────────────
# L3 / L4 engines reference HoldingInput and PortfolioPosition — keep these
# pointing at StockPosition so existing engine code doesn't break.

HoldingInput = StockPosition


class PortfolioPosition(BaseModel):
    """Enriched holding after market data fetch."""
    ticker: str
    shares: float
    purchase_price: float
    current_price: float
    notional_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    asset_class: str
    weight_in_portfolio: float
