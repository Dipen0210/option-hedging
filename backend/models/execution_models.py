from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class OrderInput(BaseModel):
    ticker: str
    instrument_type: str            # "option" | "future" | "etf"
    action: Literal["buy", "sell"]
    quantity: int = Field(..., ge=1)
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = None

    # Options-specific
    strike: Optional[float] = None
    expiry: Optional[str] = None    # "YYYY-MM-DD"
    option_right: Optional[Literal["put", "call"]] = None

    execution_mode: Literal["paper", "live"] = "paper"
    confirmed: bool = False         # must be True for live trades


class ExecutionResult(BaseModel):
    order_id: str
    status: Literal["filled", "pending", "rejected", "cancelled"]
    ticker: str
    quantity: int
    filled_price: Optional[float] = None
    total_cost: Optional[float] = None
    execution_mode: str
    timestamp: datetime
    broker_message: str = ""


class PaperPosition(BaseModel):
    """Tracks open paper trade positions in SQLite."""
    id: int
    ticker: str
    instrument_type: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    opened_at: datetime
    initial_delta: float = 0.0
    initial_vix: float = 0.0
