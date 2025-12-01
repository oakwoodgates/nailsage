"""Position-related schemas."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from api.schemas.common import TimestampMixin


PositionSide = Literal["long", "short"]
PositionStatus = Literal["open", "closed", "liquidated"]
ExitReason = Literal["signal", "stop_loss", "take_profit", "manual"]


class PositionBase(BaseModel):
    """Base position fields."""

    strategy_id: int = Field(description="Strategy ID")
    starlisting_id: int = Field(description="Kirby starlisting ID")
    side: PositionSide = Field(description="Position side (long/short)")
    size: float = Field(description="Position size in USDT")
    entry_price: float = Field(description="Entry price")
    entry_timestamp: int = Field(description="Entry timestamp (Unix ms)")


class PositionResponse(PositionBase, TimestampMixin):
    """Position response with all fields."""

    id: int = Field(description="Position ID")

    # Exit fields (None for open positions)
    exit_price: Optional[float] = Field(None, description="Exit price")
    exit_timestamp: Optional[int] = Field(None, description="Exit timestamp (Unix ms)")
    exit_reason: Optional[ExitReason] = Field(None, description="Exit reason")

    # P&L fields
    realized_pnl: Optional[float] = Field(None, description="Realized P&L in USDT")
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized P&L in USDT")
    fees_paid: float = Field(default=0.0, description="Total fees paid in USDT")

    # Status and risk management
    status: PositionStatus = Field(description="Position status")
    stop_loss_price: Optional[float] = Field(None, description="Stop loss price")
    take_profit_price: Optional[float] = Field(None, description="Take profit price")

    # Enriched fields (optional)
    strategy_name: Optional[str] = Field(None, description="Strategy name")
    duration_minutes: Optional[float] = Field(None, description="Position duration in minutes")
    pnl_pct: Optional[float] = Field(None, description="P&L percentage")

    model_config = {"from_attributes": True}


class PositionListResponse(BaseModel):
    """Response for position list endpoint."""

    positions: List[PositionResponse] = Field(description="List of positions")
    total: int = Field(description="Total number of positions")
    limit: int = Field(description="Limit used for this request")
    offset: int = Field(description="Offset used for this request")
