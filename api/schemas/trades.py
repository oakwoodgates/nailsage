"""Trade-related schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field

from api.schemas.common import TimestampMixin


class TradeBase(BaseModel):
    """Base trade fields."""

    position_id: int = Field(description="Associated position ID")
    strategy_id: int = Field(description="Strategy ID")
    starlisting_id: int = Field(description="Kirby starlisting ID")
    trade_type: str = Field(
        description="Trade type (open_long, open_short, close_long, close_short)"
    )
    size: float = Field(description="Trade size in USDT")
    price: float = Field(description="Execution price")
    fees: float = Field(description="Transaction fees in USDT")
    slippage: float = Field(description="Slippage in USDT")
    timestamp: int = Field(description="Trade execution timestamp (Unix ms)")
    signal_id: Optional[int] = Field(None, description="Associated signal ID")


class TradeResponse(TradeBase):
    """Trade response with all fields."""

    id: int = Field(description="Trade ID")
    created_at: Optional[int] = Field(None, description="Creation timestamp (Unix ms)")

    # Enriched fields (optional, populated when joining with other tables)
    strategy_name: Optional[str] = Field(None, description="Strategy name")
    position_side: Optional[str] = Field(None, description="Position side (long/short)")

    model_config = {"from_attributes": True}


class TradeListResponse(BaseModel):
    """Response for trade list endpoint."""

    trades: List[TradeResponse] = Field(description="List of trades")
    total: int = Field(description="Total number of trades")
    limit: int = Field(description="Limit used for this request")
    offset: int = Field(description="Offset used for this request")
