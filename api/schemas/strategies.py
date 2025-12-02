"""Strategy-related schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field

from api.schemas.common import TimestampMixin


class StrategyBase(BaseModel):
    """Base strategy fields."""

    strategy_name: str = Field(description="Strategy name")
    version: str = Field(description="Strategy version")
    starlisting_id: int = Field(description="Kirby starlisting ID")
    interval: str = Field(description="Trading interval (1m, 15m, 4h, etc.)")
    model_id: Optional[str] = Field(None, description="ML model identifier")
    is_active: bool = Field(default=True, description="Whether strategy is active")


class StrategyResponse(StrategyBase, TimestampMixin):
    """Strategy response with all fields."""

    id: int = Field(description="Strategy ID")

    model_config = {"from_attributes": True}


class StrategyWithStats(StrategyResponse):
    """Strategy response with computed performance statistics."""

    # Trade counts
    total_trades: int = Field(default=0, description="Total number of trades")
    open_positions: int = Field(default=0, description="Number of open positions")

    # P&L metrics
    total_pnl_usd: float = Field(default=0.0, description="Total P&L in USD")
    total_pnl_pct: float = Field(default=0.0, description="Total P&L percentage")
    realized_pnl_usd: float = Field(default=0.0, description="Realized P&L in USD")
    unrealized_pnl_usd: float = Field(default=0.0, description="Unrealized P&L in USD")

    # Win/loss metrics
    win_count: int = Field(default=0, description="Number of winning trades")
    loss_count: int = Field(default=0, description="Number of losing trades")
    win_rate: float = Field(default=0.0, description="Win rate (0-100)")

    # Additional metrics
    avg_win_usd: float = Field(default=0.0, description="Average winning trade in USD")
    avg_loss_usd: float = Field(default=0.0, description="Average losing trade in USD")
    profit_factor: float = Field(default=0.0, description="Profit factor (gross profit / gross loss)")


class StrategyListResponse(BaseModel):
    """Response for strategy list endpoint."""

    strategies: List[StrategyResponse] = Field(description="List of strategies")
    total: int = Field(description="Total number of strategies")
