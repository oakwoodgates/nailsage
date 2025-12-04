"""Strategy-related schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field

from api.schemas.common import TimestampMixin
from api.schemas.models import ModelSummary
from api.schemas.arenas import ArenaSummary


class StrategyBase(BaseModel):
    """Base strategy fields."""

    strategy_name: str = Field(description="Strategy name")
    version: str = Field(description="Strategy version")
    starlisting_id: int = Field(description="Kirby starlisting ID")
    arena_id: Optional[int] = Field(None, description="Arena ID (Nailsage internal)")
    interval: str = Field(description="Trading interval (1m, 15m, 4h, etc.)")
    model_id: Optional[str] = Field(None, description="ML model identifier")
    is_active: bool = Field(default=True, description="Whether strategy is active")
    initial_bankroll: float = Field(default=10000.0, description="Initial capital for this strategy (USDT)")
    current_bankroll: float = Field(default=10000.0, description="Current capital after P&L (USDT)")


class StrategyResponse(StrategyBase, TimestampMixin):
    """Strategy response with all fields."""

    id: int = Field(description="Strategy ID")
    arena: Optional[ArenaSummary] = Field(None, description="Arena metadata (if loaded)")

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


class StrategyWithModel(StrategyResponse):
    """Strategy response with active model metadata embedded."""

    model: Optional[ModelSummary] = Field(None, description="Active model for this strategy")


class StrategyListResponse(BaseModel):
    """Response for strategy list endpoint."""

    strategies: List[StrategyResponse] = Field(description="List of strategies")
    total: int = Field(description="Total number of strategies")


class BankrollResponse(BaseModel):
    """Response for bankroll endpoint."""

    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    initial_bankroll: float = Field(description="Initial capital (USDT)")
    current_bankroll: float = Field(description="Current capital after P&L (USDT)")
    pnl: float = Field(description="Total P&L (current - initial)")
    pnl_pct: float = Field(description="P&L percentage")
    is_active: bool = Field(description="Whether strategy can trade (bankroll > 0)")


class BankrollUpdateRequest(BaseModel):
    """Request to update strategy bankroll."""

    bankroll: float = Field(description="New bankroll value (USDT)", gt=0)
