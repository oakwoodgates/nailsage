"""Portfolio-related schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PortfolioSummary(BaseModel):
    """Complete portfolio overview."""

    # Capital metrics
    initial_capital_usd: float = Field(description="Initial capital in USD")
    total_equity_usd: float = Field(description="Current total equity in USD")
    available_capital_usd: float = Field(description="Available (unreserved) capital in USD")
    allocated_capital_usd: float = Field(description="Capital allocated to positions in USD")

    # P&L metrics
    total_pnl_usd: float = Field(description="Total P&L in USD")
    total_pnl_pct: float = Field(description="Total P&L percentage")
    realized_pnl_usd: float = Field(description="Realized P&L in USD")
    unrealized_pnl_usd: float = Field(description="Unrealized P&L in USD")

    # Position metrics
    total_open_positions: int = Field(description="Number of open positions")
    max_positions: int = Field(description="Maximum allowed positions")
    position_utilization_pct: float = Field(description="Position utilization percentage")

    # Exposure metrics
    total_exposure_usd: float = Field(description="Total market exposure in USD")
    long_exposure_usd: float = Field(description="Long exposure in USD")
    short_exposure_usd: float = Field(description="Short exposure in USD")
    net_exposure_usd: float = Field(description="Net exposure (long - short) in USD")

    # Activity metrics
    active_strategies: int = Field(description="Number of active strategies")
    total_strategies: int = Field(description="Total number of strategies")
    trades_today: int = Field(description="Number of trades today")

    # Fees
    total_fees_paid: float = Field(description="Total fees paid in USD")

    # Timestamp
    timestamp: int = Field(description="Snapshot timestamp (Unix ms)")


class AllocationItem(BaseModel):
    """Capital allocation for a single strategy."""

    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    allocated_usd: float = Field(description="Allocated capital in USD")
    allocation_pct: float = Field(description="Allocation percentage of total")
    open_positions: int = Field(description="Number of open positions")
    unrealized_pnl_usd: float = Field(description="Unrealized P&L in USD")


class AllocationResponse(BaseModel):
    """Response for allocation endpoint."""

    allocations: List[AllocationItem] = Field(description="Allocation by strategy")
    total_allocated_usd: float = Field(description="Total allocated capital in USD")
    unallocated_usd: float = Field(description="Unallocated capital in USD")


class ExposureSummary(BaseModel):
    """Current market exposure summary."""

    total_exposure_usd: float = Field(description="Total market exposure in USD")
    long_exposure_usd: float = Field(description="Long exposure in USD")
    short_exposure_usd: float = Field(description="Short exposure in USD")
    net_exposure_usd: float = Field(description="Net exposure (long - short) in USD")
    gross_exposure_usd: float = Field(description="Gross exposure (long + short) in USD")
    exposure_pct_of_equity: float = Field(description="Exposure as percentage of equity")

    # Per-asset breakdown (optional)
    by_starlisting: Optional[List[dict]] = Field(
        None,
        description="Exposure breakdown by starlisting"
    )


class EquityPoint(BaseModel):
    """Single point on the equity curve."""

    timestamp: int = Field(description="Timestamp (Unix ms)")
    equity_usd: float = Field(description="Total equity in USD")
    realized_pnl_usd: float = Field(description="Cumulative realized P&L")
    unrealized_pnl_usd: float = Field(description="Unrealized P&L at this point")
    open_positions: int = Field(description="Number of open positions")


class EquityHistoryResponse(BaseModel):
    """Response for equity history endpoint."""

    points: List[EquityPoint] = Field(description="Equity curve data points")
    initial_capital_usd: float = Field(description="Initial capital in USD")
    current_equity_usd: float = Field(description="Current equity in USD")
    max_equity_usd: float = Field(description="Maximum equity reached")
    min_equity_usd: float = Field(description="Minimum equity reached")
    max_drawdown_usd: float = Field(description="Maximum drawdown in USD")
    max_drawdown_pct: float = Field(description="Maximum drawdown percentage")
