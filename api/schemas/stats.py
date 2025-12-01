"""Statistics and leaderboard schemas."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


Timeframe = Literal["daily", "weekly", "monthly", "all_time"]


class FinancialSummary(BaseModel):
    """Comprehensive financial statistics."""

    # Equity
    total_equity_usd: float = Field(description="Total equity in USD")

    # P&L
    total_pnl_usd: float = Field(description="Total P&L in USD")
    total_pnl_pct: float = Field(description="Total P&L percentage")
    realized_pnl_usd: float = Field(description="Realized P&L in USD")
    unrealized_pnl_usd: float = Field(description="Unrealized P&L in USD")

    # Win/Loss counts
    total_trades: int = Field(description="Total number of closed trades")
    total_wins: int = Field(description="Number of winning trades")
    total_losses: int = Field(description="Number of losing trades")
    win_rate: float = Field(description="Win rate (0-100)")

    # Win/Loss amounts
    avg_win_usd: float = Field(description="Average winning trade in USD")
    avg_loss_usd: float = Field(description="Average losing trade in USD")
    largest_win_usd: float = Field(description="Largest winning trade in USD")
    largest_loss_usd: float = Field(description="Largest losing trade in USD")

    # Risk metrics
    profit_factor: float = Field(description="Profit factor (gross profit / gross loss)")
    avg_trade_usd: float = Field(description="Average trade P&L in USD")
    expectancy_usd: float = Field(description="Expected value per trade in USD")

    # Drawdown
    max_drawdown_usd: float = Field(default=0.0, description="Maximum drawdown in USD")
    max_drawdown_pct: float = Field(default=0.0, description="Maximum drawdown percentage")

    # Fees
    total_fees_paid: float = Field(description="Total fees paid in USD")

    # Timestamp
    timestamp: int = Field(description="Calculation timestamp (Unix ms)")


class DailyPnL(BaseModel):
    """Daily P&L record."""

    date: str = Field(description="Date (YYYY-MM-DD)")
    timestamp_start: int = Field(description="Day start timestamp (Unix ms)")
    timestamp_end: int = Field(description="Day end timestamp (Unix ms)")

    # P&L
    pnl_usd: float = Field(description="P&L for the day in USD")
    pnl_pct: float = Field(description="P&L percentage for the day")
    cumulative_pnl_usd: float = Field(description="Cumulative P&L up to this day")

    # Activity
    trades: int = Field(description="Number of trades on this day")
    wins: int = Field(description="Number of winning trades")
    losses: int = Field(description="Number of losing trades")

    # Equity
    starting_equity_usd: float = Field(description="Equity at start of day")
    ending_equity_usd: float = Field(description="Equity at end of day")


class DailyPnLResponse(BaseModel):
    """Response for daily P&L endpoint."""

    daily: List[DailyPnL] = Field(description="Daily P&L records")
    total_days: int = Field(description="Total number of trading days")
    profitable_days: int = Field(description="Number of profitable days")
    losing_days: int = Field(description="Number of losing days")
    avg_daily_pnl_usd: float = Field(description="Average daily P&L in USD")
    best_day_usd: float = Field(description="Best day P&L in USD")
    worst_day_usd: float = Field(description="Worst day P&L in USD")


class StrategyStats(BaseModel):
    """Statistics for a single strategy."""

    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    is_active: bool = Field(description="Whether strategy is active")

    # P&L
    total_pnl_usd: float = Field(description="Total P&L in USD")
    realized_pnl_usd: float = Field(description="Realized P&L in USD")
    unrealized_pnl_usd: float = Field(description="Unrealized P&L in USD")

    # Trade stats
    total_trades: int = Field(description="Total number of trades")
    open_positions: int = Field(description="Number of open positions")
    win_count: int = Field(description="Number of winning trades")
    loss_count: int = Field(description="Number of losing trades")
    win_rate: float = Field(description="Win rate (0-100)")

    # Averages
    avg_win_usd: float = Field(description="Average winning trade in USD")
    avg_loss_usd: float = Field(description="Average losing trade in USD")
    profit_factor: float = Field(description="Profit factor")


class LeaderboardEntry(BaseModel):
    """Leaderboard entry for a strategy."""

    rank: int = Field(description="Ranking position")
    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    interval: str = Field(description="Trading interval")

    # Key metrics
    total_pnl_usd: float = Field(description="Total P&L in USD")
    total_pnl_pct: float = Field(description="Total P&L percentage")
    win_rate: float = Field(description="Win rate (0-100)")
    total_trades: int = Field(description="Total number of trades")
    profit_factor: float = Field(description="Profit factor")

    # Optional detailed metrics
    avg_trade_usd: Optional[float] = Field(None, description="Average trade P&L")
    max_drawdown_pct: Optional[float] = Field(None, description="Max drawdown percentage")


class LeaderboardResponse(BaseModel):
    """Response for leaderboard endpoint."""

    entries: List[LeaderboardEntry] = Field(description="Leaderboard entries")
    metric: str = Field(description="Metric used for ranking")
    total_strategies: int = Field(description="Total number of strategies")
