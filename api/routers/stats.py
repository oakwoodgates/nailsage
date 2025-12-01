"""Statistics endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_stats_service
from api.services.stats_service import StatsService
from api.schemas.stats import (
    FinancialSummary,
    DailyPnLResponse,
    StrategyStats,
    LeaderboardResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stats", tags=["Statistics"])


@router.get("/summary", response_model=FinancialSummary)
async def get_financial_summary(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    stats_service: StatsService = Depends(get_stats_service),
):
    """Get comprehensive financial summary.

    Returns all financial metrics including:
    - Equity and P&L
    - Win/loss counts and rates
    - Average win/loss amounts
    - Profit factor and expectancy
    - Fees paid

    Args:
        strategy_id: Optional filter for specific strategy
    """
    try:
        return stats_service.get_financial_summary(strategy_id=strategy_id)
    except Exception as e:
        logger.error(f"Error getting financial summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily", response_model=DailyPnLResponse)
async def get_daily_pnl(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    stats_service: StatsService = Depends(get_stats_service),
):
    """Get daily P&L breakdown.

    Returns P&L aggregated by day for the specified period.
    Includes cumulative P&L and daily win/loss counts.

    Args:
        days: Number of days to include (max 365)
        strategy_id: Optional filter for specific strategy
    """
    try:
        return stats_service.get_daily_pnl(days=days, strategy_id=strategy_id)
    except Exception as e:
        logger.error(f"Error getting daily P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-strategy", response_model=List[StrategyStats])
async def get_stats_by_strategy(
    stats_service: StatsService = Depends(get_stats_service),
):
    """Get statistics grouped by strategy.

    Returns performance metrics for each strategy including:
    - Total P&L
    - Trade counts
    - Win rate
    - Profit factor
    """
    try:
        return stats_service.get_stats_by_strategy()
    except Exception as e:
        logger.error(f"Error getting stats by strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    metric: str = Query(
        "total_pnl",
        description="Metric to rank by (total_pnl, win_rate, profit_factor)"
    ),
    limit: int = Query(10, ge=1, le=100, description="Number of entries to return"),
    stats_service: StatsService = Depends(get_stats_service),
):
    """Get strategy leaderboard.

    Returns strategies ranked by the specified metric.

    Args:
        metric: Ranking metric (total_pnl, win_rate, profit_factor)
        limit: Maximum number of entries
    """
    try:
        return stats_service.get_leaderboard(metric=metric, limit=limit)
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
