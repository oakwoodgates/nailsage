"""Portfolio endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_portfolio_service
from api.services.portfolio_service import PortfolioService
from api.schemas.portfolio import (
    PortfolioSummary,
    AllocationResponse,
    ExposureSummary,
    EquityHistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio"])


@router.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
):
    """Get complete portfolio overview.

    Returns comprehensive portfolio metrics including:
    - Capital (initial, total, available, allocated)
    - P&L (total, realized, unrealized)
    - Position metrics
    - Exposure metrics
    - Activity metrics
    """
    try:
        return portfolio_service.get_summary()
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/allocation", response_model=AllocationResponse)
async def get_portfolio_allocation(
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
):
    """Get capital allocation by strategy.

    Returns breakdown of capital allocated to each active strategy.
    """
    try:
        return portfolio_service.get_allocation()
    except Exception as e:
        logger.error(f"Error getting portfolio allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exposure", response_model=ExposureSummary)
async def get_portfolio_exposure(
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
):
    """Get current market exposure.

    Returns:
    - Long/short exposure
    - Net/gross exposure
    - Exposure as percentage of equity
    - Breakdown by asset (starlisting)
    """
    try:
        return portfolio_service.get_exposure()
    except Exception as e:
        logger.error(f"Error getting portfolio exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=EquityHistoryResponse)
async def get_equity_history(
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of data points"),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
):
    """Get equity curve data.

    Returns historical equity snapshots for charting the equity curve.
    Includes drawdown calculations.

    Args:
        limit: Maximum number of data points to return
    """
    try:
        return portfolio_service.get_equity_history(limit=limit)
    except Exception as e:
        logger.error(f"Error getting equity history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
