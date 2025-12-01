"""Trade endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_trade_service
from api.services.trade_service import TradeService
from api.schemas.trades import TradeResponse, TradeListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/trades", tags=["Trades"])


@router.get("", response_model=TradeListResponse)
async def list_trades(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    limit: int = Query(100, ge=1, le=1000, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    trade_service: TradeService = Depends(get_trade_service),
):
    """List trades with optional filtering.

    Args:
        strategy_id: Filter by strategy ID
        limit: Maximum number of trades
        offset: Pagination offset

    Returns:
        Paginated list of trades
    """
    try:
        return trade_service.get_trades(
            strategy_id=strategy_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error listing trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent", response_model=List[TradeResponse])
async def get_recent_trades(
    limit: int = Query(10, ge=1, le=100, description="Number of trades to return"),
    trade_service: TradeService = Depends(get_trade_service),
):
    """Get most recent trades.

    Args:
        limit: Number of trades to return

    Returns:
        List of recent trades
    """
    try:
        return trade_service.get_recent_trades(limit=limit)
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(
    trade_id: int,
    trade_service: TradeService = Depends(get_trade_service),
):
    """Get trade by ID.

    Args:
        trade_id: Trade ID

    Returns:
        Trade details
    """
    try:
        trade = trade_service.get_trade_by_id(trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        return trade
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trade {trade_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
