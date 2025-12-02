"""Position endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_position_service
from api.services.position_service import PositionService
from api.schemas.positions import PositionResponse, PositionListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/positions", tags=["Positions"])


@router.get("", response_model=PositionListResponse)
async def list_positions(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    status: Optional[str] = Query(None, description="Filter by status (open/closed)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of positions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    position_service: PositionService = Depends(get_position_service),
):
    """List positions with optional filtering.

    Args:
        strategy_id: Filter by strategy ID
        status: Filter by status (open/closed)
        limit: Maximum number of positions
        offset: Pagination offset

    Returns:
        Paginated list of positions
    """
    try:
        return position_service.get_positions(
            strategy_id=strategy_id,
            status=status,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error listing positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/open", response_model=List[PositionResponse])
async def get_open_positions(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    position_service: PositionService = Depends(get_position_service),
):
    """Get all open positions.

    Args:
        strategy_id: Filter by strategy ID

    Returns:
        List of open positions
    """
    try:
        return position_service.get_open_positions(strategy_id=strategy_id)
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/closed", response_model=PositionListResponse)
async def get_closed_positions(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    limit: int = Query(100, ge=1, le=1000, description="Number of positions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    position_service: PositionService = Depends(get_position_service),
):
    """Get closed positions (historical).

    Args:
        strategy_id: Filter by strategy ID
        limit: Maximum number of positions
        offset: Pagination offset

    Returns:
        Paginated list of closed positions
    """
    try:
        return position_service.get_closed_positions(
            strategy_id=strategy_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error getting closed positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{position_id}", response_model=PositionResponse)
async def get_position(
    position_id: int,
    position_service: PositionService = Depends(get_position_service),
):
    """Get position by ID.

    Args:
        position_id: Position ID

    Returns:
        Position details
    """
    try:
        position = position_service.get_position_by_id(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        return position
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
