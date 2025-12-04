"""Arena and lookup table endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_arena_service
from api.services.arena_service import ArenaService
from api.schemas.arenas import (
    ArenaResponse,
    ArenaListResponse,
    ArenaSummary,
    ArenaSyncRequest,
    ArenaSyncResponse,
    ExchangeListResponse,
    CoinListResponse,
    MarketTypeListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Arenas"])


# =============================================================================
# Arena Endpoints
# =============================================================================


@router.get("/arenas", response_model=ArenaListResponse)
async def list_arenas(
    active_only: bool = Query(True, description="Filter to active arenas only"),
    exchange_id: Optional[int] = Query(None, description="Filter by exchange ID"),
    coin_id: Optional[int] = Query(None, description="Filter by coin ID (base asset)"),
    arena_service: ArenaService = Depends(get_arena_service),
):
    """List all arenas with optional filters.

    Args:
        active_only: If True, return only active arenas
        exchange_id: Filter by exchange ID
        coin_id: Filter by coin ID (base asset)

    Returns:
        List of arenas with full metadata
    """
    try:
        arenas = arena_service.get_all_arenas(
            active_only=active_only,
            exchange_id=exchange_id,
            coin_id=coin_id,
        )
        return ArenaListResponse(arenas=arenas, total=len(arenas))
    except Exception as e:
        logger.error(f"Error listing arenas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/arenas/{arena_id}", response_model=ArenaResponse)
async def get_arena(
    arena_id: int,
    arena_service: ArenaService = Depends(get_arena_service),
):
    """Get arena by ID.

    Args:
        arena_id: Arena ID (Nailsage internal ID)

    Returns:
        Arena details with nested lookup objects
    """
    try:
        arena = arena_service.get_arena_by_id(arena_id)
        if not arena:
            raise HTTPException(status_code=404, detail="Arena not found")
        return arena
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting arena {arena_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/arenas/by-starlisting/{starlisting_id}", response_model=ArenaResponse)
async def get_arena_by_starlisting(
    starlisting_id: int,
    arena_service: ArenaService = Depends(get_arena_service),
):
    """Get arena by Kirby starlisting ID.

    Args:
        starlisting_id: Kirby starlisting ID

    Returns:
        Arena details
    """
    try:
        arena = arena_service.get_arena_by_starlisting(starlisting_id)
        if not arena:
            raise HTTPException(status_code=404, detail="Arena not found for starlisting")
        return arena
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting arena by starlisting {starlisting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/arenas/sync", response_model=ArenaSyncResponse)
async def sync_arena(
    request: ArenaSyncRequest,
    arena_service: ArenaService = Depends(get_arena_service),
):
    """Sync arena metadata from Kirby API.

    Fetches starlisting metadata from Kirby and creates/updates
    the local arena record along with any required lookup records.

    Args:
        request: Sync request with starlisting_id

    Returns:
        Synced arena with created flag
    """
    try:
        arena, created = await arena_service.sync_from_kirby(request.starlisting_id)
        return ArenaSyncResponse(arena=arena, created=created)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing arena for starlisting {request.starlisting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Lookup Table Endpoints (for dropdowns/filters)
# =============================================================================


@router.get("/exchanges", response_model=ExchangeListResponse)
async def list_exchanges(
    arena_service: ArenaService = Depends(get_arena_service),
):
    """List all exchanges.

    Returns:
        List of exchanges for dropdown/filter UI
    """
    try:
        exchanges = arena_service.get_all_exchanges()
        return ExchangeListResponse(exchanges=exchanges, total=len(exchanges))
    except Exception as e:
        logger.error(f"Error listing exchanges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coins", response_model=CoinListResponse)
async def list_coins(
    arena_service: ArenaService = Depends(get_arena_service),
):
    """List all coins/assets.

    Returns:
        List of coins for dropdown/filter UI
    """
    try:
        coins = arena_service.get_all_coins()
        return CoinListResponse(coins=coins, total=len(coins))
    except Exception as e:
        logger.error(f"Error listing coins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-types", response_model=MarketTypeListResponse)
async def list_market_types(
    arena_service: ArenaService = Depends(get_arena_service),
):
    """List all market types.

    Returns:
        List of market types for dropdown/filter UI
    """
    try:
        market_types = arena_service.get_all_market_types()
        return MarketTypeListResponse(market_types=market_types, total=len(market_types))
    except Exception as e:
        logger.error(f"Error listing market types: {e}")
        raise HTTPException(status_code=500, detail=str(e))
