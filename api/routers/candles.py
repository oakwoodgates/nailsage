"""REST endpoints for candle/price data.

Provides historical candle data for charting.
"""

import logging
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.config import get_config, APIConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/candles", tags=["candles"])


class CandleResponse(BaseModel):
    """Single candle data."""

    time: str = Field(..., description="Candle timestamp (ISO format)")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., description="Volume")


class CandlesListResponse(BaseModel):
    """Response for candles list endpoint."""

    starlisting_id: int = Field(..., description="Starlisting ID")
    interval: str = Field(..., description="Candle interval")
    count: int = Field(..., description="Number of candles returned")
    candles: List[CandleResponse] = Field(..., description="Candle data")


@router.get(
    "/{starlisting_id}",
    response_model=CandlesListResponse,
    summary="Get historical candles",
    description="""
    Fetch historical candles for a starlisting.

    This endpoint fetches data from the Kirby API.
    For real-time updates, use the WebSocket prices channel.
    """,
)
async def get_candles(
    starlisting_id: int,
    interval: str = Query(default="15m", description="Candle interval (e.g., 1m, 5m, 15m, 1h, 4h, 1d)"),
    limit: int = Query(default=500, ge=1, le=1000, description="Number of candles to return"),
    config: APIConfig = Depends(get_config),
) -> CandlesListResponse:
    """Get historical candles for a starlisting.

    Args:
        starlisting_id: Kirby starlisting ID
        interval: Candle interval
        limit: Maximum number of candles to return
        config: API configuration

    Returns:
        List of historical candles

    Raises:
        HTTPException: If Kirby API is not configured or request fails
    """
    if not config.kirby_ws_url or not config.kirby_api_key:
        raise HTTPException(
            status_code=503,
            detail="Kirby API not configured - candles endpoint unavailable",
        )

    # Convert WebSocket URL to REST API URL
    # ws://kirby.example.com/ws -> http://kirby.example.com/api
    kirby_base_url = config.kirby_ws_url.replace("wss://", "https://").replace("ws://", "http://")
    if kirby_base_url.endswith("/ws"):
        kirby_base_url = kirby_base_url[:-3]

    # Build Kirby API URL for historical candles
    # Note: This assumes Kirby has a REST endpoint for historical data
    # Adjust the URL pattern based on actual Kirby API
    kirby_url = f"{kirby_base_url}/api/v1/candles/{starlisting_id}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                kirby_url,
                params={"interval": interval, "limit": limit},
                headers={"Authorization": f"Bearer {config.kirby_api_key}"},
            )

            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Starlisting {starlisting_id} not found",
                )

            if response.status_code != 200:
                logger.error(f"Kirby API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=502,
                    detail="Failed to fetch candles from Kirby API",
                )

            data = response.json()

            # Parse response - adjust based on actual Kirby response format
            candles = data.get("candles", data.get("data", []))

            return CandlesListResponse(
                starlisting_id=starlisting_id,
                interval=interval,
                count=len(candles),
                candles=[
                    CandleResponse(
                        time=c.get("time", ""),
                        open=float(c.get("open", 0)),
                        high=float(c.get("high", 0)),
                        low=float(c.get("low", 0)),
                        close=float(c.get("close", 0)),
                        volume=float(c.get("volume", 0)),
                    )
                    for c in candles
                ],
            )

    except httpx.TimeoutException:
        logger.error("Timeout fetching candles from Kirby")
        raise HTTPException(
            status_code=504,
            detail="Timeout fetching candles from Kirby API",
        )
    except httpx.RequestError as e:
        logger.error(f"Error fetching candles from Kirby: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Kirby API",
        )
