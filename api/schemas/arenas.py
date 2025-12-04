"""Arena and lookup table schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field

from api.schemas.common import TimestampMixin


# ============================================================================
# Lookup Table Schemas
# ============================================================================


class ExchangeResponse(BaseModel):
    """Exchange lookup response."""

    id: int = Field(description="Exchange ID")
    slug: str = Field(description="Exchange slug (e.g., 'binance', 'hyperliquid')")
    name: str = Field(description="Display name (e.g., 'Binance', 'Hyperliquid')")

    model_config = {"from_attributes": True}


class CoinResponse(BaseModel):
    """Coin/asset lookup response."""

    id: int = Field(description="Coin ID")
    symbol: str = Field(description="Coin symbol (e.g., 'BTC', 'USD')")
    name: str = Field(description="Coin name (e.g., 'Bitcoin', 'US Dollar')")

    model_config = {"from_attributes": True}


class MarketTypeResponse(BaseModel):
    """Market type lookup response."""

    id: int = Field(description="Market type ID")
    type: str = Field(description="Market type (e.g., 'perps', 'spot')")
    name: str = Field(description="Display name (e.g., 'Perpetuals', 'Spot')")

    model_config = {"from_attributes": True}


# ============================================================================
# Arena Schemas
# ============================================================================


class ArenaResponse(TimestampMixin):
    """Full arena response with nested lookup objects."""

    id: int = Field(description="Arena ID (Nailsage internal)")
    starlisting_id: int = Field(description="Kirby starlisting ID (external)")

    # Trading pair info
    trading_pair: str = Field(description="Trading pair (e.g., 'BTC/USD')")
    trading_pair_id: Optional[int] = Field(None, description="Kirby trading_pair_id")

    # Nested lookup objects
    coin: CoinResponse = Field(description="Base asset")
    quote: CoinResponse = Field(description="Quote asset")
    exchange: ExchangeResponse = Field(description="Exchange")
    market_type: MarketTypeResponse = Field(description="Market type")

    # Interval
    interval: str = Field(description="Candle interval (e.g., '15m')")
    interval_seconds: int = Field(description="Interval in seconds (e.g., 900)")

    # Status
    is_active: bool = Field(description="Whether arena is active/tradeable")
    last_synced_at: Optional[int] = Field(None, description="Last Kirby sync timestamp (Unix ms)")

    model_config = {"from_attributes": True}


class ArenaSummary(BaseModel):
    """Lightweight arena summary for embedding in other responses."""

    id: int = Field(description="Arena ID")
    starlisting_id: int = Field(description="Kirby starlisting ID")
    trading_pair: str = Field(description="Trading pair (e.g., 'BTC/USD')")
    interval: str = Field(description="Candle interval (e.g., '15m')")

    # Flattened lookup values with names
    coin: str = Field(description="Base asset symbol (e.g., 'BTC')")
    coin_name: str = Field(description="Base asset name (e.g., 'Bitcoin')")
    quote: str = Field(description="Quote asset symbol (e.g., 'USD')")
    quote_name: str = Field(description="Quote asset name (e.g., 'US Dollar')")
    exchange: str = Field(description="Exchange slug (e.g., 'hyperliquid')")
    exchange_name: str = Field(description="Exchange name (e.g., 'Hyperliquid')")
    market_type: str = Field(description="Market type (e.g., 'perps')")
    market_name: str = Field(description="Market type name (e.g., 'Perpetuals')")


# ============================================================================
# List Responses
# ============================================================================


class ArenaListResponse(BaseModel):
    """Response for arena list endpoint."""

    arenas: List[ArenaResponse] = Field(description="List of arenas")
    total: int = Field(description="Total number of arenas")


class ExchangeListResponse(BaseModel):
    """Response for exchange list endpoint."""

    exchanges: List[ExchangeResponse] = Field(description="List of exchanges")
    total: int = Field(description="Total number of exchanges")


class CoinListResponse(BaseModel):
    """Response for coin list endpoint."""

    coins: List[CoinResponse] = Field(description="List of coins")
    total: int = Field(description="Total number of coins")


class MarketTypeListResponse(BaseModel):
    """Response for market type list endpoint."""

    market_types: List[MarketTypeResponse] = Field(description="List of market types")
    total: int = Field(description="Total number of market types")


# ============================================================================
# Sync Schemas
# ============================================================================


class ArenaSyncRequest(BaseModel):
    """Request to sync arena from Kirby API."""

    starlisting_id: int = Field(description="Kirby starlisting ID to fetch and cache")


class ArenaSyncResponse(BaseModel):
    """Response after syncing arena from Kirby."""

    arena: ArenaResponse = Field(description="Synced arena metadata")
    created: bool = Field(description="True if newly created, False if updated")
