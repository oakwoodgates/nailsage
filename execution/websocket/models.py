"""Data models for Kirby WebSocket API messages.

This module defines Pydantic models for all WebSocket message types:
- Client → Server: Subscribe requests
- Server → Client: Candle updates, funding rates, open interest, heartbeats, errors

All price/rate values from Kirby are 18-decimal precision strings.
We convert them to float for internal use.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client → Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Server → Client (Kirby's actual format)
    CANDLE = "candle"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUCCESS = "success"  # Subscription confirmation


# ============================================================================
# Core Data Models
# ============================================================================


class Candle(BaseModel):
    """
    OHLCV candle data from Kirby.

    Kirby sends candle data with "time" (datetime string) field.
    We convert it to timestamp (Unix milliseconds) for internal use.

    Attributes:
        time: Candle time (ISO format datetime string from Kirby)
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        num_trades: Number of trades (optional)
    """

    time: str = Field(..., description="Candle time (ISO datetime)")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    num_trades: Optional[int] = Field(default=None, description="Number of trades")

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def convert_decimal_string(cls, v: Union[str, float, Decimal]) -> float:
        """Convert 18-decimal precision string to float."""
        if isinstance(v, str):
            return float(v)
        elif isinstance(v, Decimal):
            return float(v)
        return v

    @property
    def datetime(self) -> datetime:
        """Get candle timestamp as datetime object."""
        from datetime import datetime as dt
        return dt.fromisoformat(self.time.replace('Z', '+00:00'))

    @property
    def timestamp(self) -> int:
        """Get candle timestamp in Unix milliseconds."""
        return int(self.datetime.timestamp() * 1000)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"Candle(time={self.time}, "
            f"O={self.open:.2f}, H={self.high:.2f}, L={self.low:.2f}, "
            f"C={self.close:.2f}, V={self.volume:.2f})"
        )


class FundingRate(BaseModel):
    """
    Funding rate data for perpetual futures.

    Attributes:
        timestamp: Timestamp (Unix milliseconds)
        funding_rate: Funding rate (8-hour rate)
        starlisting_id: Kirby starlisting ID
    """

    timestamp: int = Field(..., description="Unix timestamp (milliseconds)")
    funding_rate: float = Field(..., description="Funding rate (8-hour)")
    starlisting_id: int = Field(..., description="Kirby starlisting ID")

    @field_validator("funding_rate", mode="before")
    @classmethod
    def convert_decimal_string(cls, v: Union[str, float, Decimal]) -> float:
        """Convert 18-decimal precision string to float."""
        if isinstance(v, str):
            return float(v)
        elif isinstance(v, Decimal):
            return float(v)
        return v

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000.0)


class OpenInterest(BaseModel):
    """
    Open interest data for perpetual futures.

    Attributes:
        timestamp: Timestamp (Unix milliseconds)
        open_interest: Total open interest (contracts)
        starlisting_id: Kirby starlisting ID
    """

    timestamp: int = Field(..., description="Unix timestamp (milliseconds)")
    open_interest: float = Field(..., description="Total open interest")
    starlisting_id: int = Field(..., description="Kirby starlisting ID")

    @field_validator("open_interest", mode="before")
    @classmethod
    def convert_decimal_string(cls, v: Union[str, float, Decimal]) -> float:
        """Convert 18-decimal precision string to float."""
        if isinstance(v, str):
            return float(v)
        elif isinstance(v, Decimal):
            return float(v)
        return v

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000.0)


# ============================================================================
# Client → Server Messages
# ============================================================================


class SubscribeRequest(BaseModel):
    """
    Client request to subscribe to one or more starlistings.

    Example:
        {
            "action": "subscribe",
            "starlisting_ids": [1, 2],
            "historical_candles": 200
        }
    """

    action: Literal[MessageType.SUBSCRIBE] = MessageType.SUBSCRIBE
    starlisting_ids: List[int] = Field(..., description="List of starlisting IDs to subscribe to")
    historical_candles: Optional[int] = Field(
        default=None,
        description="Number of historical candles to request (max 1000)"
    )

    @field_validator("historical_candles")
    @classmethod
    def validate_historical_candles(cls, v: Optional[int]) -> Optional[int]:
        """Validate historical candles is within allowed range."""
        if v is not None:
            if v < 0:
                raise ValueError("historical_candles must be non-negative")
            if v > 1000:
                raise ValueError("historical_candles cannot exceed 1000")
        return v


class UnsubscribeRequest(BaseModel):
    """
    Client request to unsubscribe from one or more starlistings.

    Example:
        {
            "action": "unsubscribe",
            "starlisting_ids": [1]
        }
    """

    action: Literal[MessageType.UNSUBSCRIBE] = MessageType.UNSUBSCRIBE
    starlisting_ids: List[int] = Field(..., description="List of starlisting IDs to unsubscribe from")


# ============================================================================
# Server → Client Messages
# ============================================================================


class CandleUpdate(BaseModel):
    """
    Server message with candle update from Kirby.

    Example:
        {
            "type": "candle",
            "starlisting_id": 1,
            "exchange": "hyperliquid",
            "coin": "BTC",
            "quote": "USD",
            "trading_pair": "BTC/USD",
            "market_type": "perps",
            "interval": "1m",
            "data": {
                "time": "2025-11-19T17:04:00+00:00",
                "open": "89840.000000000000000000",
                "high": "89853.000000000000000000",
                "low": "89670.000000000000000000",
                "close": "89695.000000000000000000",
                "volume": "69.587600000000000000",
                "num_trades": 480
            }
        }
    """

    type: Literal[MessageType.CANDLE] = MessageType.CANDLE
    starlisting_id: int = Field(..., description="Starlisting ID")
    exchange: str = Field(..., description="Exchange name")
    coin: str = Field(..., description="Base coin")
    quote: str = Field(..., description="Quote currency")
    trading_pair: str = Field(..., description="Trading pair")
    market_type: str = Field(..., description="Market type (spot/perps)")
    interval: str = Field(..., description="Timeframe")
    data: Candle = Field(..., description="Candle data")

    @property
    def candle(self) -> Candle:
        """Get candle data (for backward compatibility)."""
        return self.data


class FundingRateUpdate(BaseModel):
    """
    Server message with funding rate update.

    Example:
        {
            "type": "funding",
            "starlisting_id": 1,
            "data": {
                "timestamp": 1700000000000,
                "funding_rate": "0.000100000000000000",
                "starlisting_id": 1
            }
        }
    """

    type: Literal[MessageType.FUNDING] = MessageType.FUNDING
    starlisting_id: int = Field(..., description="Starlisting ID")
    data: FundingRate = Field(..., description="Funding rate data", alias="funding_rate")

    @property
    def funding_rate(self) -> FundingRate:
        """Get funding rate data (for backward compatibility)."""
        return self.data


class OpenInterestUpdate(BaseModel):
    """
    Server message with open interest update.

    Example:
        {
            "type": "open_interest",
            "starlisting_id": 1,
            "data": {
                "timestamp": 1700000000000,
                "open_interest": "1234567.890123456789012345",
                "starlisting_id": 1
            }
        }
    """

    type: Literal[MessageType.OPEN_INTEREST] = MessageType.OPEN_INTEREST
    starlisting_id: int = Field(..., description="Starlisting ID")
    data: OpenInterest = Field(..., description="Open interest data", alias="open_interest")

    @property
    def open_interest(self) -> OpenInterest:
        """Get open interest data (for backward compatibility)."""
        return self.data


class Heartbeat(BaseModel):
    """
    Server heartbeat message (every 30 seconds).

    Example:
        {
            "type": "heartbeat",
            "timestamp": 1700000000000
        }
    """

    type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT
    timestamp: int = Field(..., description="Server timestamp (milliseconds)")

    @property
    def datetime(self) -> datetime:
        """Get heartbeat timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000.0)


class SuccessMessage(BaseModel):
    """
    Server success message (subscription confirmation).

    Example:
        {
            "type": "success",
            "message": "Subscribed successfully"
        }
    """

    type: Literal[MessageType.SUCCESS] = MessageType.SUCCESS
    message: str = Field(..., description="Success message")


class ErrorMessage(BaseModel):
    """
    Server error message.

    Kirby uses 'message' and 'code' fields, but we also support 'error' and 'details'
    for flexibility.

    Example:
        {
            "type": "error",
            "message": "Invalid starlisting_id",
            "code": "validation_error"
        }
    """

    type: Literal[MessageType.ERROR] = MessageType.ERROR
    message: Optional[str] = Field(default=None, description="Error message")
    error: Optional[str] = Field(default=None, description="Error message (alternative)")
    code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[str] = Field(default=None, description="Additional error details")

    @property
    def error_message(self) -> str:
        """Get error message from either 'message' or 'error' field."""
        return self.message or self.error or "Unknown error"


# ============================================================================
# Union Types for Message Parsing
# ============================================================================


# All possible server → client messages
ServerMessage = Union[
    CandleUpdate,
    FundingRateUpdate,
    OpenInterestUpdate,
    Heartbeat,
    SuccessMessage,
    ErrorMessage,
]

# All possible client → server messages
ClientMessage = Union[
    SubscribeRequest,
    UnsubscribeRequest,
]
