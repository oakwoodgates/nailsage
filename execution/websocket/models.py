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
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client → Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Server → Client
    CANDLE_UPDATE = "candle_update"
    FUNDING_RATE_UPDATE = "funding_rate_update"
    OPEN_INTEREST_UPDATE = "open_interest_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    UNSUBSCRIPTION_CONFIRMED = "unsubscription_confirmed"


# ============================================================================
# Core Data Models
# ============================================================================


class Candle(BaseModel):
    """
    OHLCV candle data.

    Attributes:
        timestamp: Candle timestamp (Unix milliseconds)
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        starlisting_id: Kirby starlisting ID
        interval: Timeframe (e.g., "15m", "4h")
    """

    timestamp: int = Field(..., description="Unix timestamp (milliseconds)")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    starlisting_id: int = Field(..., description="Kirby starlisting ID")
    interval: str = Field(..., description="Timeframe (e.g., 15m, 4h)")

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def convert_decimal_string(cls, v: Union[str, float, Decimal]) -> float:
        """Convert 18-decimal precision string to float."""
        if isinstance(v, str):
            return float(v)
        elif isinstance(v, Decimal):
            return float(v)
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate timestamp is reasonable (not in distant past/future)."""
        if v < 1_000_000_000_000:  # Before ~2001 (milliseconds)
            raise ValueError("Timestamp appears to be in seconds, not milliseconds")
        if v > 2_000_000_000_000:  # After ~2033 (milliseconds)
            raise ValueError("Timestamp is too far in the future")
        return v

    @property
    def datetime(self) -> datetime:
        """Get candle timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000.0)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"Candle(timestamp={self.datetime.isoformat()}, "
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
    Client request to subscribe to a starlisting.

    Example:
        {
            "action": "subscribe",
            "starlisting_id": 1,
            "historical_candles": 200
        }
    """

    action: Literal[MessageType.SUBSCRIBE] = MessageType.SUBSCRIBE
    starlisting_id: int = Field(..., description="Starlisting ID to subscribe to")
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
    Client request to unsubscribe from a starlisting.

    Example:
        {
            "action": "unsubscribe",
            "starlisting_id": 1
        }
    """

    action: Literal[MessageType.UNSUBSCRIBE] = MessageType.UNSUBSCRIBE
    starlisting_id: int = Field(..., description="Starlisting ID to unsubscribe from")


# ============================================================================
# Server → Client Messages
# ============================================================================


class CandleUpdate(BaseModel):
    """
    Server message with candle update.

    Example:
        {
            "type": "candle_update",
            "starlisting_id": 1,
            "candle": {
                "timestamp": 1700000000000,
                "open": "42000.123456789012345678",
                "high": "42100.123456789012345678",
                "low": "41900.123456789012345678",
                "close": "42050.123456789012345678",
                "volume": "123.456789012345678901",
                "starlisting_id": 1,
                "interval": "15m"
            }
        }
    """

    type: Literal[MessageType.CANDLE_UPDATE] = MessageType.CANDLE_UPDATE
    starlisting_id: int = Field(..., description="Starlisting ID")
    candle: Candle = Field(..., description="Candle data")


class FundingRateUpdate(BaseModel):
    """
    Server message with funding rate update.

    Example:
        {
            "type": "funding_rate_update",
            "starlisting_id": 1,
            "funding_rate": {
                "timestamp": 1700000000000,
                "funding_rate": "0.000100000000000000",
                "starlisting_id": 1
            }
        }
    """

    type: Literal[MessageType.FUNDING_RATE_UPDATE] = MessageType.FUNDING_RATE_UPDATE
    starlisting_id: int = Field(..., description="Starlisting ID")
    funding_rate: FundingRate = Field(..., description="Funding rate data")


class OpenInterestUpdate(BaseModel):
    """
    Server message with open interest update.

    Example:
        {
            "type": "open_interest_update",
            "starlisting_id": 1,
            "open_interest": {
                "timestamp": 1700000000000,
                "open_interest": "1234567.890123456789012345",
                "starlisting_id": 1
            }
        }
    """

    type: Literal[MessageType.OPEN_INTEREST_UPDATE] = MessageType.OPEN_INTEREST_UPDATE
    starlisting_id: int = Field(..., description="Starlisting ID")
    open_interest: OpenInterest = Field(..., description="Open interest data")


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


class SubscriptionConfirmed(BaseModel):
    """
    Server confirmation of subscription.

    Example:
        {
            "type": "subscription_confirmed",
            "starlisting_id": 1,
            "message": "Subscribed to starlisting 1"
        }
    """

    type: Literal[MessageType.SUBSCRIPTION_CONFIRMED] = MessageType.SUBSCRIPTION_CONFIRMED
    starlisting_id: int = Field(..., description="Starlisting ID")
    message: str = Field(..., description="Confirmation message")


class UnsubscriptionConfirmed(BaseModel):
    """
    Server confirmation of unsubscription.

    Example:
        {
            "type": "unsubscription_confirmed",
            "starlisting_id": 1,
            "message": "Unsubscribed from starlisting 1"
        }
    """

    type: Literal[MessageType.UNSUBSCRIPTION_CONFIRMED] = MessageType.UNSUBSCRIPTION_CONFIRMED
    starlisting_id: int = Field(..., description="Starlisting ID")
    message: str = Field(..., description="Confirmation message")


class ErrorMessage(BaseModel):
    """
    Server error message.

    Example:
        {
            "type": "error",
            "error": "Invalid starlisting_id",
            "details": "Starlisting 999 not found"
        }
    """

    type: Literal[MessageType.ERROR] = MessageType.ERROR
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")


# ============================================================================
# Union Types for Message Parsing
# ============================================================================


# All possible server → client messages
ServerMessage = Union[
    CandleUpdate,
    FundingRateUpdate,
    OpenInterestUpdate,
    Heartbeat,
    SubscriptionConfirmed,
    UnsubscriptionConfirmed,
    ErrorMessage,
]

# All possible client → server messages
ClientMessage = Union[
    SubscribeRequest,
    UnsubscribeRequest,
]
