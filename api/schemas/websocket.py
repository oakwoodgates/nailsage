"""WebSocket message schemas."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Channel types
Channel = Literal["trades", "positions", "portfolio", "prices", "signals"]

# Event types
TradeEventType = Literal["trade.new"]
PositionEventType = Literal["position.opened", "position.closed", "position.pnl_update"]
PortfolioEventType = Literal["portfolio.update"]
PriceEventType = Literal["price.candle", "price.ticker"]
SignalEventType = Literal["signal.generated", "signal.executed"]


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure."""

    type: str = Field(description="Message type (e.g., 'trade.new')")
    channel: Channel = Field(description="Channel this message belongs to")
    timestamp: int = Field(description="Message timestamp (Unix ms)")
    data: Dict[str, Any] = Field(description="Message payload")


class SubscribeRequest(BaseModel):
    """Client subscription request."""

    action: Literal["subscribe"] = Field(description="Action type")
    channels: List[Channel] = Field(description="Channels to subscribe to")


class UnsubscribeRequest(BaseModel):
    """Client unsubscription request."""

    action: Literal["unsubscribe"] = Field(description="Action type")
    channels: List[Channel] = Field(description="Channels to unsubscribe from")


class SubscribeResponse(BaseModel):
    """Server subscription confirmation."""

    action: str = Field(description="Action performed")
    channels: List[Channel] = Field(description="Channels affected")
    status: Literal["subscribed", "unsubscribed", "error"] = Field(description="Status")
    message: Optional[str] = Field(None, description="Optional message")


# Event payloads

class TradeEvent(BaseModel):
    """Trade event data."""

    type: TradeEventType = Field(description="Event type")
    channel: Literal["trades"] = Field(default="trades")
    timestamp: int = Field(description="Event timestamp (Unix ms)")

    # Trade data
    trade_id: int = Field(description="Trade ID")
    position_id: int = Field(description="Position ID")
    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    starlisting_id: int = Field(description="Starlisting ID")
    trade_type: str = Field(description="Trade type")
    side: str = Field(description="Position side (long/short)")
    size: float = Field(description="Trade size in USD")
    price: float = Field(description="Execution price")
    fees: float = Field(description="Transaction fees")


class PositionEvent(BaseModel):
    """Position event data."""

    type: PositionEventType = Field(description="Event type")
    channel: Literal["positions"] = Field(default="positions")
    timestamp: int = Field(description="Event timestamp (Unix ms)")

    # Position data
    position_id: int = Field(description="Position ID")
    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    starlisting_id: int = Field(description="Starlisting ID")
    side: str = Field(description="Position side")
    size: float = Field(description="Position size")
    entry_price: float = Field(description="Entry price")
    status: str = Field(description="Position status")

    # P&L (may be None for newly opened positions)
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized P&L")
    realized_pnl: Optional[float] = Field(None, description="Realized P&L")
    current_price: Optional[float] = Field(None, description="Current market price")

    # Exit info (only for closed positions)
    exit_price: Optional[float] = Field(None, description="Exit price")
    exit_reason: Optional[str] = Field(None, description="Exit reason")


class PortfolioEvent(BaseModel):
    """Portfolio update event data."""

    type: PortfolioEventType = Field(description="Event type")
    channel: Literal["portfolio"] = Field(default="portfolio")
    timestamp: int = Field(description="Event timestamp (Unix ms)")

    # Portfolio metrics
    total_equity_usd: float = Field(description="Total equity")
    total_pnl_usd: float = Field(description="Total P&L")
    total_pnl_pct: float = Field(description="Total P&L percentage")
    realized_pnl_usd: float = Field(description="Realized P&L")
    unrealized_pnl_usd: float = Field(description="Unrealized P&L")
    open_positions: int = Field(description="Number of open positions")
    active_strategies: int = Field(description="Number of active strategies")


class PriceEvent(BaseModel):
    """Price update event data."""

    type: PriceEventType = Field(description="Event type")
    channel: Literal["prices"] = Field(default="prices")
    timestamp: int = Field(description="Event timestamp (Unix ms)")

    # Price data
    starlisting_id: int = Field(description="Starlisting ID")
    coin: str = Field(description="Coin symbol")
    interval: str = Field(description="Candle interval")

    # OHLCV data
    open: float = Field(description="Open price")
    high: float = Field(description="High price")
    low: float = Field(description="Low price")
    close: float = Field(description="Close price")
    volume: float = Field(description="Volume")


class SignalEvent(BaseModel):
    """Signal event data."""

    type: SignalEventType = Field(description="Event type")
    channel: Literal["signals"] = Field(default="signals")
    timestamp: int = Field(description="Event timestamp (Unix ms)")

    # Signal data
    signal_id: int = Field(description="Signal ID")
    strategy_id: int = Field(description="Strategy ID")
    strategy_name: str = Field(description="Strategy name")
    starlisting_id: int = Field(description="Starlisting ID")
    signal_type: str = Field(description="Signal type (long/short/neutral/close)")
    confidence: Optional[float] = Field(None, description="Model confidence (0-1)")
    price_at_signal: float = Field(description="Price when signal was generated")
    was_executed: bool = Field(description="Whether signal was executed")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if not executed")
