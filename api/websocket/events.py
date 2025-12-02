"""Event dispatcher for real-time updates."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types for WebSocket broadcasting."""

    # Trade events
    TRADE_EXECUTED = "trade.new"

    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_PNL_UPDATE = "position.pnl_update"

    # Portfolio events
    PORTFOLIO_UPDATE = "portfolio.update"

    # Signal events
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_EXECUTED = "signal.executed"

    # Price events
    PRICE_UPDATE = "price.candle"


# Map event types to channels
EVENT_TO_CHANNEL: Dict[EventType, str] = {
    EventType.TRADE_EXECUTED: "trades",
    EventType.POSITION_OPENED: "positions",
    EventType.POSITION_CLOSED: "positions",
    EventType.POSITION_PNL_UPDATE: "positions",
    EventType.PORTFOLIO_UPDATE: "portfolio",
    EventType.SIGNAL_GENERATED: "signals",
    EventType.SIGNAL_EXECUTED: "signals",
    EventType.PRICE_UPDATE: "prices",
}


class EventDispatcher:
    """
    Central event bus for propagating trading events to WebSocket clients.

    This is a singleton that can be accessed from anywhere in the codebase
    to emit events that should be broadcast to connected WebSocket clients.

    Integration points:
    - TradeExecutionPipeline emits trade events
    - PositionTracker emits position events
    - Kirby client forwards price events
    """

    _instance: Optional["EventDispatcher"] = None

    def __new__(cls) -> "EventDispatcher":
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize event dispatcher."""
        if self._initialized:
            return

        self._connection_manager = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }

        self._initialized = True
        logger.info("EventDispatcher initialized")

    @classmethod
    def get_instance(cls) -> "EventDispatcher":
        """Get the singleton EventDispatcher instance.

        Returns:
            EventDispatcher instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_connection_manager(self, manager) -> None:
        """Set the WebSocket connection manager.

        Args:
            manager: ConnectionManager instance
        """
        self._connection_manager = manager
        logger.info("EventDispatcher connected to ConnectionManager")

    def register_callback(
        self,
        event_type: EventType,
        callback: Callable,
    ) -> None:
        """Register a callback for an event type.

        Args:
            event_type: Event type to listen for
            callback: Callback function (async or sync)
        """
        self._callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for {event_type}")

    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
    ) -> None:
        """Emit an event to be broadcast to WebSocket clients.

        This method is thread-safe and can be called from sync code
        by using asyncio.run_coroutine_threadsafe().

        Args:
            event_type: Type of event
            data: Event data payload
        """
        if self._connection_manager is None:
            logger.debug(f"No connection manager, skipping event: {event_type}")
            return

        channel = EVENT_TO_CHANNEL.get(event_type)
        if not channel:
            logger.warning(f"Unknown event type: {event_type}")
            return

        message = {
            "type": event_type.value,
            "channel": channel,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": data,
        }

        # Broadcast to channel
        sent_count = await self._connection_manager.broadcast_to_channel(
            channel, message
        )

        logger.debug(
            f"Emitted {event_type} to {sent_count} clients on channel '{channel}'"
        )

        # Execute registered callbacks
        for callback in self._callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")

    def emit_sync(
        self,
        event_type: EventType,
        data: Dict[str, Any],
    ) -> None:
        """Emit an event synchronously (for use from non-async code).

        This queues the event to be processed by the async event loop.

        Args:
            event_type: Type of event
            data: Event data payload
        """
        try:
            self._event_queue.put_nowait((event_type, data))
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")

    async def start(self) -> None:
        """Start the event processing task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("EventDispatcher started")

    async def stop(self) -> None:
        """Stop the event processing task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EventDispatcher stopped")

    async def _process_events(self) -> None:
        """Process events from the sync queue."""
        while self._running:
            try:
                event_type, data = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                await self.emit(event_type, data)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")


# Convenience functions for emitting events

async def emit_trade_event(
    trade_id: int,
    position_id: int,
    strategy_id: int,
    strategy_name: str,
    starlisting_id: int,
    trade_type: str,
    side: str,
    size: float,
    price: float,
    fees: float,
) -> None:
    """Emit a trade executed event.

    Args:
        trade_id: Trade ID
        position_id: Position ID
        strategy_id: Strategy ID
        strategy_name: Strategy name
        starlisting_id: Starlisting ID
        trade_type: Trade type
        side: Position side (long/short)
        size: Trade size
        price: Execution price
        fees: Transaction fees
    """
    dispatcher = EventDispatcher.get_instance()
    await dispatcher.emit(
        EventType.TRADE_EXECUTED,
        {
            "trade_id": trade_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "starlisting_id": starlisting_id,
            "trade_type": trade_type,
            "side": side,
            "size": size,
            "price": price,
            "fees": fees,
        },
    )


async def emit_position_event(
    event_type: EventType,
    position_id: int,
    strategy_id: int,
    strategy_name: str,
    starlisting_id: int,
    side: str,
    size: float,
    entry_price: float,
    status: str,
    unrealized_pnl: Optional[float] = None,
    realized_pnl: Optional[float] = None,
    current_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    exit_reason: Optional[str] = None,
) -> None:
    """Emit a position event.

    Args:
        event_type: Position event type
        position_id: Position ID
        strategy_id: Strategy ID
        strategy_name: Strategy name
        starlisting_id: Starlisting ID
        side: Position side
        size: Position size
        entry_price: Entry price
        status: Position status
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L
        current_price: Current market price
        exit_price: Exit price (for closed positions)
        exit_reason: Exit reason (for closed positions)
    """
    dispatcher = EventDispatcher.get_instance()
    await dispatcher.emit(
        event_type,
        {
            "position_id": position_id,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "starlisting_id": starlisting_id,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "status": status,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "current_price": current_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
        },
    )


async def emit_portfolio_event(
    total_equity_usd: float,
    total_pnl_usd: float,
    total_pnl_pct: float,
    realized_pnl_usd: float,
    unrealized_pnl_usd: float,
    open_positions: int,
    active_strategies: int,
) -> None:
    """Emit a portfolio update event.

    Args:
        total_equity_usd: Total equity
        total_pnl_usd: Total P&L
        total_pnl_pct: Total P&L percentage
        realized_pnl_usd: Realized P&L
        unrealized_pnl_usd: Unrealized P&L
        open_positions: Number of open positions
        active_strategies: Number of active strategies
    """
    dispatcher = EventDispatcher.get_instance()
    await dispatcher.emit(
        EventType.PORTFOLIO_UPDATE,
        {
            "total_equity_usd": total_equity_usd,
            "total_pnl_usd": total_pnl_usd,
            "total_pnl_pct": total_pnl_pct,
            "realized_pnl_usd": realized_pnl_usd,
            "unrealized_pnl_usd": unrealized_pnl_usd,
            "open_positions": open_positions,
            "active_strategies": active_strategies,
        },
    )


async def emit_price_event(
    starlisting_id: int,
    coin: str,
    interval: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> None:
    """Emit a price update event.

    Args:
        starlisting_id: Starlisting ID
        coin: Coin symbol
        interval: Candle interval
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume
    """
    dispatcher = EventDispatcher.get_instance()
    await dispatcher.emit(
        EventType.PRICE_UPDATE,
        {
            "starlisting_id": starlisting_id,
            "coin": coin,
            "interval": interval,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
    )


async def emit_signal_event(
    signal_id: int,
    strategy_id: int,
    strategy_name: str,
    starlisting_id: int,
    signal_type: str,
    confidence: Optional[float],
    price_at_signal: float,
    was_executed: bool,
    rejection_reason: Optional[str] = None,
) -> None:
    """Emit a signal generated event.

    Args:
        signal_id: Signal ID
        strategy_id: Strategy ID
        strategy_name: Strategy name
        starlisting_id: Starlisting ID
        signal_type: Signal type ('long', 'short', 'neutral', 'close')
        confidence: Model confidence (0-1)
        price_at_signal: Price when signal was generated
        was_executed: Whether signal was acted upon
        rejection_reason: Reason for rejection if not executed
    """
    dispatcher = EventDispatcher.get_instance()
    await dispatcher.emit(
        EventType.SIGNAL_GENERATED,
        {
            "signal_id": signal_id,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "starlisting_id": starlisting_id,
            "signal_type": signal_type,
            "confidence": confidence,
            "price_at_signal": price_at_signal,
            "was_executed": was_executed,
            "rejection_reason": rejection_reason,
        },
    )
