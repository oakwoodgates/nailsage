"""WebSocket client for Kirby API.

This module provides an async WebSocket client that:
- Connects to Kirby API with authentication
- Subscribes to real-time candle updates
- Handles reconnection with exponential backoff
- Monitors heartbeats and triggers reconnection on timeout
- Emits callbacks for different message types

Example usage:
    async def on_candle(candle: Candle):
        print(f"Received candle: {candle}")

    client = KirbyWebSocketClient(config)
    client.on_candle_update(on_candle)

    await client.connect()
    await client.subscribe(starlisting_id=1, historical_candles=200)

    # Run forever
    await client.wait_until_closed()
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from pydantic import ValidationError
from websockets.client import WebSocketClientProtocol

from config.paper_trading import WebSocketConfig
from execution.websocket.models import (
    Candle,
    CandleUpdate,
    ClientMessage,
    ErrorMessage,
    FundingRate,
    FundingRateUpdate,
    Heartbeat,
    MessageType,
    OpenInterest,
    OpenInterestUpdate,
    ServerMessage,
    SubscribeRequest,
    SubscriptionConfirmed,
    UnsubscribeRequest,
    UnsubscriptionConfirmed,
)

logger = logging.getLogger(__name__)


class KirbyWebSocketClient:
    """
    Async WebSocket client for Kirby API.

    This client handles:
    - Connection management with authentication
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring
    - Message parsing and validation
    - Event callbacks for different message types

    Attributes:
        config: WebSocket configuration
        _ws: Active WebSocket connection (None if disconnected)
        _running: Whether client is running
        _reconnect_task: Background task for reconnection
        _heartbeat_task: Background task for heartbeat monitoring
        _subscribed_starlistings: Set of currently subscribed starlisting IDs
    """

    def __init__(self, config: WebSocketConfig):
        """
        Initialize WebSocket client.

        Args:
            config: WebSocket configuration
        """
        self.config = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running: bool = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._subscribed_starlistings: Set[int] = set()
        self._last_heartbeat: Optional[datetime] = None
        self._reconnect_attempts: int = 0

        # Callbacks
        self._candle_callbacks: List[Callable[[Candle], Any]] = []
        self._funding_rate_callbacks: List[Callable[[FundingRate], Any]] = []
        self._open_interest_callbacks: List[Callable[[OpenInterest], Any]] = []
        self._error_callbacks: List[Callable[[str, Optional[str]], Any]] = []

    # ========================================================================
    # Connection Management
    # ========================================================================

    async def connect(self) -> None:
        """
        Connect to Kirby WebSocket API.

        This establishes the WebSocket connection with authentication
        and starts background tasks for heartbeat monitoring and message receiving.

        Raises:
            ConnectionError: If connection fails
        """
        if self._running:
            logger.warning("Client is already running")
            return

        self._running = True
        await self._establish_connection()

    async def _establish_connection(self) -> None:
        """
        Establish WebSocket connection with authentication.

        Internal method that handles the actual connection logic.
        """
        try:
            # Build WebSocket URL with authentication
            url = f"{self.config.url}?api_key={self.config.api_key}"

            logger.info(f"Connecting to Kirby WebSocket: {self.config.url}")
            self._ws = await websockets.connect(url)
            logger.info("Successfully connected to Kirby WebSocket")

            # Reset reconnection state
            self._reconnect_attempts = 0
            self._last_heartbeat = datetime.now()

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._monitor_heartbeat())
            self._receive_task = asyncio.create_task(self._receive_messages())

            # Resubscribe to previously subscribed starlistings
            if self._subscribed_starlistings:
                logger.info(f"Resubscribing to {len(self._subscribed_starlistings)} starlistings")
                for starlisting_id in list(self._subscribed_starlistings):
                    await self.subscribe(starlisting_id, historical_candles=0)

        except Exception as e:
            logger.error(f"Failed to connect to Kirby WebSocket: {e}")
            self._ws = None
            if self.config.reconnect_enabled:
                await self._schedule_reconnect()
            else:
                raise ConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """
        Disconnect from Kirby WebSocket API.

        This gracefully closes the connection and stops all background tasks.
        """
        logger.info("Disconnecting from Kirby WebSocket")
        self._running = False

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("Disconnected from Kirby WebSocket")

    async def _schedule_reconnect(self) -> None:
        """
        Schedule reconnection with exponential backoff.

        This implements the industry-standard exponential backoff algorithm
        for reconnection attempts.
        """
        if not self.config.reconnect_enabled:
            return

        # Check if we've exceeded max attempts (0 = infinite)
        if (
            self.config.reconnect_max_attempts > 0
            and self._reconnect_attempts >= self.config.reconnect_max_attempts
        ):
            logger.error(
                f"Max reconnection attempts ({self.config.reconnect_max_attempts}) exceeded"
            )
            self._running = False
            return

        # Calculate delay with exponential backoff
        delay = min(
            self.config.reconnect_initial_delay
            * (self.config.reconnect_backoff_multiplier ** self._reconnect_attempts),
            self.config.reconnect_max_delay,
        )

        self._reconnect_attempts += 1
        logger.info(
            f"Scheduling reconnection attempt {self._reconnect_attempts} "
            f"in {delay:.1f} seconds"
        )

        await asyncio.sleep(delay)
        await self._establish_connection()

    # ========================================================================
    # Subscription Management
    # ========================================================================

    async def subscribe(
        self, starlisting_id: int, historical_candles: Optional[int] = None
    ) -> None:
        """
        Subscribe to a starlisting for real-time updates.

        Args:
            starlisting_id: Kirby starlisting ID to subscribe to
            historical_candles: Number of historical candles to request (max 1000)

        Raises:
            ConnectionError: If not connected to WebSocket
        """
        if not self._ws:
            raise ConnectionError("Not connected to WebSocket")

        # Use configured default if not specified
        if historical_candles is None:
            historical_candles = self.config.historical_candles

        request = SubscribeRequest(
            starlisting_id=starlisting_id, historical_candles=historical_candles
        )

        logger.info(
            f"Subscribing to starlisting {starlisting_id} "
            f"(requesting {historical_candles} historical candles)"
        )

        await self._send_message(request)
        self._subscribed_starlistings.add(starlisting_id)

    async def unsubscribe(self, starlisting_id: int) -> None:
        """
        Unsubscribe from a starlisting.

        Args:
            starlisting_id: Kirby starlisting ID to unsubscribe from

        Raises:
            ConnectionError: If not connected to WebSocket
        """
        if not self._ws:
            raise ConnectionError("Not connected to WebSocket")

        request = UnsubscribeRequest(starlisting_id=starlisting_id)

        logger.info(f"Unsubscribing from starlisting {starlisting_id}")

        await self._send_message(request)
        self._subscribed_starlistings.discard(starlisting_id)

    # ========================================================================
    # Message Handling
    # ========================================================================

    async def _send_message(self, message: ClientMessage) -> None:
        """
        Send a message to the WebSocket server.

        Args:
            message: Client message to send

        Raises:
            ConnectionError: If not connected to WebSocket
        """
        if not self._ws:
            raise ConnectionError("Not connected to WebSocket")

        payload = message.model_dump(mode="json")
        await self._ws.send(json.dumps(payload))

    async def _receive_messages(self) -> None:
        """
        Background task to receive and process WebSocket messages.

        This runs continuously while connected, parsing incoming messages
        and dispatching them to appropriate callbacks.
        """
        try:
            while self._running and self._ws:
                try:
                    raw_message = await self._ws.recv()
                    await self._handle_message(raw_message)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    if self._running and self.config.reconnect_enabled:
                        await self._schedule_reconnect()
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}", exc_info=True)

        except asyncio.CancelledError:
            logger.debug("Receive task cancelled")
            raise

    async def _handle_message(self, raw_message: str) -> None:
        """
        Parse and handle incoming WebSocket message.

        Args:
            raw_message: Raw JSON message from server
        """
        try:
            # Parse JSON
            data = json.loads(raw_message)
            message_type = data.get("type")

            # Dispatch to appropriate handler
            if message_type == MessageType.CANDLE_UPDATE:
                msg = CandleUpdate(**data)
                await self._handle_candle_update(msg)

            elif message_type == MessageType.FUNDING_RATE_UPDATE:
                msg = FundingRateUpdate(**data)
                await self._handle_funding_rate_update(msg)

            elif message_type == MessageType.OPEN_INTEREST_UPDATE:
                msg = OpenInterestUpdate(**data)
                await self._handle_open_interest_update(msg)

            elif message_type == MessageType.HEARTBEAT:
                msg = Heartbeat(**data)
                await self._handle_heartbeat(msg)

            elif message_type == MessageType.SUBSCRIPTION_CONFIRMED:
                msg = SubscriptionConfirmed(**data)
                logger.info(f"Subscription confirmed: {msg.message}")

            elif message_type == MessageType.UNSUBSCRIPTION_CONFIRMED:
                msg = UnsubscriptionConfirmed(**data)
                logger.info(f"Unsubscription confirmed: {msg.message}")

            elif message_type == MessageType.ERROR:
                msg = ErrorMessage(**data)
                await self._handle_error(msg)

            else:
                logger.warning(f"Unknown message type: {message_type}")

        except ValidationError as e:
            logger.error(f"Failed to parse message: {e}")
            logger.debug(f"Raw message: {raw_message}")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def _handle_candle_update(self, msg: CandleUpdate) -> None:
        """Handle candle update message."""
        for callback in self._candle_callbacks:
            try:
                result = callback(msg.candle)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in candle callback: {e}", exc_info=True)

    async def _handle_funding_rate_update(self, msg: FundingRateUpdate) -> None:
        """Handle funding rate update message."""
        for callback in self._funding_rate_callbacks:
            try:
                result = callback(msg.funding_rate)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in funding rate callback: {e}", exc_info=True)

    async def _handle_open_interest_update(self, msg: OpenInterestUpdate) -> None:
        """Handle open interest update message."""
        for callback in self._open_interest_callbacks:
            try:
                result = callback(msg.open_interest)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in open interest callback: {e}", exc_info=True)

    async def _handle_heartbeat(self, msg: Heartbeat) -> None:
        """Handle heartbeat message."""
        self._last_heartbeat = datetime.now()
        logger.debug(f"Received heartbeat at {self._last_heartbeat}")

    async def _handle_error(self, msg: ErrorMessage) -> None:
        """Handle error message."""
        logger.error(f"Server error: {msg.error}")
        if msg.details:
            logger.error(f"Error details: {msg.details}")

        for callback in self._error_callbacks:
            try:
                result = callback(msg.error, msg.details)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in error callback: {e}", exc_info=True)

    # ========================================================================
    # Heartbeat Monitoring
    # ========================================================================

    async def _monitor_heartbeat(self) -> None:
        """
        Background task to monitor heartbeat and trigger reconnection on timeout.

        This runs continuously while connected, checking if we've received
        a heartbeat within the expected interval.
        """
        try:
            while self._running and self._ws:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Check if heartbeat timeout exceeded
                if self._last_heartbeat:
                    time_since_heartbeat = datetime.now() - self._last_heartbeat
                    if time_since_heartbeat.total_seconds() > self.config.heartbeat_timeout:
                        logger.warning(
                            f"Heartbeat timeout ({time_since_heartbeat.total_seconds():.1f}s > "
                            f"{self.config.heartbeat_timeout}s) - triggering reconnection"
                        )
                        if self._ws:
                            await self._ws.close()
                        if self.config.reconnect_enabled:
                            await self._schedule_reconnect()
                        break

        except asyncio.CancelledError:
            logger.debug("Heartbeat monitor task cancelled")
            raise

    # ========================================================================
    # Callback Registration
    # ========================================================================

    def on_candle_update(self, callback: Callable[[Candle], Any]) -> None:
        """
        Register callback for candle updates.

        Args:
            callback: Function to call on each candle update.
                     Can be sync or async.
        """
        self._candle_callbacks.append(callback)

    def on_funding_rate_update(self, callback: Callable[[FundingRate], Any]) -> None:
        """
        Register callback for funding rate updates.

        Args:
            callback: Function to call on each funding rate update.
                     Can be sync or async.
        """
        self._funding_rate_callbacks.append(callback)

    def on_open_interest_update(self, callback: Callable[[OpenInterest], Any]) -> None:
        """
        Register callback for open interest updates.

        Args:
            callback: Function to call on each open interest update.
                     Can be sync or async.
        """
        self._open_interest_callbacks.append(callback)

    def on_error(self, callback: Callable[[str, Optional[str]], Any]) -> None:
        """
        Register callback for error messages.

        Args:
            callback: Function to call on each error message.
                     Receives (error, details) as arguments.
                     Can be sync or async.
        """
        self._error_callbacks.append(callback)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to WebSocket."""
        return self._ws is not None and self._ws.open

    async def wait_until_closed(self) -> None:
        """Wait until the client is closed (useful for keeping the event loop alive)."""
        while self._running:
            await asyncio.sleep(1)
