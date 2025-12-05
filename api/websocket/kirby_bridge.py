"""Bridge for proxying Kirby price data to API WebSocket clients.

This module provides a bridge that connects to Kirby's WebSocket API
and forwards price data (candles, funding rates, etc.) to the API's
WebSocket clients subscribed to the 'prices' channel.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import websockets
from websockets.client import WebSocketClientProtocol

from api.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)


class KirbyBridge:
    """
    Bridge for proxying Kirby price data to API WebSocket clients.

    This class:
    - Maintains a single WebSocket connection to Kirby
    - Aggregates subscriptions from multiple API clients
    - Forwards price data to the 'prices' channel
    - Handles reconnection with exponential backoff
    """

    def __init__(
        self,
        kirby_url: str,
        kirby_api_key: str,
        connection_manager: ConnectionManager,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
    ):
        """Initialize KirbyBridge.

        Args:
            kirby_url: Kirby WebSocket URL (wss://...)
            kirby_api_key: Kirby API key (kb_...)
            connection_manager: ConnectionManager for broadcasting
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
        """
        self._kirby_url = kirby_url
        self._kirby_api_key = kirby_api_key
        self._manager = connection_manager
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._connect_task: Optional[asyncio.Task] = None

        # Track subscriptions: starlisting_id -> set of connection_ids
        self._subscriptions: Dict[int, Set[str]] = {}
        # Track history requests: starlisting_id -> history count
        self._pending_history: Dict[int, int] = {}
        # Queue for subscriptions that came in before WebSocket connected
        self._pending_subscriptions: Dict[int, int] = {}  # starlisting_id -> history

        logger.info(f"KirbyBridge initialized for {kirby_url}")

    async def start(self) -> None:
        """Start the Kirby bridge."""
        if self._running:
            return

        self._running = True
        # Run connection in background to avoid blocking startup
        self._connect_task = asyncio.create_task(self._connect())
        logger.info("KirbyBridge started")

    async def stop(self) -> None:
        """Stop the Kirby bridge."""
        self._running = False

        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("KirbyBridge stopped")

    async def subscribe(
        self,
        connection_id: str,
        starlisting_ids: List[int],
        history: int = 500,
    ) -> None:
        """Subscribe a client to price updates for starlistings.

        Args:
            connection_id: Client connection ID
            starlisting_ids: List of starlisting IDs to subscribe to
            history: Number of historical candles to request
        """
        logger.info(f"Client {connection_id} subscribing to starlistings: {starlisting_ids} (history={history})")

        for starlisting_id in starlisting_ids:
            is_new_subscription = starlisting_id not in self._subscriptions

            if is_new_subscription:
                self._subscriptions[starlisting_id] = set()
                self._pending_history[starlisting_id] = history
                # Need to subscribe to Kirby
                if self._ws:
                    await self._subscribe_to_kirby(starlisting_id, history)
                else:
                    # Queue subscription for when WebSocket connects
                    self._pending_subscriptions[starlisting_id] = history
                    logger.warning(
                        f"WebSocket not connected - queued subscription for starlisting {starlisting_id}"
                    )
            elif history > 0 and self._ws:
                # Already subscribed to Kirby, but new client needs historical data
                # Re-subscribe to trigger historical data (will be sent to all clients)
                logger.info(
                    f"Re-requesting historical data for starlisting {starlisting_id} "
                    f"(existing subscription, new client {connection_id})"
                )
                await self._subscribe_to_kirby(starlisting_id, history)

            self._subscriptions[starlisting_id].add(connection_id)
            logger.info(
                f"Client {connection_id} subscribed to starlisting {starlisting_id}"
            )

    async def unsubscribe(
        self,
        connection_id: str,
        starlisting_ids: Optional[List[int]] = None,
    ) -> None:
        """Unsubscribe a client from price updates.

        Args:
            connection_id: Client connection ID
            starlisting_ids: List of starlisting IDs (None = all)
        """
        if starlisting_ids is None:
            # Unsubscribe from all
            starlisting_ids = list(self._subscriptions.keys())

        for starlisting_id in starlisting_ids:
            if starlisting_id in self._subscriptions:
                self._subscriptions[starlisting_id].discard(connection_id)

                # If no more clients, unsubscribe from Kirby
                if not self._subscriptions[starlisting_id]:
                    await self._unsubscribe_from_kirby(starlisting_id)
                    del self._subscriptions[starlisting_id]
                    logger.debug(f"No more clients for starlisting {starlisting_id}")

    async def _connect(self) -> None:
        """Connect to Kirby WebSocket."""
        delay = self._reconnect_delay

        while self._running:
            try:
                # Add API key as query param
                url = f"{self._kirby_url}?api_key={self._kirby_api_key}"

                logger.info(f"Connecting to Kirby...")
                self._ws = await websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=10,
                )
                logger.info("Connected to Kirby")

                # Reset delay on successful connection
                delay = self._reconnect_delay

                # Send any pending subscriptions that were queued before connection
                if self._pending_subscriptions:
                    logger.info(f"Sending {len(self._pending_subscriptions)} pending subscriptions")
                    for starlisting_id, history in self._pending_subscriptions.items():
                        await self._subscribe_to_kirby(starlisting_id, history)
                    self._pending_subscriptions.clear()

                # Re-subscribe to all active subscriptions (for reconnection case)
                for starlisting_id, clients in self._subscriptions.items():
                    if clients and starlisting_id not in self._pending_subscriptions:
                        history = self._pending_history.get(starlisting_id, 0)
                        await self._subscribe_to_kirby(starlisting_id, history)

                # Start receiving messages
                self._receive_task = asyncio.create_task(self._receive_loop())
                await self._receive_task

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Kirby connection closed: {e}")
            except Exception as e:
                logger.error(f"Error connecting to Kirby: {e}")

            if self._running:
                logger.info(f"Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _receive_loop(self) -> None:
        """Receive and process messages from Kirby."""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Kirby connection closed during receive")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving Kirby message: {e}")

    async def _handle_message(self, raw_message: str) -> None:
        """Handle a message from Kirby.

        Args:
            raw_message: Raw JSON message from Kirby
        """
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")

            # Log all message types for debugging
            if msg_type not in ("heartbeat", "ping", "pong"):
                starlisting_id = message.get("starlisting_id", "N/A")
                logger.info(f"Kirby message: type={msg_type}, starlisting_id={starlisting_id}")

            if msg_type == "candle":
                await self._handle_candle(message)
            elif msg_type == "historical":
                await self._handle_historical(message)
            elif msg_type == "historical_funding":
                await self._handle_historical_funding(message)
            elif msg_type == "historical_oi":
                await self._handle_historical_oi(message)
            elif msg_type == "funding":
                await self._handle_funding(message)
            elif msg_type == "open_interest":
                await self._handle_open_interest(message)
            elif msg_type == "success":
                logger.info(f"Kirby success: {message.get('message')}")
            elif msg_type in ("heartbeat", "ping", "pong"):
                # Ignore heartbeats
                pass
            elif msg_type == "error":
                logger.error(f"Kirby error: {message.get('message')}")
            else:
                logger.warning(f"Unknown Kirby message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from Kirby: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"Error handling Kirby message: {e}", exc_info=True)

    async def _handle_candle(self, message: dict) -> None:
        """Handle a candle update from Kirby."""
        starlisting_id = message.get("starlisting_id")
        if not starlisting_id or starlisting_id not in self._subscriptions:
            return

        # Format for API clients
        candle_data = message.get("data", {})
        ws_message = {
            "type": "price.candle",
            "channel": "prices",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": {
                "starlisting_id": starlisting_id,
                "coin": message.get("coin", ""),
                "interval": message.get("interval", ""),
                "time": candle_data.get("time"),
                "open": float(candle_data.get("open", 0)),
                "high": float(candle_data.get("high", 0)),
                "low": float(candle_data.get("low", 0)),
                "close": float(candle_data.get("close", 0)),
                "volume": float(candle_data.get("volume", 0)),
            },
        }

        await self._manager.broadcast_to_channel("prices", ws_message)

    async def _handle_historical(self, message: dict) -> None:
        """Handle historical candles from Kirby."""
        starlisting_id = message.get("starlisting_id")
        logger.info(f"Processing historical candles for starlisting {starlisting_id}")

        if not starlisting_id:
            logger.warning("Historical message missing starlisting_id")
            return

        if starlisting_id not in self._subscriptions:
            logger.warning(
                f"Historical data for starlisting {starlisting_id} but not in subscriptions: {list(self._subscriptions.keys())}"
            )
            return

        candles = message.get("data", [])
        logger.info(f"Received {len(candles)} historical candles from Kirby for starlisting {starlisting_id}")

        # Format for API clients
        ws_message = {
            "type": "price.historical",
            "channel": "prices",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": {
                "starlisting_id": starlisting_id,
                "coin": message.get("coin", ""),
                "interval": message.get("interval", ""),
                "count": len(candles),
                "candles": [
                    {
                        "time": c.get("time"),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("high", 0)),
                        "low": float(c.get("low", 0)),
                        "close": float(c.get("close", 0)),
                        "volume": float(c.get("volume", 0)),
                    }
                    for c in candles
                ],
            },
        }

        sent_count = await self._manager.broadcast_to_channel("prices", ws_message)
        logger.info(
            f"Forwarded {len(candles)} historical candles for starlisting {starlisting_id} to {sent_count} clients"
        )

    async def _handle_funding(self, message: dict) -> None:
        """Handle funding rate update from Kirby."""
        starlisting_id = message.get("starlisting_id")
        if not starlisting_id or starlisting_id not in self._subscriptions:
            return

        funding_data = message.get("data", {})
        ws_message = {
            "type": "price.funding",
            "channel": "prices",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": {
                "starlisting_id": starlisting_id,
                "funding_rate": float(funding_data.get("funding_rate", 0)),
                "time": funding_data.get("time"),
            },
        }

        await self._manager.broadcast_to_channel("prices", ws_message)

    async def _handle_open_interest(self, message: dict) -> None:
        """Handle open interest update from Kirby."""
        starlisting_id = message.get("starlisting_id")
        if not starlisting_id or starlisting_id not in self._subscriptions:
            return

        oi_data = message.get("data", {})
        ws_message = {
            "type": "price.open_interest",
            "channel": "prices",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": {
                "starlisting_id": starlisting_id,
                "open_interest": float(oi_data.get("open_interest", 0)),
                "time": oi_data.get("time"),
            },
        }

        await self._manager.broadcast_to_channel("prices", ws_message)

    async def _handle_historical_funding(self, message: dict) -> None:
        """Handle historical funding rate data from Kirby."""
        starlisting_id = message.get("starlisting_id")
        if not starlisting_id or starlisting_id not in self._subscriptions:
            return

        funding_data = message.get("data", [])

        ws_message = {
            "type": "price.historical_funding",
            "channel": "prices",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": {
                "starlisting_id": starlisting_id,
                "coin": message.get("coin", ""),
                "count": len(funding_data),
                "funding_rates": [
                    {
                        "time": f.get("time"),
                        "funding_rate": float(f.get("funding_rate", 0)),
                        "mark_price": float(f.get("mark_price", 0)) if f.get("mark_price") else None,
                    }
                    for f in funding_data
                ],
            },
        }

        await self._manager.broadcast_to_channel("prices", ws_message)
        logger.info(
            f"Forwarded {len(funding_data)} historical funding rates for starlisting {starlisting_id}"
        )

    async def _handle_historical_oi(self, message: dict) -> None:
        """Handle historical open interest data from Kirby."""
        starlisting_id = message.get("starlisting_id")
        if not starlisting_id or starlisting_id not in self._subscriptions:
            return

        oi_data = message.get("data", [])

        ws_message = {
            "type": "price.historical_oi",
            "channel": "prices",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "data": {
                "starlisting_id": starlisting_id,
                "coin": message.get("coin", ""),
                "count": len(oi_data),
                "open_interest": [
                    {
                        "time": o.get("time"),
                        "open_interest": float(o.get("open_interest", 0)),
                        "notional_value": float(o.get("notional_value", 0)) if o.get("notional_value") else None,
                    }
                    for o in oi_data
                ],
            },
        }

        await self._manager.broadcast_to_channel("prices", ws_message)
        logger.info(
            f"Forwarded {len(oi_data)} historical OI data points for starlisting {starlisting_id}"
        )

    async def _subscribe_to_kirby(self, starlisting_id: int, history: int = 0) -> None:
        """Subscribe to a starlisting on Kirby.

        Args:
            starlisting_id: Starlisting ID to subscribe to
            history: Number of historical candles to request
        """
        if not self._ws:
            return

        subscribe_msg = {
            "action": "subscribe",
            "starlisting_ids": [starlisting_id],
        }
        if history > 0:
            subscribe_msg["history"] = history

        await self._ws.send(json.dumps(subscribe_msg))
        logger.info(
            f"Subscribed to Kirby starlisting {starlisting_id} (history={history})"
        )

    async def _unsubscribe_from_kirby(self, starlisting_id: int) -> None:
        """Unsubscribe from a starlisting on Kirby.

        Args:
            starlisting_id: Starlisting ID to unsubscribe from
        """
        if not self._ws:
            return

        unsubscribe_msg = {
            "action": "unsubscribe",
            "starlisting_ids": [starlisting_id],
        }

        await self._ws.send(json.dumps(unsubscribe_msg))
        logger.info(f"Unsubscribed from Kirby starlisting {starlisting_id}")


# Global KirbyBridge instance
_bridge: Optional[KirbyBridge] = None


def get_kirby_bridge() -> Optional[KirbyBridge]:
    """Get the global KirbyBridge instance."""
    return _bridge


def set_kirby_bridge(bridge: KirbyBridge) -> None:
    """Set the global KirbyBridge instance."""
    global _bridge
    _bridge = bridge
