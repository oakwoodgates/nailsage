"""WebSocket connection manager with subscription support."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from api.schemas.websocket import (
    Channel,
    SubscribeRequest,
    UnsubscribeRequest,
    SubscribeResponse,
    WebSocketMessage,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections with subscription-based messaging.

    Features:
    - Channel-based subscriptions (trades, positions, prices, portfolio, signals)
    - Per-connection subscription tracking
    - Automatic cleanup on disconnect
    - Heartbeat support
    """

    def __init__(self, max_connections: int = 100):
        """Initialize connection manager.

        Args:
            max_connections: Maximum allowed concurrent connections
        """
        self.max_connections = max_connections

        # Connection tracking: connection_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

        # Subscription tracking: channel -> set of connection_ids
        self.subscriptions: Dict[str, Set[str]] = {
            "trades": set(),
            "positions": set(),
            "portfolio": set(),
            "prices": set(),
            "signals": set(),
        }

        # Reverse mapping: connection_id -> set of channels
        self.connection_channels: Dict[str, Set[str]] = {}

        # Filter tracking: connection_id -> {channel -> filters}
        # e.g., {"conn123": {"positions": {"strategy_id": 1}}}
        self.connection_filters: Dict[str, Dict[str, Dict]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> Optional[str]:
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept

        Returns:
            Connection ID if successful, None if at capacity
        """
        async with self._lock:
            if len(self.active_connections) >= self.max_connections:
                logger.warning(f"Connection rejected: at capacity ({self.max_connections})")
                await websocket.close(code=1013, reason="Server at capacity")
                return None

            await websocket.accept()

            connection_id = str(uuid4())
            self.active_connections[connection_id] = websocket
            self.connection_channels[connection_id] = set()
            self.connection_filters[connection_id] = {}

            logger.info(
                f"WebSocket connected: {connection_id} "
                f"(total: {len(self.active_connections)})"
            )

            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "connection_id": connection_id,
                "message": "Connected to Nailsage API",
                "available_channels": list(self.subscriptions.keys()),
            })

            return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Handle WebSocket disconnection.

        Args:
            connection_id: ID of the connection to remove
        """
        was_subscribed_to_prices = False

        async with self._lock:
            if connection_id not in self.active_connections:
                return

            # Remove from all subscriptions
            channels = self.connection_channels.get(connection_id, set())
            was_subscribed_to_prices = "prices" in channels

            for channel in channels:
                self.subscriptions[channel].discard(connection_id)

            # Clean up connection tracking
            del self.active_connections[connection_id]
            if connection_id in self.connection_channels:
                del self.connection_channels[connection_id]
            if connection_id in self.connection_filters:
                del self.connection_filters[connection_id]

            logger.info(
                f"WebSocket disconnected: {connection_id} "
                f"(total: {len(self.active_connections)})"
            )

        # Clean up KirbyBridge subscription (outside lock to avoid deadlock)
        if was_subscribed_to_prices:
            from api.websocket.kirby_bridge import get_kirby_bridge
            bridge = get_kirby_bridge()
            if bridge:
                await bridge.unsubscribe(connection_id)
                logger.debug(f"Cleaned up KirbyBridge subscription for {connection_id}")

    async def subscribe(
        self,
        connection_id: str,
        channels: List[str],
        filters: Optional[Dict] = None,
    ) -> SubscribeResponse:
        """Subscribe a connection to channels.

        Args:
            connection_id: Connection ID
            channels: List of channels to subscribe to
            filters: Optional filters (e.g., {"strategy_id": 1})

        Returns:
            Subscription response
        """
        async with self._lock:
            if connection_id not in self.active_connections:
                return SubscribeResponse(
                    action="subscribe",
                    channels=[],
                    status="error",
                    message="Connection not found",
                )

            subscribed_channels = []
            for channel in channels:
                if channel in self.subscriptions:
                    self.subscriptions[channel].add(connection_id)
                    self.connection_channels[connection_id].add(channel)
                    subscribed_channels.append(channel)

                    # Store filters for this channel
                    if filters:
                        self.connection_filters[connection_id][channel] = filters
                else:
                    logger.warning(f"Unknown channel: {channel}")

            logger.debug(
                f"Connection {connection_id} subscribed to: {subscribed_channels}"
                f"{' with filters: ' + str(filters) if filters else ''}"
            )

            return SubscribeResponse(
                action="subscribe",
                channels=subscribed_channels,
                status="subscribed",
                message=f"Subscribed to {len(subscribed_channels)} channel(s)",
            )

    async def unsubscribe(
        self,
        connection_id: str,
        channels: List[str],
    ) -> SubscribeResponse:
        """Unsubscribe a connection from channels.

        Args:
            connection_id: Connection ID
            channels: List of channels to unsubscribe from

        Returns:
            Unsubscription response
        """
        async with self._lock:
            if connection_id not in self.active_connections:
                return SubscribeResponse(
                    action="unsubscribe",
                    channels=[],
                    status="error",
                    message="Connection not found",
                )

            unsubscribed_channels = []
            for channel in channels:
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(connection_id)
                    self.connection_channels[connection_id].discard(channel)
                    unsubscribed_channels.append(channel)

            logger.debug(
                f"Connection {connection_id} unsubscribed from: {unsubscribed_channels}"
            )

            return SubscribeResponse(
                action="unsubscribe",
                channels=unsubscribed_channels,
                status="unsubscribed",
                message=f"Unsubscribed from {len(unsubscribed_channels)} channel(s)",
            )

    async def broadcast_to_channel(
        self,
        channel: str,
        message: dict,
    ) -> int:
        """Broadcast a message to all subscribers of a channel.

        Applies filters if the connection has any set for this channel.
        Filters are applied by checking message["data"]["strategy_id"] against
        the filter's strategy_id value.

        Args:
            channel: Channel to broadcast to
            message: Message to send

        Returns:
            Number of connections that received the message
        """
        if channel not in self.subscriptions:
            logger.warning(f"Broadcast to unknown channel: {channel}")
            return 0

        subscribers = self.subscriptions[channel].copy()
        if not subscribers:
            return 0

        sent_count = 0
        disconnected = []

        for conn_id in subscribers:
            websocket = self.active_connections.get(conn_id)
            if not websocket:
                disconnected.append(conn_id)
                continue

            # Check filters
            filters = self.connection_filters.get(conn_id, {}).get(channel, {})
            if filters:
                # Check strategy_id filter
                filter_strategy_id = filters.get("strategy_id")
                if filter_strategy_id is not None:
                    msg_strategy_id = message.get("data", {}).get("strategy_id")
                    if msg_strategy_id != filter_strategy_id:
                        # Skip this connection - doesn't match filter
                        continue

            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"Error sending to {conn_id}: {e}")
                disconnected.append(conn_id)

        # Clean up disconnected clients
        for conn_id in disconnected:
            await self.disconnect(conn_id)

        return sent_count

    async def broadcast_to_all(self, message: dict) -> int:
        """Broadcast a message to all connected clients.

        Args:
            message: Message to send

        Returns:
            Number of connections that received the message
        """
        if not self.active_connections:
            return 0

        sent_count = 0
        disconnected = []

        for conn_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"Error sending to {conn_id}: {e}")
                disconnected.append(conn_id)

        # Clean up disconnected clients
        for conn_id in disconnected:
            await self.disconnect(conn_id)

        return sent_count

    async def send_to_connection(
        self,
        connection_id: str,
        message: dict,
    ) -> bool:
        """Send a message to a specific connection.

        Args:
            connection_id: Target connection ID
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        websocket = self.active_connections.get(connection_id)
        if not websocket:
            return False

        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False

    async def handle_message(
        self,
        connection_id: str,
        data: str,
    ) -> None:
        """Handle incoming message from a client.

        Args:
            connection_id: Connection ID
            data: Raw message data

        Message formats:
            Subscribe: {"action": "subscribe", "channels": ["trades"], "filters": {"strategy_id": 1}}
            Unsubscribe: {"action": "unsubscribe", "channels": ["trades"]}
            Ping: {"action": "ping"}
        """
        try:
            message = json.loads(data)
            action = message.get("action")

            if action == "subscribe":
                channels = message.get("channels", [])
                filters = message.get("filters")  # Optional filters
                response = await self.subscribe(connection_id, channels, filters)
                await self.send_to_connection(connection_id, response.model_dump())

                # Handle special price subscriptions via KirbyBridge
                if "prices" in channels:
                    starlisting_ids = message.get("starlisting_ids", [])
                    history = message.get("history", 500)
                    if starlisting_ids:
                        from api.websocket.kirby_bridge import get_kirby_bridge
                        bridge = get_kirby_bridge()
                        if bridge:
                            await bridge.subscribe(connection_id, starlisting_ids, history)
                        else:
                            logger.warning(
                                "KirbyBridge not available - price subscriptions disabled"
                            )

            elif action == "unsubscribe":
                channels = message.get("channels", [])
                response = await self.unsubscribe(connection_id, channels)
                await self.send_to_connection(connection_id, response.model_dump())

                # Handle price unsubscriptions
                if "prices" in channels:
                    from api.websocket.kirby_bridge import get_kirby_bridge
                    bridge = get_kirby_bridge()
                    if bridge:
                        await bridge.unsubscribe(connection_id)

            elif action == "ping":
                await self.send_to_connection(connection_id, {"type": "pong"})

            else:
                logger.debug(f"Unknown action from {connection_id}: {action}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {connection_id}: {data[:100]}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")

    def get_stats(self) -> dict:
        """Get connection statistics.

        Returns:
            Dictionary with connection stats
        """
        return {
            "total_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "subscriptions": {
                channel: len(subs)
                for channel, subs in self.subscriptions.items()
            },
        }


# Global connection manager instance
_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global ConnectionManager instance.

    Returns:
        ConnectionManager instance
    """
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
