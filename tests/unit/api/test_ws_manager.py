"""Unit tests for WebSocket ConnectionManager."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from api.websocket.manager import ConnectionManager
from api.schemas.websocket import SubscribeResponse


class TestConnectionManager:
    """Tests for ConnectionManager."""

    @pytest.fixture
    def manager(self):
        """Create ConnectionManager instance."""
        return ConnectionManager(max_connections=10)

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_text = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_success(self, manager, mock_websocket):
        """Test successful connection."""
        conn_id = await manager.connect(mock_websocket)

        assert conn_id is not None
        assert conn_id in manager.active_connections
        assert len(manager.active_connections) == 1
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_at_capacity(self, mock_websocket):
        """Test connection rejected when at capacity."""
        manager = ConnectionManager(max_connections=1)

        # First connection succeeds
        conn1 = await manager.connect(mock_websocket)
        assert conn1 is not None

        # Second connection fails
        mock_websocket2 = AsyncMock()
        mock_websocket2.close = AsyncMock()
        conn2 = await manager.connect(mock_websocket2)

        assert conn2 is None
        mock_websocket2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test disconnection."""
        conn_id = await manager.connect(mock_websocket)

        # Subscribe to a channel first
        await manager.subscribe(conn_id, ["trades"])

        # Disconnect
        await manager.disconnect(conn_id)

        assert conn_id not in manager.active_connections
        assert conn_id not in manager.connection_channels
        assert conn_id not in manager.subscriptions["trades"]

    @pytest.mark.asyncio
    async def test_subscribe_to_channels(self, manager, mock_websocket):
        """Test subscribing to channels."""
        conn_id = await manager.connect(mock_websocket)

        response = await manager.subscribe(conn_id, ["trades", "positions"])

        assert response.status == "subscribed"
        assert "trades" in response.channels
        assert "positions" in response.channels
        assert conn_id in manager.subscriptions["trades"]
        assert conn_id in manager.subscriptions["positions"]

    @pytest.mark.asyncio
    async def test_subscribe_invalid_channel(self, manager, mock_websocket):
        """Test subscribing to invalid channel."""
        conn_id = await manager.connect(mock_websocket)

        response = await manager.subscribe(conn_id, ["invalid_channel"])

        # Invalid channel is ignored
        assert "invalid_channel" not in response.channels

    @pytest.mark.asyncio
    async def test_unsubscribe_from_channels(self, manager, mock_websocket):
        """Test unsubscribing from channels."""
        conn_id = await manager.connect(mock_websocket)

        # Subscribe first
        await manager.subscribe(conn_id, ["trades", "positions"])

        # Unsubscribe
        response = await manager.unsubscribe(conn_id, ["trades"])

        assert response.status == "unsubscribed"
        assert conn_id not in manager.subscriptions["trades"]
        assert conn_id in manager.subscriptions["positions"]

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, manager, mock_websocket):
        """Test broadcasting to a channel."""
        conn_id = await manager.connect(mock_websocket)
        await manager.subscribe(conn_id, ["trades"])

        # Reset send_json mock
        mock_websocket.send_json.reset_mock()

        message = {"type": "trade.new", "data": {"id": 1}}
        sent_count = await manager.broadcast_to_channel("trades", message)

        assert sent_count == 1
        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_unsubscribed_channel(self, manager, mock_websocket):
        """Test broadcasting when no subscribers."""
        await manager.connect(mock_websocket)
        # Don't subscribe to trades

        mock_websocket.send_json.reset_mock()

        message = {"type": "trade.new", "data": {"id": 1}}
        sent_count = await manager.broadcast_to_channel("trades", message)

        assert sent_count == 0
        # send_json was called once during connect (welcome message)
        # but not for the broadcast

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager):
        """Test broadcasting to all connections."""
        # Create multiple mock websockets
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1)
        await manager.connect(ws2)

        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        message = {"type": "announcement", "data": "test"}
        sent_count = await manager.broadcast_to_all(message)

        assert sent_count == 2
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_to_connection(self, manager, mock_websocket):
        """Test sending to specific connection."""
        conn_id = await manager.connect(mock_websocket)
        mock_websocket.send_json.reset_mock()

        message = {"type": "direct", "data": "test"}
        result = await manager.send_to_connection(conn_id, message)

        assert result is True
        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_to_invalid_connection(self, manager):
        """Test sending to non-existent connection."""
        result = await manager.send_to_connection("invalid_id", {"data": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, manager, mock_websocket):
        """Test handling subscribe message from client."""
        conn_id = await manager.connect(mock_websocket)
        mock_websocket.send_json.reset_mock()

        message = '{"action": "subscribe", "channels": ["trades"]}'
        await manager.handle_message(conn_id, message)

        # Should have sent subscription response
        assert mock_websocket.send_json.called
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["status"] == "subscribed"

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, manager, mock_websocket):
        """Test handling ping message from client."""
        conn_id = await manager.connect(mock_websocket)
        mock_websocket.send_json.reset_mock()

        message = '{"action": "ping"}'
        await manager.handle_message(conn_id, message)

        # Should have sent pong response
        mock_websocket.send_json.assert_called_with({"type": "pong"})

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, manager, mock_websocket):
        """Test handling invalid JSON message."""
        conn_id = await manager.connect(mock_websocket)

        # Should not raise exception
        await manager.handle_message(conn_id, "invalid json{")

    def test_get_stats(self, manager):
        """Test getting connection statistics."""
        stats = manager.get_stats()

        assert "total_connections" in stats
        assert "max_connections" in stats
        assert "subscriptions" in stats
        assert stats["total_connections"] == 0
        assert stats["max_connections"] == 10
