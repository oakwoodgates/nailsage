"""WebSocket request handlers."""

import logging

from fastapi import WebSocket, WebSocketDisconnect

from api.websocket.manager import ConnectionManager, get_connection_manager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handler for WebSocket connections."""

    def __init__(self, manager: ConnectionManager | None = None):
        """Initialize WebSocket handler.

        Args:
            manager: Connection manager instance (uses global if not provided)
        """
        self.manager = manager or get_connection_manager()

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection lifecycle.

        Args:
            websocket: WebSocket connection
        """
        connection_id = await self.manager.connect(websocket)
        if not connection_id:
            return  # Connection rejected

        try:
            while True:
                # Wait for messages from client
                data = await websocket.receive_text()
                await self.manager.handle_message(connection_id, data)

        except WebSocketDisconnect:
            logger.info(f"Client {connection_id} disconnected normally")
        except Exception as e:
            logger.error(f"Error in WebSocket handler for {connection_id}: {e}")
        finally:
            await self.manager.disconnect(connection_id)
