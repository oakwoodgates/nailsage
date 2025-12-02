"""Price stream bridge for forwarding Kirby data to WebSocket clients."""

import asyncio
import logging
from typing import Optional

from api.websocket.events import emit_price_event

logger = logging.getLogger(__name__)


class PriceStreamBridge:
    """
    Bridges Kirby WebSocket client to frontend WebSocket connections.

    Subscribes to Kirby candle updates and forwards them to connected clients
    who have subscribed to the 'prices' channel.
    """

    def __init__(self):
        """Initialize price stream bridge."""
        self._kirby_client = None
        self._running = False
        logger.info("PriceStreamBridge initialized")

    def set_kirby_client(self, kirby_client) -> None:
        """Set the Kirby WebSocket client.

        Args:
            kirby_client: KirbyWebSocketClient instance
        """
        self._kirby_client = kirby_client

        # Register callback for candle updates
        kirby_client.on_candle_update(self._on_candle)
        logger.info("PriceStreamBridge connected to Kirby client")

    async def _on_candle(self, candle_update) -> None:
        """Handle candle update from Kirby.

        Args:
            candle_update: CandleUpdate from Kirby WebSocket
        """
        try:
            await emit_price_event(
                starlisting_id=candle_update.starlisting_id,
                coin=candle_update.coin,
                interval=candle_update.interval,
                open=float(candle_update.data.open),
                high=float(candle_update.data.high),
                low=float(candle_update.data.low),
                close=float(candle_update.data.close),
                volume=float(candle_update.data.volume),
            )
        except Exception as e:
            logger.error(f"Error forwarding candle update: {e}")

    def start(self) -> None:
        """Start the price stream bridge."""
        self._running = True
        logger.info("PriceStreamBridge started")

    def stop(self) -> None:
        """Stop the price stream bridge."""
        self._running = False
        logger.info("PriceStreamBridge stopped")


# Global instance
_bridge: Optional[PriceStreamBridge] = None


def get_price_stream_bridge() -> PriceStreamBridge:
    """Get or create the global PriceStreamBridge instance.

    Returns:
        PriceStreamBridge instance
    """
    global _bridge
    if _bridge is None:
        _bridge = PriceStreamBridge()
    return _bridge
