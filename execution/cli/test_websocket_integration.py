"""Integration test for WebSocket client and candle buffering.

This script tests the core infrastructure by:
1. Loading config from .env
2. Connecting to Kirby WebSocket API
3. Subscribing to a starlisting
4. Receiving and buffering candles
5. Saving state to database

Usage:
    python scripts/test_websocket_integration.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paper_trading import load_paper_trading_config
from execution.persistence.state_manager import StateManager, StateSnapshot
from execution.streaming.candle_buffer import MultiCandleBuffer
from execution.websocket.client import KirbyWebSocketClient
from execution.websocket.models import Candle, CandleUpdate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class WebSocketTester:
    """Integration tester for WebSocket infrastructure."""

    def __init__(self):
        """Initialize tester."""
        self.config = None
        self.ws_client = None
        self.candle_buffer = None
        self.state_manager = None
        self.candles_received = 0
        self.test_duration = 60  # Run for 60 seconds to see all message types

    async def run(self):
        """Run integration test."""
        try:
            logger.info("=" * 70)
            logger.info("WebSocket Integration Test")
            logger.info("=" * 70)

            # Step 1: Load configuration
            logger.info("\n[1/6] Loading configuration from .env...")
            self.config = load_paper_trading_config()
            logger.info(f"  ✓ WebSocket URL: {self.config.websocket.url}")
            logger.info(f"  ✓ API Key: {self.config.kirby_api_key[:10]}...")
            logger.info(f"  ✓ BTC/USDT 15m starlisting ID: {self.config.starlisting_btc_usdt_15m}")
            logger.info(f"  ✓ Initial capital: ${self.config.paper_trading_initial_capital:,.2f}")

            # Step 2: Initialize candle buffer
            logger.info("\n[2/6] Initializing candle buffer...")
            self.candle_buffer = MultiCandleBuffer(maxlen=500)
            logger.info(f"  ✓ Buffer initialized (maxlen=500)")

            # Step 3: Initialize state manager
            logger.info("\n[3/6] Initializing state manager...")
            self.state_manager = StateManager(self.config.db_path_obj)
            logger.info(f"  ✓ Database: {self.config.db_path_obj}")

            # Log test start event
            self.state_manager.log_event(
                "websocket_test_start",
                {"test_duration": self.test_duration},
                "INFO",
            )

            # Step 4: Connect to WebSocket
            logger.info("\n[4/6] Connecting to Kirby WebSocket...")
            self.ws_client = KirbyWebSocketClient(self.config.websocket)

            # Register candle callback
            self.ws_client.on_candle_update(self._on_candle)
            self.ws_client.on_error(self._on_error)

            await self.ws_client.connect()

            if not self.ws_client.is_connected:
                raise ConnectionError("Failed to connect to WebSocket")

            logger.info("  ✓ Connected successfully")

            # Step 5: Subscribe to BTC/USDT 15m
            logger.info("\n[5/6] Subscribing to BTC/USDT 15m...")
            starlisting_id = self.config.starlisting_btc_usdt_15m
            await self.ws_client.subscribe(
                starlisting_id=starlisting_id,
                historical_candles=50,  # Request 50 historical candles
            )
            logger.info(f"  ✓ Subscribed to starlisting {starlisting_id}")

            # Step 6: Wait for candles
            logger.info(f"\n[6/6] Receiving candles (will run for {self.test_duration}s)...")
            logger.info("  Press Ctrl+C to stop early\n")

            # Wait for test duration
            await asyncio.sleep(self.test_duration)

            # Print results
            logger.info("\n" + "=" * 70)
            logger.info("Test Results")
            logger.info("=" * 70)

            buffer = self.candle_buffer.get(starlisting_id)
            if buffer and len(buffer) > 0:
                logger.info(f"✓ Candles received: {self.candles_received}")
                logger.info(f"✓ Candles buffered: {len(buffer)}")

                oldest, latest = buffer.get_time_range()
                logger.info(f"✓ Time range: {oldest.isoformat()} to {latest.isoformat()}")

                latest_candle = buffer.get_latest()
                logger.info(f"✓ Latest candle: O={latest_candle.open:.2f}, "
                           f"H={latest_candle.high:.2f}, L={latest_candle.low:.2f}, "
                           f"C={latest_candle.close:.2f}")

                # Test DataFrame conversion
                df = buffer.to_dataframe(n=5)
                logger.info(f"✓ DataFrame conversion: {len(df)} rows")
                logger.info(f"\n{df[['open', 'high', 'low', 'close', 'volume']].to_string()}")

                # Save state snapshot
                snapshot = StateSnapshot(
                    id=None,
                    total_equity=self.config.paper_trading_initial_capital,
                    available_capital=self.config.paper_trading_initial_capital,
                    allocated_capital=0.0,
                    total_unrealized_pnl=0.0,
                    total_realized_pnl=0.0,
                    num_open_positions=0,
                    num_strategies_active=0,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
                snapshot_id = self.state_manager.save_snapshot(snapshot)
                logger.info(f"✓ State snapshot saved (ID: {snapshot_id})")

                # Log test completion
                self.state_manager.log_event(
                    "websocket_test_complete",
                    {
                        "candles_received": self.candles_received,
                        "candles_buffered": len(buffer),
                        "test_duration": self.test_duration,
                    },
                    "INFO",
                )

                logger.info("\n✅ All tests passed!")
            else:
                logger.error("❌ No candles received - check Kirby is running and serving data")

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Test interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Test failed: {e}", exc_info=True)
            if self.state_manager:
                self.state_manager.log_event(
                    "websocket_test_failed",
                    {"error": str(e)},
                    "ERROR",
                )
        finally:
            # Cleanup
            if self.ws_client:
                logger.info("\nDisconnecting from WebSocket...")
                await self.ws_client.disconnect()

            if self.state_manager:
                self.state_manager.close()

            logger.info("Test complete.\n")

    def _on_candle(self, candle_update: CandleUpdate):
        """Handle candle update."""
        self.candles_received += 1

        # Add to buffer
        self.candle_buffer.add_candle(
            candle_update.candle,
            starlisting_id=candle_update.starlisting_id,
            interval=candle_update.interval
        )

        # Log every 10th candle
        if self.candles_received % 10 == 0:
            logger.info(
                f"  Received {self.candles_received} candles | "
                f"Latest: {candle_update.candle.datetime.isoformat()} | "
                f"Close: ${candle_update.candle.close:,.2f}"
            )

    def _on_error(self, error: str, details: str = None):
        """Handle error message."""
        logger.error(f"WebSocket error: {error}")
        if details:
            logger.error(f"Details: {details}")


async def main():
    """Main entry point."""
    tester = WebSocketTester()
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
