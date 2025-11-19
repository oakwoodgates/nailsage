"""Main entry point for paper trading engine.

This script:
- Loads configuration from .env
- Initializes all components (predictor, signal generator, executor, tracker)
- Connects to Kirby WebSocket API
- Runs live strategies on streaming data
- Handles graceful shutdown

Usage:
    python scripts/run_paper_trading.py
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.feature import FeatureConfig
from config.paper_trading import load_paper_trading_config
from execution.inference.predictor import ModelPredictor
from execution.inference.signal_generator import SignalGenerator, SignalGeneratorConfig
from execution.persistence.state_manager import StateManager
from execution.runner.live_strategy import LiveStrategy, LiveStrategyConfig
from execution.simulator.order_executor import OrderExecutor, OrderExecutorConfig
from execution.streaming.candle_buffer import MultiCandleBuffer
from execution.tracking.position_tracker import PositionTracker
from execution.websocket.client import KirbyWebSocketClient
from execution.websocket.models import CandleUpdate
from features.engine import FeatureEngine
from models.registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Main paper trading engine.

    Orchestrates all components and handles WebSocket events.
    """

    def __init__(self):
        """Initialize paper trading engine."""
        self.config = None
        self.ws_client = None
        self.candle_buffer = None
        self.state_manager = None
        self.position_tracker = None
        self.order_executor = None
        self.model_registry = None
        self.strategies = {}  # strategy_id -> LiveStrategy
        self._running = False

        logger.info("=" * 70)
        logger.info("NailSage Paper Trading Engine")
        logger.info("=" * 70)

    async def initialize(self):
        """Initialize all components."""
        logger.info("\n[1/6] Loading configuration...")
        self.config = load_paper_trading_config()
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  WebSocket: {self.config.websocket.url}")
        logger.info(f"  Initial capital: ${self.config.paper_trading_initial_capital:,.2f}")

        logger.info("\n[2/6] Initializing database...")
        self.state_manager = StateManager(self.config.db_path_obj)
        logger.info(f"  Database: {self.config.db_path_obj}")

        logger.info("\n[3/6] Initializing components...")
        self.candle_buffer = MultiCandleBuffer(maxlen=500)
        self.position_tracker = PositionTracker(self.state_manager)
        self.order_executor = OrderExecutor(OrderExecutorConfig())
        self.model_registry = ModelRegistry()
        logger.info("  ✓ Candle buffer initialized")
        logger.info("  ✓ Position tracker initialized")
        logger.info("  ✓ Order executor initialized")
        logger.info("  ✓ Model registry initialized")

        logger.info("\n[4/6] Setting up strategies...")
        await self._setup_strategies()

        logger.info("\n[5/6] Connecting to Kirby WebSocket...")
        self.ws_client = KirbyWebSocketClient(self.config.websocket)
        self.ws_client.on_candle_update(self._on_candle)
        self.ws_client.on_error(self._on_error)
        await self.ws_client.connect()
        logger.info("  ✓ Connected to Kirby")

        logger.info("\n[6/6] Subscribing to starlistings...")
        # Subscribe to BTC/USDT 15m (starlisting 1)
        await self.ws_client.subscribe(
            starlisting_id=self.config.starlisting_btc_usdt_15m,
            historical_candles=500,
        )
        logger.info(f"  ✓ Subscribed to starlisting {self.config.starlisting_btc_usdt_15m}")

        # TODO: Subscribe to SOL/USDT 4h when we have that strategy
        # await self.ws_client.subscribe(
        #     starlisting_id=self.config.starlisting_sol_usdt_4h,
        #     historical_candles=500,
        # )

        logger.info("\n" + "=" * 70)
        logger.info("Paper trading engine initialized successfully!")
        logger.info("=" * 70 + "\n")

    async def _setup_strategies(self):
        """
        Set up strategies from registered strategies in database.

        For MVP, we'll hardcode a single strategy.
        In production, this would load from database.
        """
        # TODO: Load strategies from database
        # For MVP, we'll use the latest momentum classifier model

        # Find latest momentum classifier model
        model_metadata = self.model_registry.get_latest_model(
            strategy_name="momentum_classifier",
            strategy_timeframe="short_term",
        )

        if not model_metadata:
            logger.warning(
                "No momentum_classifier model found in registry. "
                "Run training first: python strategies/short_term/train_momentum_classifier.py"
            )
            logger.info("Engine will run without strategies (monitoring only)")
            return

        logger.info(f"  Found model: {model_metadata.model_id}")
        logger.info(f"    Model type: {model_metadata.model_type}")
        logger.info(f"    Trained: {model_metadata.trained_at}")

        # Load feature config
        feature_config = FeatureConfig.model_validate(model_metadata.feature_config)
        feature_engine = FeatureEngine(feature_config)

        # Initialize predictor
        predictor = ModelPredictor(
            model_id=model_metadata.model_id,
            registry=self.model_registry,
            feature_engine=feature_engine,
        )
        await predictor.load_model()

        # Initialize signal generator
        signal_gen_config = SignalGeneratorConfig(
            strategy_name="momentum_classifier_v1",
            asset="BTC/USDT",
            confidence_threshold=0.6,
            position_size_usd=10000.0,
            cooldown_bars=4,
            allow_neutral_signals=True,
        )
        signal_generator = SignalGenerator(signal_gen_config)

        # Create live strategy
        strategy_config = LiveStrategyConfig(
            strategy_id=1,  # TODO: Get from database
            strategy_name="momentum_classifier_v1",
            starlisting_id=self.config.starlisting_btc_usdt_15m,
            asset="BTC/USDT",
            candle_interval_ms=900000,  # 15 minutes
            max_lookback=500,
            enable_trading=True,
        )

        strategy = LiveStrategy(
            config=strategy_config,
            predictor=predictor,
            signal_generator=signal_generator,
            order_executor=self.order_executor,
            position_tracker=self.position_tracker,
            state_manager=self.state_manager,
        )

        await strategy.start()
        self.strategies[strategy_config.starlisting_id] = strategy
        logger.info(f"  ✓ Strategy '{strategy_config.strategy_name}' initialized")

    def _on_candle(self, candle_update: CandleUpdate):
        """
        Handle incoming candle update.

        This is called synchronously by WebSocket client, so we need to
        schedule async processing.
        """
        # Add to buffer
        self.candle_buffer.add_candle(
            candle_update.candle,
            starlisting_id=candle_update.starlisting_id,
            interval=candle_update.interval,
        )

        # Process with strategy (if we have one for this starlisting)
        if candle_update.starlisting_id in self.strategies:
            strategy = self.strategies[candle_update.starlisting_id]

            # Create candle getter function for this starlisting
            def get_candles(n):
                buffer = self.candle_buffer.get(candle_update.starlisting_id)
                if buffer:
                    return buffer.to_dataframe(n=n)
                return None

            # Schedule async processing
            asyncio.create_task(
                strategy.on_candle_update(candle_update, get_candles)
            )

    def _on_error(self, error: str, details: str = None):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        if details:
            logger.error(f"Details: {details}")

    async def run(self):
        """Run paper trading engine (main loop)."""
        self._running = True

        try:
            logger.info("Starting paper trading engine...")
            logger.info("Press Ctrl+C to stop\n")

            # Run until interrupted
            while self._running:
                await asyncio.sleep(1)

                # Log status every 60 seconds
                if int(datetime.now().timestamp()) % 60 == 0:
                    self._log_status()

        except asyncio.CancelledError:
            logger.info("\nShutdown signal received")
        finally:
            await self.shutdown()

    def _log_status(self):
        """Log current status."""
        logger.info("-" * 70)
        logger.info("Status Update")
        logger.info("-" * 70)

        for starlisting_id, strategy in self.strategies.items():
            stats = strategy.get_stats()
            logger.info(f"Strategy: {stats['strategy_name']}")
            logger.info(f"  Candles processed: {stats['candles_processed']}")
            logger.info(f"  Signals generated: {stats['signals_generated']}")
            logger.info(f"  Trades executed: {stats['trades_executed']}")
            logger.info(f"  Open positions: {stats['position_stats']['num_open_positions']}")
            logger.info(f"  Unrealized P&L: ${stats['position_stats']['total_unrealized_pnl']:,.2f}")

        logger.info("-" * 70 + "\n")

    async def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("\nShutting down paper trading engine...")

        # Stop strategies
        for strategy in self.strategies.values():
            await strategy.stop()

        # Disconnect WebSocket
        if self.ws_client:
            await self.ws_client.disconnect()

        # Close database
        if self.state_manager:
            self.state_manager.close()

        logger.info("Shutdown complete.")


async def main():
    """Main entry point."""
    engine = PaperTradingEngine()

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal")
        engine._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize
        await engine.initialize()

        # Run
        await engine.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        await engine.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
