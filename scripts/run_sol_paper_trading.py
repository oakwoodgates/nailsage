"""Paper trading for SOL swing momentum strategy.

This script runs the profitable SOL swing model in paper trading with PRODUCTION settings.

Strategy Details:
- Asset: SOL/USDT
- Timeframe: 4H
- Model: RandomForest (profitable: +65.54% return in backtest)
- Lookahead: 6 bars (24 hours)
- Target: 1.5% threshold

Production Settings:
- Confidence threshold: 0.5 (only trade when >50% confident)
- Cooldown: 4 bars (16 hours between trades)
- Position size: $10,000 per trade

Usage:
    python scripts/run_sol_paper_trading.py --dry-run  # Test without live trading
    python scripts/run_sol_paper_trading.py            # Run live paper trading
"""

import asyncio
import argparse
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
from execution.persistence.state_manager import StateManager, Strategy
from execution.runner.live_strategy import LiveStrategy, LiveStrategyConfig
from execution.simulator.order_executor import OrderExecutor, OrderExecutorConfig
from execution.streaming.candle_buffer import MultiCandleBuffer
from execution.tracking.position_tracker import PositionTracker
from execution.websocket.client import KirbyWebSocketClient
from execution.websocket.models import Candle
from features.engine import FeatureEngine
from models.registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SOLPaperTradingEngine:
    """Paper trading engine for SOL swing momentum strategy."""

    def __init__(self, dry_run=False):
        """Initialize SOL paper trading engine.

        Args:
            dry_run: If True, don't execute any trades (monitoring only)
        """
        self.dry_run = dry_run
        self.config = None
        self.ws_client = None
        self.candle_buffer = None
        self.state_manager = None
        self.position_tracker = None
        self.order_executor = None
        self.model_registry = None
        self.live_strategy = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info("=" * 70)
        logger.info("SOL Swing Momentum Paper Trading")
        logger.info("=" * 70)
        if dry_run:
            logger.info("DRY RUN MODE: No trades will be executed")

    async def initialize(self):
        """Initialize all components."""
        logger.info("\n[1/7] Loading configuration...")
        self.config = load_paper_trading_config()
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  WebSocket: {self.config.websocket.url}")
        logger.info(f"  Initial capital: ${self.config.paper_trading_initial_capital:,.2f}")

        logger.info("\n[2/7] Initializing database...")
        self.state_manager = StateManager(self.config.db_path_obj)
        logger.info(f"  Database: {self.config.db_path_obj}")

        logger.info("\n[3/7] Initializing components...")
        self.candle_buffer = MultiCandleBuffer(maxlen=500)
        self.position_tracker = PositionTracker(self.state_manager)

        # Production executor settings (realistic fees and slippage for SOL)
        executor_config = OrderExecutorConfig(
            fee_rate=0.001,  # 0.1% taker fee
            slippage_bps=3.0,  # 3 bps (slightly lower than BTC due to better liquidity)
            min_order_size_usd=10.0,
            max_order_size_usd=50_000.0,
        )
        self.order_executor = OrderExecutor(executor_config)
        self.model_registry = ModelRegistry()
        logger.info("  ✓ Components initialized")

        logger.info("\n[4/7] Loading SOL swing momentum model...")
        await self._setup_strategy()

        logger.info("\n[5/7] Connecting to Kirby WebSocket...")
        self.ws_client = KirbyWebSocketClient(self.config.websocket)
        self.ws_client.on_candle_update(self._on_candle)
        self.ws_client.on_error(self._on_error)
        await self.ws_client.connect()
        logger.info("  ✓ Connected to Kirby")

        logger.info("\n[6/7] Subscribing to SOL/USDT 4H...")
        # Subscribe to SOL/USDT 4H with 500 historical candles
        # 500 candles @ 4H = ~83 days of history
        await self.ws_client.subscribe(
            starlisting_id=self.config.starlisting_sol_usdt_4h,
            history=500,
        )
        logger.info(f"  ✓ Subscribed to starlisting {self.config.starlisting_sol_usdt_4h} (SOL/USDT 4H)")

        logger.info("\n[7/7] Ready for trading!")
        logger.info("=" * 70)
        logger.info("PRODUCTION SETTINGS:")
        logger.info("  - Confidence threshold: 0.5 (50%)")
        logger.info("  - Cooldown: 4 bars (16 hours)")
        logger.info("  - Position size: $10,000")
        logger.info("  - Fees: 0.1% + 3 bps slippage")
        logger.info("=" * 70 + "\n")

    async def _setup_strategy(self):
        """Set up SOL swing momentum strategy."""
        # Find latest SOL swing momentum model
        model_metadata = self.model_registry.get_latest_model(
            strategy_name="sol_swing_momentum",
            strategy_timeframe="long_term",
        )

        if not model_metadata:
            raise RuntimeError(
                "No sol_swing_momentum model found in registry. "
                "Train it first: python strategies/short_term/train_momentum_classifier.py "
                "--config configs/strategies/sol_swing_momentum_v1.yaml"
            )

        logger.info(f"  Model ID: {model_metadata.model_id}")
        logger.info(f"  Model type: {model_metadata.model_type}")
        logger.info(f"  Trained: {model_metadata.trained_at}")

        # Log validation metrics if available
        if model_metadata.validation_metrics:
            logger.info(f"  Backtest return: {model_metadata.validation_metrics.get('total_return', 'N/A')}")
            logger.info(f"  Win rate: {model_metadata.validation_metrics.get('win_rate', 'N/A')}")

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

        # Initialize signal generator with PRODUCTION settings
        signal_gen_config = SignalGeneratorConfig(
            strategy_name="sol_swing_momentum_v1",
            asset="SOL/USDT",
            confidence_threshold=0.5,  # PRODUCTION: Only trade when >50% confident
            position_size_usd=10_000.0,  # PRODUCTION: Standard position size
            cooldown_bars=4,  # PRODUCTION: 16 hours cooldown (4 x 4H bars)
            allow_neutral_signals=True,
        )
        signal_generator = SignalGenerator(signal_gen_config)

        # Register strategy in database
        existing_strategies = self.state_manager.get_active_strategies()
        strategy_id = None
        for strat in existing_strategies:
            if (strat.strategy_name == "sol_swing_momentum_v1" and
                strat.version == "1.0" and
                strat.starlisting_id == self.config.starlisting_sol_usdt_4h):
                strategy_id = strat.id
                logger.info(f"  Using existing strategy (ID: {strategy_id})")
                break

        if strategy_id is None:
            strategy_record = Strategy(
                id=None,
                strategy_name="sol_swing_momentum_v1",
                version="1.0",
                starlisting_id=self.config.starlisting_sol_usdt_4h,
                interval="4h",
                model_id=model_metadata.model_id,
                config_path="configs/strategies/sol_swing_momentum_v1.yaml",
                is_active=True,
            )
            strategy_id = self.state_manager.save_strategy(strategy_record)
            logger.info(f"  Registered new strategy (ID: {strategy_id})")

        # Create live strategy
        live_strategy_config = LiveStrategyConfig(
            strategy_id=strategy_id,
            strategy_name="sol_swing_momentum_v1",
            starlisting_id=self.config.starlisting_sol_usdt_4h,
            asset="SOL/USDT",
            candle_interval_ms=14_400_000,  # 4 hours
            max_lookback=500,
            enable_trading=not self.dry_run,  # Disable trading in dry-run mode
        )

        self.live_strategy = LiveStrategy(
            config=live_strategy_config,
            predictor=predictor,
            signal_generator=signal_generator,
            order_executor=self.order_executor,
            position_tracker=self.position_tracker,
            state_manager=self.state_manager,
        )

        # Start the strategy
        await self.live_strategy.start()

        logger.info("  ✓ Strategy initialized")

    async def _on_candle(self, candle_update):
        """Handle candle updates from WebSocket."""
        try:
            # Add to buffer (candle_update.data is the Candle object)
            self.candle_buffer.add_candle(
                candle=candle_update.data,  # Extract Candle from CandleUpdate
                starlisting_id=candle_update.starlisting_id,
                interval=candle_update.interval,  # Use actual interval from update
            )

            # Process with live strategy
            if self.live_strategy:
                # Create function that returns DataFrame with N candles
                def get_candle_df(n: int):
                    buffer = self.candle_buffer.get(candle_update.starlisting_id)
                    if buffer is None:
                        raise ValueError(f"No buffer for starlisting {candle_update.starlisting_id}")
                    return buffer.to_dataframe(n)

                await self.live_strategy.on_candle_update(candle_update, get_candle_df)

        except Exception as e:
            logger.error(f"Error processing candle: {e}", exc_info=True)

    async def _on_error(self, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")

    async def run(self):
        """Run the paper trading engine."""
        self._running = True

        # Set up signal handlers for graceful shutdown (Unix only)
        # On Windows, signal handlers are not supported in asyncio
        import platform
        if platform.system() != 'Windows':
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        logger.info("Paper trading engine is running. Press Ctrl+C to stop.\n")

        # Wait for shutdown signal (will be interrupted by KeyboardInterrupt on Windows)
        await self._shutdown_event.wait()

    async def shutdown(self):
        """Graceful shutdown."""
        if not self._running:
            return

        logger.info("\n" + "=" * 70)
        logger.info("Shutting down paper trading engine...")
        logger.info("=" * 70)
        self._running = False
        self._shutdown_event.set()

    async def cleanup(self):
        """Clean up resources."""
        if self.ws_client:
            await self.ws_client.disconnect()
            logger.info("  ✓ Disconnected from WebSocket")

        if self.state_manager:
            self.state_manager.close()
            logger.info("  ✓ Closed database connection")

        logger.info("\nGoodbye! 👋\n")


async def main(dry_run=False):
    """Main entry point."""
    engine = SOLPaperTradingEngine(dry_run=dry_run)

    try:
        await engine.initialize()
        await engine.run()
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Always cleanup on exit
        if engine.state_manager or engine.ws_client:
            await engine.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOL swing momentum paper trading")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no trades executed, monitoring only)"
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(dry_run=args.dry_run))
    except KeyboardInterrupt:
        pass
