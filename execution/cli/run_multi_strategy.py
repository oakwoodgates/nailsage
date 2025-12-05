"""Multi-strategy paper trading runner for Docker.

This script runs multiple strategies in a single container, sharing WebSocket connection
and candle buffer. Strategies are configured via environment variables.

Environment Variables:
    EXCHANGE: Exchange name (e.g., 'binance', 'hyperliquid')
    STRATEGY_IDS: Comma-separated strategy IDs (e.g., 'btc_momentum_v1,btc_mean_rev_v1')
    STARLISTING_IDS: Comma-separated starlisting IDs (e.g., '1,2,3')
    DATABASE_URL: Database connection string
    KIRBY_WS_URL: Kirby WebSocket URL
    KIRBY_API_KEY: Kirby API key

Usage:
    python scripts/run_multi_strategy.py
"""

import asyncio
import logging
import os
import signal
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
from execution.websocket.models import CandleUpdate
from features.engine import FeatureEngine
from models.registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=os.getenv('PAPER_TRADING_LOG_LEVEL', 'INFO'),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MultiStrategyEngine:
    """
    Multi-strategy paper trading engine.

    Runs multiple strategies in a single process, sharing WebSocket connection
    and candle buffer.
    """

    def __init__(self):
        """Initialize multi-strategy engine."""
        self.exchange = os.getenv('EXCHANGE', 'unknown')
        self.strategy_ids = self._parse_csv_env('STRATEGY_IDS')
        self.starlisting_ids = self._parse_csv_env('STARLISTING_IDS', int_values=True)

        self.config = None
        self.ws_client = None
        self.candle_buffer = None
        self.state_manager = None
        self.position_tracker = None
        self.order_executor = None
        self.model_registry = None
        # Strategy storage: keyed by strategy_id (unique) not starlisting_id
        self.strategies: Dict[str, LiveStrategy] = {}  # strategy_id -> LiveStrategy
        # Mapping: starlisting_id -> list of strategy_ids (for routing candles)
        self.starlisting_to_strategies: Dict[int, List[str]] = {}
        self._running = False

        logger.info("=" * 70)
        logger.info(f"NailSage Multi-Strategy Engine ({self.exchange.upper()})")
        logger.info("=" * 70)
        logger.info(f"Strategies to run: {', '.join(self.strategy_ids) if self.strategy_ids else 'None'}")
        logger.info(f"Starlistings: {', '.join(map(str, self.starlisting_ids)) if self.starlisting_ids else 'None'}")

    def _parse_csv_env(self, key: str, int_values: bool = False) -> List:
        """Parse comma-separated environment variable."""
        value = os.getenv(key, '')
        if not value:
            return []
        items = [item.strip() for item in value.split(',') if item.strip()]
        if int_values:
            return [int(item) for item in items]
        return items

    async def initialize(self):
        """Initialize all components."""
        logger.info("\n[1/6] Loading configuration...")
        self.config = load_paper_trading_config()
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  WebSocket: {self.config.websocket.url}")
        logger.info(f"  Initial capital: ${self.config.paper_trading_initial_capital:,.2f}")

        logger.info("\n[2/6] Initializing database...")
        # Use DATABASE_URL env var (set by Docker) or fall back to config
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            self.state_manager = StateManager(database_url=database_url)
            logger.info(f"  Database: {database_url.split('@')[-1] if '@' in database_url else 'SQLite'}")
        else:
            self.state_manager = StateManager(db_path=self.config.db_path_obj)
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
        if not self.strategy_ids:
            logger.warning("  No strategies specified in STRATEGY_IDS")
            logger.info("  Engine will run in monitoring mode only")
        else:
            await self._setup_strategies()

        logger.info("\n[5/6] Connecting to Kirby WebSocket...")
        self.ws_client = KirbyWebSocketClient(self.config.websocket)
        self.ws_client.on_candle_update(self._on_candle)
        self.ws_client.on_error(self._on_error)
        await self.ws_client.connect()
        logger.info("  ✓ Connected to Kirby")

        logger.info("\n[6/6] Subscribing to starlistings...")
        if not self.starlisting_ids:
            logger.warning("  No starlistings specified in STARLISTING_IDS")
        else:
            # Subscribe to unique starlistings only (avoid duplicate subscriptions)
            unique_starlistings = set(self.starlisting_ids)
            for starlisting_id in unique_starlistings:
                await self.ws_client.subscribe(
                    starlisting_id=starlisting_id,
                    history=500,
                )
                # Show how many strategies use this starlisting
                strategy_count = len(self.starlisting_to_strategies.get(starlisting_id, []))
                logger.info(f"  ✓ Subscribed to starlisting {starlisting_id} ({strategy_count} strategies)")

        logger.info("\n" + "=" * 70)
        logger.info("Multi-strategy engine initialized successfully!")
        logger.info("=" * 70 + "\n")

    async def _setup_strategies(self):
        """Set up strategies from environment configuration."""
        for idx, strategy_id in enumerate(self.strategy_ids):
            try:
                await self._setup_single_strategy(strategy_id, idx)
            except Exception as e:
                logger.error(f"Failed to setup strategy {strategy_id}: {e}", exc_info=True)
                logger.warning(f"Continuing without strategy {strategy_id}")

    async def _setup_single_strategy(self, strategy_id: str, strategy_index: int):
        """Set up a single strategy."""
        logger.info(f"\n  Setting up strategy: {strategy_id}")

        # Load strategy config from YAML
        config_path = Path("strategies") / f"{strategy_id}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Strategy config not found: {config_path}")

        with open(config_path, 'r') as f:
            strategy_config = yaml.safe_load(f)

        strategy_name = strategy_config['strategy_name']
        strategy_timeframe = strategy_config['strategy_timeframe']
        version = strategy_config.get('version', 'v1')

        # Extract asset/symbol from data source file or strategy name
        # For now, parse from data source file (e.g., "merged_binance_SOL_USDT_perps_4h_*.parquet")
        data_source = strategy_config.get('data', {}).get('source_file', '')
        if 'SOL' in data_source.upper():
            symbol = 'SOL'
        elif 'BTC' in data_source.upper():
            symbol = 'BTC'
        elif 'ETH' in data_source.upper():
            symbol = 'ETH'
        else:
            # Fallback: parse from strategy_id
            symbol = strategy_id.split('_')[0].upper()

        logger.info(f"    Strategy name: {strategy_name}")
        logger.info(f"    Timeframe: {strategy_timeframe}")
        logger.info(f"    Version: {version}")
        logger.info(f"    Asset: {symbol}/USDT")

        # Find model using actual strategy name, timeframe, and version from config
        model_metadata = self.model_registry.get_latest_model(
            strategy_name=strategy_name,
            strategy_timeframe=strategy_timeframe,
            version=version,
        )

        if not model_metadata:
            logger.warning(f"    No model found for {strategy_name} ({strategy_timeframe}). Skipping strategy.")
            return

        logger.info(f"    Model: {model_metadata.model_id}")

        # Load feature config
        feature_config = FeatureConfig.model_validate(model_metadata.feature_config)
        # Disable cache for live trading (streaming data doesn't benefit from caching)
        feature_config.enable_cache = False
        feature_engine = FeatureEngine(feature_config)

        # Initialize predictor
        predictor = ModelPredictor(
            model_id=model_metadata.model_id,
            registry=self.model_registry,
            feature_engine=feature_engine,
        )
        await predictor.load_model()

        # Initialize signal generator
        # Get config from environment or use defaults
        # Position size is now a percentage of the strategy's current bankroll (default: 10%)
        signal_gen_config = SignalGeneratorConfig(
            strategy_name=strategy_id,
            asset=f"{symbol}/USDT",
            confidence_threshold=float(os.getenv(f'{strategy_id.upper()}_CONFIDENCE_THRESHOLD', '0.5')),
            position_size_pct=float(os.getenv(f'{strategy_id.upper()}_POSITION_SIZE_PCT', '10.0')),
            cooldown_bars=int(os.getenv(f'{strategy_id.upper()}_COOLDOWN_BARS', '4')),
            allow_neutral_signals=True,
        )
        signal_generator = SignalGenerator(signal_gen_config)

        # Map strategy to its corresponding starlisting ID by index
        # strategy_ids[0] -> starlisting_ids[0], strategy_ids[1] -> starlisting_ids[1], etc.
        if strategy_index < len(self.starlisting_ids):
            starlisting_id = self.starlisting_ids[strategy_index]
        else:
            # Fallback if starlisting IDs list is shorter than strategies
            starlisting_id = self.starlisting_ids[0] if self.starlisting_ids else 1
            logger.warning(f"    No starlisting ID for strategy index {strategy_index}, using {starlisting_id}")

        logger.info(f"    Starlisting ID: {starlisting_id}")

        # Register strategy in database
        existing_strategies = self.state_manager.get_active_strategies()
        db_strategy_id = None
        for strat in existing_strategies:
            if (strat.strategy_name == strategy_id and
                strat.version == version and
                strat.starlisting_id == starlisting_id):
                db_strategy_id = strat.id
                logger.info(f"    Using existing DB record (ID: {db_strategy_id})")
                break

        if db_strategy_id is None:
            strategy_record = Strategy(
                id=None,
                strategy_name=strategy_id,
                version=version,
                starlisting_id=starlisting_id,
                interval=os.getenv(f'{strategy_id.upper()}_INTERVAL', '15m'),
                model_id=model_metadata.model_id,
                config_path=None,
                is_active=True,
            )
            db_strategy_id = self.state_manager.save_strategy(strategy_record)
            logger.info(f"    Created DB record (ID: {db_strategy_id})")

        # Create live strategy
        strategy_config = LiveStrategyConfig(
            strategy_id=db_strategy_id,
            strategy_name=strategy_id,
            starlisting_id=starlisting_id,
            asset=f"{symbol}/USDT",
            candle_interval_ms=900000,  # 15 minutes (TODO: make configurable)
            max_lookback=500,
            enable_trading=True,
        )

        # Phase 1.3: Use LiveStrategy with testable components
        strategy = LiveStrategy(
            config=strategy_config,
            predictor=predictor,
            signal_generator=signal_generator,
            order_executor=self.order_executor,
            position_tracker=self.position_tracker,
            state_manager=self.state_manager,
        )
        # TODO: Phase 1.2 - Integrate RiskManager for pre-trade validation

        await strategy.start()

        # Store strategy by strategy_id (unique key, not starlisting_id)
        self.strategies[strategy_id] = strategy

        # Map starlisting to strategies for candle routing
        if starlisting_id not in self.starlisting_to_strategies:
            self.starlisting_to_strategies[starlisting_id] = []
        self.starlisting_to_strategies[starlisting_id].append(strategy_id)

        logger.info(f"    ✓ Strategy '{strategy_id}' initialized (starlisting {starlisting_id})")

    def _on_candle(self, candle_update: CandleUpdate):
        """Handle incoming candle update."""
        # Add to buffer
        self.candle_buffer.add_candle(
            candle_update.candle,
            starlisting_id=candle_update.starlisting_id,
            interval=candle_update.interval,
        )

        # Route candle to ALL strategies that use this starlisting
        starlisting_id = candle_update.starlisting_id
        if starlisting_id in self.starlisting_to_strategies:
            # Create candle getter function for this starlisting (shared by all strategies)
            def get_candles(n):
                buffer = self.candle_buffer.get(starlisting_id)
                if buffer:
                    return buffer.to_dataframe(n=n)
                return None

            # Process ALL strategies that use this starlisting
            for strategy_id in self.starlisting_to_strategies[starlisting_id]:
                strategy = self.strategies[strategy_id]
                # Schedule async processing for each strategy
                asyncio.create_task(
                    strategy.on_candle_update(candle_update, get_candles)
                )

    def _on_error(self, error: str, details: str = None):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        if details:
            logger.error(f"Details: {details}")

    async def run(self):
        """Run multi-strategy engine (main loop)."""
        self._running = True

        try:
            logger.info("Starting multi-strategy engine...")
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
        """Log current status with enhanced P&L and position details."""
        if not self.strategies:
            return

        logger.info("-" * 70)
        logger.info(f"Status Update ({self.exchange.upper()})")
        logger.info("-" * 70)

        for strategy_id, strategy in self.strategies.items():
            stats = strategy.get_stats()
            pos_stats = stats['position_stats']

            # Get bankroll info from database
            db_strategy = self.state_manager.get_strategy_by_id(stats['strategy_id'])
            initial_bankroll = db_strategy.initial_bankroll if db_strategy else 10000.0
            current_bankroll = db_strategy.current_bankroll if db_strategy else 10000.0
            bankroll_pnl = current_bankroll - initial_bankroll
            bankroll_pct = (bankroll_pnl / initial_bankroll * 100) if initial_bankroll > 0 else 0

            # Basic stats
            logger.info(f"Strategy: {stats['strategy_name']}")
            logger.info(
                f"  Candles: {stats['candles_processed']} | "
                f"Signals: {stats['signals_generated']} | "
                f"Trades: {stats['trades_executed']}"
            )

            # Bankroll stats
            logger.info("")
            logger.info("  Bankroll:")
            status = "ACTIVE" if current_bankroll > 0 else "DEPLETED"
            logger.info(
                f"    ${current_bankroll:,.2f} ({bankroll_pnl:+,.2f} / {bankroll_pct:+.1f}%) [{status}]"
            )

            # Performance stats
            total_pnl = pos_stats['total_unrealized_pnl'] + pos_stats['total_realized_pnl']
            logger.info("")
            logger.info("  Performance:")
            logger.info(
                f"    Total P&L: ${total_pnl:,.2f} "
                f"(Unrealized: ${pos_stats['total_unrealized_pnl']:,.2f} | "
                f"Realized: ${pos_stats['total_realized_pnl']:,.2f})"
            )

            # Win rate
            win_rate = pos_stats['win_rate']
            num_wins = pos_stats['num_wins']
            num_losses = pos_stats['num_losses']
            logger.info(
                f"    Win Rate: {win_rate:.1f}% ({num_wins}W / {num_losses}L) | "
                f"Open Positions: {pos_stats['num_open_positions']}"
            )

            # Show open position details if any
            if pos_stats['num_open_positions'] > 0:
                current_price = stats['current_price']
                open_positions = stats['open_positions']

                logger.info("")
                logger.info("  Open Position Details:")
                for position in open_positions:
                    pos_id = position.id if position.id is not None else "?"
                    pos_side = position.side.upper() if position.side else "UNKNOWN"
                    pos_pnl = position.unrealized_pnl if position.unrealized_pnl is not None else 0.0
                    logger.info(
                        f"    Position #{pos_id} [{pos_side}]: "
                        f"Entry ${position.entry_price:,.2f} → Current ${current_price:,.2f} | "
                        f"P&L: ${pos_pnl:,.2f}"
                    )

        logger.info("-" * 70 + "\n")

    async def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("\nShutting down...")

        # Stop strategies
        for strategy in self.strategies.values():
            await strategy.stop()

        # Disconnect WebSocket
        if self.ws_client:
            await self.ws_client.disconnect()

        # Close database
        if self.state_manager:
            self.state_manager.close()

        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    engine = MultiStrategyEngine()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\nReceived shutdown signal")
        engine._running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize and run
    await engine.initialize()
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
