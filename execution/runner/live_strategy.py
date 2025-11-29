"""Live strategy orchestrator with improved testability.

This implementation uses smaller, testable components:
- CandleCloseDetector: Handles candle close detection
- TradeExecutionPipeline: Orchestrates trading workflow

This design makes it easier to:
- Test individual components
- Mock dependencies
- Add new features (e.g., risk checks)
- Debug issues
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from execution.inference.predictor import ModelPredictor
from execution.inference.signal_generator import SignalGenerator
from execution.persistence.state_manager import StateManager
from execution.runner.candle_close_detector import CandleCloseDetector
from execution.runner.trade_execution_pipeline import TradeExecutionPipeline, TradeContext
from execution.simulator.order_executor import OrderExecutor
from execution.tracking.position_tracker import PositionTracker
from execution.websocket.models import CandleUpdate

logger = logging.getLogger(__name__)


@dataclass
class LiveStrategyConfig:
    """Configuration for live strategy.

    Attributes:
        strategy_id: Database strategy ID
        strategy_name: Strategy name (for logging/signals)
        starlisting_id: Kirby starlisting ID
        asset: Trading pair (e.g., 'BTC/USDT')
        candle_interval_ms: Interval between candles in milliseconds
        max_lookback: Maximum candles needed for features
        enable_trading: If False, run in dry-run mode (no actual trades)
    """

    strategy_id: int
    strategy_name: str
    starlisting_id: int
    asset: str
    candle_interval_ms: int
    max_lookback: int = 500
    enable_trading: bool = True


class LiveStrategy:
    """
    Live strategy orchestrator with improved architecture.

    Key architecture features:
    - Separated concerns (candle detection vs trade execution)
    - Testable components (each can be unit tested)
    - Easier to mock for testing
    - Clearer control flow
    - Easier to add features (e.g., risk management)

    Attributes:
        config: LiveStrategyConfig
        candle_detector: CandleCloseDetector for detecting candle closes
        pipeline: TradeExecutionPipeline for orchestrating trades
    """

    def __init__(
        self,
        config: LiveStrategyConfig,
        predictor: ModelPredictor,
        signal_generator: SignalGenerator,
        order_executor: OrderExecutor,
        position_tracker: PositionTracker,
        state_manager: StateManager,
    ):
        """
        Initialize live strategy.

        Args:
            config: LiveStrategyConfig
            predictor: ModelPredictor instance (must be loaded)
            signal_generator: SignalGenerator instance
            order_executor: OrderExecutor instance
            position_tracker: PositionTracker instance
            state_manager: StateManager instance
        """
        self.config = config

        # Create components
        self.candle_detector = CandleCloseDetector()
        self.pipeline = TradeExecutionPipeline(
            predictor=predictor,
            signal_generator=signal_generator,
            order_executor=order_executor,
            position_tracker=position_tracker,
            state_manager=state_manager,
        )

        # State
        self._is_running = False
        self._candles_processed = 0
        self._signals_generated = 0
        self._trades_executed = 0
        self._current_price = 0.0  # Latest candle close price

        logger.info(
            f"Initialized LiveStrategy: {config.strategy_name}",
            extra={
                "strategy_id": config.strategy_id,
                "starlisting_id": config.starlisting_id,
                "asset": config.asset,
                "enable_trading": config.enable_trading,
            }
        )

    async def start(self) -> None:
        """Start strategy execution."""
        if not self.pipeline.predictor.is_loaded():
            raise ValueError("Predictor model not loaded - call predictor.load_model() first")

        self._is_running = True
        logger.info(f"Started strategy: {self.config.strategy_name}")

    async def stop(self) -> None:
        """Stop strategy execution."""
        self._is_running = False
        logger.info(f"Stopped strategy: {self.config.strategy_name}")

    async def on_candle_update(
        self,
        candle_update: CandleUpdate,
        candle_df_func,  # Function that returns DataFrame with N candles
    ) -> None:
        """
        Process incoming candle update.

        This is the main event handler. It:
        1. Validates candle is for this strategy
        2. Detects if candle closed (using CandleCloseDetector)
        3. If candle closed, runs trade execution pipeline
        4. If still forming, just updates P&L

        Args:
            candle_update: CandleUpdate from WebSocket
            candle_df_func: Callable that returns DataFrame with candles
        """
        if not self._is_running:
            return

        # Validate candle is for this strategy
        if candle_update.starlisting_id != self.config.starlisting_id:
            return

        candle = candle_update.data
        timestamp = candle.timestamp
        price = candle.close

        # Store current price for status reporting
        self._current_price = price

        # Detect candle close
        try:
            close_event = self.candle_detector.process_candle(timestamp)
        except ValueError as e:
            logger.error(f"Candle detection error: {e}")
            return

        # If candle still forming (no close event), just update P&L
        if close_event is None:
            logger.debug(f"Candle still forming (timestamp {timestamp}), updating P&L only")
            await self._update_positions_pnl(price)
            return

        # Candle closed - run trading pipeline
        logger.info(
            f"Processing closed candle for {self.config.strategy_name}",
            extra={
                "timestamp": timestamp,
                "price": f"${price:,.2f}",
                "is_first": close_event.is_first_candle,
            }
        )

        try:
            # Get candle data
            candle_df = candle_df_func(self.config.max_lookback)

            if len(candle_df) < self.config.max_lookback:
                logger.warning(
                    f"Insufficient data: need {self.config.max_lookback}, "
                    f"got {len(candle_df)}. Skipping candle."
                )
                return

            # Create trade context
            context = TradeContext(
                strategy_id=self.config.strategy_id,
                strategy_name=self.config.strategy_name,
                starlisting_id=self.config.starlisting_id,
                timestamp=timestamp,
                price=price,
                candle_df=candle_df,
                candle_interval_ms=self.config.candle_interval_ms,
                enable_trading=self.config.enable_trading,
            )

            # Execute trading pipeline
            result = await self.pipeline.execute(context)

            # Update stats
            self._candles_processed += 1
            if result.signal:
                self._signals_generated += 1
            if result.executed:
                self._trades_executed += 1

            if result.error:
                logger.error(f"Pipeline execution failed: {result.error}")

        except Exception as e:
            logger.error(
                f"Error processing candle: {e}",
                exc_info=True,
                extra={
                    "strategy_name": self.config.strategy_name,
                    "timestamp": timestamp,
                }
            )

    async def _update_positions_pnl(self, price: float) -> None:
        """
        Update P&L for all open positions.

        Args:
            price: Current price
        """
        positions = self.pipeline.position_tracker.get_open_positions()
        for position in positions:
            await asyncio.to_thread(
                self.pipeline.position_tracker.update_position_pnl,
                position.id,
                price
            )

    def get_stats(self) -> dict:
        """
        Get strategy statistics.

        Returns:
            Dict with candles processed, signals generated, trades executed, P&L, and win rate
        """
        # Get position stats from position tracker (filtered by strategy_id)
        position_stats = self.pipeline.position_tracker.get_stats(strategy_id=self.config.strategy_id)

        # Get open positions for this strategy
        open_positions = self.pipeline.position_tracker.get_open_positions(strategy_id=self.config.strategy_id)

        return {
            "strategy_name": self.config.strategy_name,
            "candles_processed": self._candles_processed,
            "signals_generated": self._signals_generated,
            "trades_executed": self._trades_executed,
            "is_running": self._is_running,
            "candles_seen": self.candle_detector.get_candles_seen(),
            "current_price": self._current_price,
            "position_stats": position_stats,
            "open_positions": open_positions,
        }

    def __repr__(self) -> str:
        return (
            f"LiveStrategy("
            f"strategy={self.config.strategy_name}, "
            f"candles_processed={self._candles_processed}, "
            f"signals_generated={self._signals_generated}, "
            f"trades_executed={self._trades_executed})"
        )
