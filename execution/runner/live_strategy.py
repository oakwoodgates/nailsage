"""Live strategy orchestrator for paper trading.

This module orchestrates a single strategy's execution by:
- Processing incoming candles from buffer
- Running inference with trained models
- Generating trading signals
- Executing trades
- Tracking positions and P&L
- Updating state in database

Example usage:
    # Initialize components
    predictor = ModelPredictor(model_id, registry, feature_engine)
    await predictor.load_model()

    signal_generator = SignalGenerator(config)
    order_executor = OrderExecutor(config)
    position_tracker = PositionTracker(state_manager)

    # Create strategy
    strategy = LiveStrategy(
        strategy_id=1,
        strategy_name="momentum_classifier_v1",
        starlisting_id=1,
        asset="BTC/USDT",
        candle_interval_ms=900000,  # 15 minutes
        predictor=predictor,
        signal_generator=signal_generator,
        order_executor=order_executor,
        position_tracker=position_tracker,
        state_manager=state_manager,
    )

    # Process a new candle
    await strategy.on_candle_update(candle_update)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from execution.inference.predictor import ModelPredictor, Prediction
from execution.inference.signal_generator import SignalGenerator
from execution.persistence.state_manager import StateManager, Signal as SignalRecord
from execution.simulator.order_executor import OrderExecutor
from execution.tracking.position_tracker import PositionTracker
from execution.websocket.models import CandleUpdate
from portfolio.signal import StrategySignal

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
    Orchestrates a single strategy's live execution.

    This class manages the complete lifecycle of a strategy:
    - Processes incoming candles
    - Runs model inference
    - Generates trading signals
    - Executes orders
    - Tracks positions
    - Updates P&L
    - Persists state

    Attributes:
        config: LiveStrategyConfig
        predictor: ModelPredictor for inference
        signal_generator: SignalGenerator for signal creation
        order_executor: OrderExecutor for trade execution
        position_tracker: PositionTracker for P&L
        state_manager: StateManager for persistence
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
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.order_executor = order_executor
        self.position_tracker = position_tracker
        self.state_manager = state_manager

        # State
        self._is_running = False
        self._candles_processed = 0
        self._signals_generated = 0
        self._trades_executed = 0

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
        if not self.predictor.is_loaded():
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

        This is the main event handler that orchestrates:
        1. Get candle data for inference
        2. Run model prediction
        3. Generate trading signal
        4. Execute trade if signal generated
        5. Update position P&L

        Args:
            candle_update: CandleUpdate from WebSocket
            candle_df_func: Callable that returns DataFrame with candles
        """
        if not self._is_running:
            return

        # Validate candle is for this strategy
        if candle_update.starlisting_id != self.config.starlisting_id:
            return

        candle = candle_update.candle
        timestamp = candle.timestamp
        price = candle.close

        logger.debug(
            f"Processing candle for {self.config.strategy_name}",
            extra={
                "timestamp": timestamp,
                "price": f"${price:,.2f}",
            }
        )

        try:
            # Get candle DataFrame for inference
            candle_df = candle_df_func(self.config.max_lookback)

            if len(candle_df) < self.config.max_lookback:
                logger.warning(
                    f"Insufficient data: need {self.config.max_lookback}, "
                    f"got {len(candle_df)}. Skipping candle."
                )
                return

            # 1. Run inference
            prediction = await self.predictor.predict(candle_df, timestamp)

            # 2. Generate signal
            signal = self.signal_generator.generate_signal(
                prediction,
                candle_interval_ms=self.config.candle_interval_ms,
            )

            # 3. Save signal to database (whether executed or not)
            await self._save_signal_record(signal, prediction, price)

            # 4. Execute trade if signal generated
            if signal and self.config.enable_trading:
                await self._execute_signal(signal, price, timestamp)

            # 5. Update open positions P&L
            await self._update_positions_pnl(price)

            self._candles_processed += 1

        except Exception as e:
            logger.error(
                f"Error processing candle: {e}",
                exc_info=True,
                extra={
                    "strategy_name": self.config.strategy_name,
                    "timestamp": timestamp,
                }
            )

    async def _save_signal_record(
        self,
        signal: Optional[StrategySignal],
        prediction: Prediction,
        price: float,
    ) -> None:
        """
        Save signal record to database.

        Args:
            signal: StrategySignal if generated, None otherwise
            prediction: Prediction from model
            price: Price at signal time
        """
        signal_record = SignalRecord(
            id=None,
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
            signal_type=prediction.get_signal().lower(),  # 'long', 'short', 'neutral'
            confidence=prediction.confidence,
            price_at_signal=price,
            timestamp=prediction.timestamp,
            was_executed=(signal is not None),
            rejection_reason=None if signal else "confidence_threshold_or_dedup",
        )

        # Save in thread pool (I/O operation)
        await asyncio.to_thread(
            self.state_manager.save_signal,
            signal_record
        )

    async def _execute_signal(
        self,
        signal: StrategySignal,
        price: float,
        timestamp: int,
    ) -> None:
        """
        Execute a trading signal.

        Handles both opening new positions and closing existing ones.

        Args:
            signal: StrategySignal to execute
            price: Current market price
            timestamp: Current timestamp
        """
        # Check if we already have an open position
        has_position = self.position_tracker.has_open_position(
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
        )

        # Determine action
        if signal.signal == 0:
            # Neutral signal - close position if we have one
            if has_position:
                await self._close_position(price, timestamp, "neutral_signal")
            else:
                logger.debug("Neutral signal but no position to close")

        elif signal.signal == 1:
            # Long signal
            if has_position:
                # Close existing position first
                await self._close_position(price, timestamp, "new_signal")

            # Open new long position
            await self._open_position('long', signal, price, timestamp)

        elif signal.signal == -1:
            # Short signal
            if has_position:
                # Close existing position first
                await self._close_position(price, timestamp, "new_signal")

            # Open new short position
            await self._open_position('short', signal, price, timestamp)

    async def _open_position(
        self,
        side: str,
        signal: StrategySignal,
        price: float,
        timestamp: int,
    ) -> None:
        """
        Open a new position.

        Args:
            side: 'long' or 'short'
            signal: StrategySignal that triggered this
            price: Current market price
            timestamp: Current timestamp
        """
        # Determine trade type
        trade_type = 'open_long' if side == 'long' else 'open_short'

        # Execute order
        order_result = await asyncio.to_thread(
            self.order_executor.execute_market_order,
            side=side,
            size_usd=signal.position_size_usd,
            current_price=price,
            timestamp=timestamp,
            position_id=0,  # Placeholder, will be updated
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
            trade_type=trade_type,
        )

        if not order_result.success:
            logger.warning(
                f"Order rejected: {order_result.rejection_reason}",
                extra={"signal": signal}
            )
            return

        # Open position
        position = await asyncio.to_thread(
            self.position_tracker.open_position,
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
            side=side,
            size=order_result.size,
            entry_price=order_result.fill_price,
            entry_timestamp=timestamp,
            fees_paid=order_result.fees_usd,
        )

        # Update trade with position_id
        order_result.trade.position_id = position.id

        # Save trade
        await asyncio.to_thread(
            self.state_manager.save_trade,
            order_result.trade
        )

        self._trades_executed += 1
        self._signals_generated += 1

        logger.info(
            f"Opened {side} position",
            extra={
                "position_id": position.id,
                "size": f"{order_result.size:.6f}",
                "fill_price": f"${order_result.fill_price:,.2f}",
                "notional_usd": f"${order_result.notional_usd:,.2f}",
            }
        )

    async def _close_position(
        self,
        price: float,
        timestamp: int,
        reason: str,
    ) -> None:
        """
        Close existing position.

        Args:
            price: Current market price
            timestamp: Current timestamp
            reason: Reason for closing
        """
        # Get open position
        positions = self.position_tracker.get_open_positions(
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
        )

        if not positions:
            logger.warning("No position to close")
            return

        position = positions[0]  # Should only be one

        # Determine trade type and order side
        if position.side == 'long':
            trade_type = 'close_long'
            order_side = 'short'  # Sell to close long
        else:
            trade_type = 'close_short'
            order_side = 'long'  # Buy to close short

        # Calculate notional value
        notional_usd = position.size * price

        # Execute closing order
        order_result = await asyncio.to_thread(
            self.order_executor.execute_market_order,
            side=order_side,
            size_usd=notional_usd,
            current_price=price,
            timestamp=timestamp,
            position_id=position.id,
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
            trade_type=trade_type,
        )

        if not order_result.success:
            logger.warning(
                f"Close order rejected: {order_result.rejection_reason}",
                extra={"position_id": position.id}
            )
            return

        # Close position
        closed_position = await asyncio.to_thread(
            self.position_tracker.close_position,
            position_id=position.id,
            exit_price=order_result.fill_price,
            exit_timestamp=timestamp,
            fees_paid=order_result.fees_usd,
            exit_reason=reason,
        )

        # Save closing trade
        await asyncio.to_thread(
            self.state_manager.save_trade,
            order_result.trade
        )

        self._trades_executed += 1

        logger.info(
            f"Closed {position.side} position",
            extra={
                "position_id": position.id,
                "entry_price": f"${position.entry_price:,.2f}",
                "exit_price": f"${order_result.fill_price:,.2f}",
                "realized_pnl": f"${closed_position.realized_pnl:,.2f}",
                "reason": reason,
            }
        )

    async def _update_positions_pnl(self, price: float) -> None:
        """
        Update P&L for all open positions.

        Args:
            price: Current market price
        """
        positions = self.position_tracker.get_open_positions(
            strategy_id=self.config.strategy_id,
            starlisting_id=self.config.starlisting_id,
        )

        for position in positions:
            await asyncio.to_thread(
                self.position_tracker.update_position_pnl,
                position_id=position.id,
                current_price=price,
            )

    def get_stats(self) -> dict:
        """
        Get statistics about strategy execution.

        Returns:
            Dict with statistics
        """
        return {
            "strategy_name": self.config.strategy_name,
            "is_running": self._is_running,
            "candles_processed": self._candles_processed,
            "signals_generated": self._signals_generated,
            "trades_executed": self._trades_executed,
            "position_stats": self.position_tracker.get_stats(),
        }
