"""Trade execution pipeline for orchestrating trading steps.

Provides a clean, testable way to orchestrate the complete trading workflow
from inference through execution.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Callable

import pandas as pd

from execution.inference.predictor import ModelPredictor, Prediction
from execution.inference.signal_generator import SignalGenerator
from execution.persistence.state_manager import StateManager, Signal as SignalRecord
from execution.simulator.order_executor import OrderExecutor
from execution.tracking.position_tracker import PositionTracker
from portfolio.signal import StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class TradeContext:
    """Context for a trading decision.

    Contains all information needed to make and execute a trading decision.

    Attributes:
        strategy_id: Database strategy ID
        strategy_name: Strategy name for logging
        starlisting_id: Starlisting ID for the asset
        timestamp: Current timestamp (Unix ms)
        price: Current price
        candle_df: DataFrame with historical candles for inference
        candle_interval_ms: Interval between candles
        enable_trading: Whether to actually execute trades
    """

    strategy_id: int
    strategy_name: str
    starlisting_id: int
    timestamp: int
    price: float
    candle_df: pd.DataFrame
    candle_interval_ms: int
    enable_trading: bool = True


@dataclass
class TradeResult:
    """Result of trade execution pipeline.

    Attributes:
        prediction: Model prediction
        signal: Generated signal (None if filtered)
        executed: Whether trade was executed
        error: Error message if pipeline failed
    """

    prediction: Optional[Prediction] = None
    signal: Optional[StrategySignal] = None
    executed: bool = False
    error: Optional[str] = None


class TradeExecutionPipeline:
    """
    Orchestrates the complete trading workflow.

    Executes the following steps:
    1. Run model inference on candle data
    2. Generate trading signal from prediction
    3. Save signal to database
    4. Execute trade (if signal generated and trading enabled)
    5. Update position P&L

    This class makes the trading workflow testable and extensible
    by separating orchestration from individual component logic.

    Attributes:
        predictor: ModelPredictor for inference
        signal_generator: SignalGenerator for signal creation
        order_executor: OrderExecutor for trade execution
        position_tracker: PositionTracker for P&L tracking
        state_manager: StateManager for persistence
    """

    def __init__(
        self,
        predictor: ModelPredictor,
        signal_generator: SignalGenerator,
        order_executor: OrderExecutor,
        position_tracker: PositionTracker,
        state_manager: StateManager,
    ):
        """
        Initialize trade execution pipeline.

        Args:
            predictor: ModelPredictor instance
            signal_generator: SignalGenerator instance
            order_executor: OrderExecutor instance
            position_tracker: PositionTracker instance
            state_manager: StateManager instance
        """
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.order_executor = order_executor
        self.position_tracker = position_tracker
        self.state_manager = state_manager

    async def execute(self, context: TradeContext) -> TradeResult:
        """
        Execute complete trading pipeline.

        Args:
            context: TradeContext with all necessary information

        Returns:
            TradeResult with outcome of pipeline execution
        """
        result = TradeResult()

        try:
            # Step 1: Run inference
            prediction = await self._run_inference(context)
            result.prediction = prediction

            if not prediction:
                return result

            # Step 2: Generate signal
            signal = await self._generate_signal(context, prediction)
            result.signal = signal

            # Step 3: Save signal to database
            await self._save_signal(context, signal, prediction)

            # Step 4: Execute trade if signal generated
            if signal and context.enable_trading:
                executed = await self._execute_trade(context, signal)
                result.executed = executed

            # Step 5: Update position P&L
            await self._update_positions(context)

        except Exception as e:
            logger.error(
                f"Pipeline execution failed: {e}",
                exc_info=True,
                extra={
                    "strategy": context.strategy_name,
                    "timestamp": context.timestamp,
                }
            )
            result.error = str(e)

        return result

    async def _run_inference(self, context: TradeContext) -> Optional[Prediction]:
        """
        Run model inference.

        Args:
            context: TradeContext

        Returns:
            Prediction from model
        """
        logger.debug(f"Running inference for {context.strategy_name}")

        prediction = await self.predictor.predict(
            context.candle_df,
            context.timestamp,
        )

        logger.info(
            f"Prediction: {prediction.get_signal()} "
            f"(confidence: {prediction.confidence:.2%})"
        )

        return prediction

    async def _generate_signal(
        self,
        context: TradeContext,
        prediction: Prediction,
    ) -> Optional[StrategySignal]:
        """
        Generate trading signal from prediction.

        Args:
            context: TradeContext
            prediction: Model prediction

        Returns:
            StrategySignal if generated, None if filtered
        """
        logger.debug(f"Generating signal for {context.strategy_name}")

        # Check if we have open positions for this strategy
        open_positions = self.position_tracker.get_open_positions(
            strategy_id=context.strategy_id
        )
        has_open_positions = len(open_positions) > 0

        signal = self.signal_generator.generate_signal(
            prediction,
            candle_interval_ms=context.candle_interval_ms,
            has_open_positions=has_open_positions,
        )

        if signal:
            logger.info(f"Signal generated: {signal.signal} @ ${context.price:,.2f}")
        else:
            logger.debug("No signal generated (filtered by confidence/cooldown)")

        return signal

    async def _save_signal(
        self,
        context: TradeContext,
        signal: Optional[StrategySignal],
        prediction: Prediction,
    ) -> None:
        """
        Save signal to database.

        Args:
            context: TradeContext
            signal: StrategySignal if generated
            prediction: Model prediction
        """
        # Map prediction to signal type string
        signal_map = {-1: 'short', 0: 'neutral', 1: 'long'}
        signal_type = signal_map.get(prediction.prediction, 'neutral')

        signal_record = SignalRecord(
            id=None,
            strategy_id=context.strategy_id,
            starlisting_id=context.starlisting_id,
            timestamp=context.timestamp,
            signal_type=signal_type,  # Map -1/0/1 to 'short'/'neutral'/'long'
            confidence=prediction.confidence,
            price_at_signal=context.price,
            was_executed=(signal is not None and context.enable_trading),
        )

        # Save to database (run in thread to avoid blocking)
        await asyncio.to_thread(
            self.state_manager.save_signal,
            signal_record
        )

        logger.debug(f"Saved signal to database: was_executed={signal_record.was_executed}")

    async def _execute_trade(
        self,
        context: TradeContext,
        signal: StrategySignal,
    ) -> bool:
        """
        Execute trade for signal.

        Args:
            context: TradeContext
            signal: StrategySignal to execute

        Returns:
            True if trade executed, False otherwise
        """
        logger.info(
            f"Executing trade for {context.strategy_name}: "
            f"signal={signal.signal}, price=${context.price:,.2f}"
        )

        # Get current position for this strategy
        positions = self.position_tracker.get_open_positions(
            strategy_id=context.strategy_id
        )
        position = positions[0] if positions else None

        # Determine action based on signal and current position
        if signal.signal == 1:  # LONG signal
            if position is None:
                # Open new long position
                await self._open_position(context, signal, "long")
                return True
            elif position.side == "short":
                # Close short, open long
                await self._close_position(context, position)
                await self._open_position(context, signal, "long")
                return True

        elif signal.signal == -1:  # SHORT signal
            if position is None:
                # Open new short position
                await self._open_position(context, signal, "short")
                return True
            elif position.side == "long":
                # Close long, open short
                await self._close_position(context, position)
                await self._open_position(context, signal, "short")
                return True

        elif signal.signal == 0:  # NEUTRAL / EXIT signal
            if position is not None:
                # Close any open position
                await self._close_position(context, position)
                return True

        logger.debug("No action taken (already in desired position)")
        return False

    async def _open_position(
        self,
        context: TradeContext,
        signal: StrategySignal,
        side: str,
    ) -> None:
        """Open a new position."""
        # Execute market order
        result = self.order_executor.execute_market_order(
            side=side,
            size_usd=signal.position_size_usd,
            current_price=context.price,
            timestamp=context.timestamp,
            position_id=None,  # Will be assigned after creation
            strategy_id=context.strategy_id,
            starlisting_id=context.starlisting_id,
            trade_type=f"open_{side}",
            signal_id=None,  # Signal ID not available yet
        )

        if result.success:
            # Create position record in tracker
            await asyncio.to_thread(
                self.position_tracker.open_position,
                strategy_id=context.strategy_id,
                starlisting_id=context.starlisting_id,
                side=side,
                size=result.size,
                entry_price=result.fill_price,
                entry_timestamp=context.timestamp,
                fees_paid=result.fees_usd,
            )

            logger.info(
                f"Opened {side} position: "
                f"size=${result.notional_usd:,.2f}, "
                f"price=${result.fill_price:,.2f}"
            )

    async def _close_position(self, context: TradeContext, position) -> None:
        """Close an existing position."""
        # Execute closing order
        opposite_side = "short" if position.side == "long" else "long"

        # Calculate position size in USD using current price
        size_usd = position.size * context.price

        result = self.order_executor.execute_market_order(
            side=opposite_side,
            size_usd=size_usd,
            current_price=context.price,
            timestamp=context.timestamp,
            position_id=position.id,
            strategy_id=context.strategy_id,
            starlisting_id=position.starlisting_id,
            trade_type=f"close_{position.side}",
            signal_id=None,
        )

        if result.success:
            # Close position in tracker
            await asyncio.to_thread(
                self.position_tracker.close_position,
                position_id=position.id,
                exit_price=result.fill_price,
                exit_timestamp=context.timestamp,
            )

            logger.info(
                f"Closed {position.side} position: "
                f"P&L=${result.fill_price - position.entry_price:+,.2f}"
            )

    async def _update_positions(self, context: TradeContext) -> None:
        """Update P&L for all open positions."""
        # TODO: Implement position P&L updates
        # For now, P&L is updated when positions are opened/closed
        # This can be added back by looping through open positions
        pass
