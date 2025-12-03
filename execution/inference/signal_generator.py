"""Signal generator that converts model predictions into trading signals.

This module provides:
- Multi-mode classification support (binary, 3-class, 5-class)
- Automatic mode detection from prediction probabilities
- Confidence-based filtering
- Signal deduplication
- Cooldown periods between signals
- Position size calculation (percentage of strategy bankroll)
- Integration with StrategySignal format

Classification Modes:
    Binary (2-class): Aggressive trading, always in position
        - Class 0 → Short (-1)
        - Class 1 → Long (+1)

    3-class: Standard mode, can stay flat
        - Class 0 → Short (-1)
        - Class 1 → Neutral (0)
        - Class 2 → Long (+1)

    5-class: Graduated confidence levels
        - Class 0 → Strong Short (-1)
        - Class 1 → Weak Short (-1)
        - Class 2 → Neutral (0)
        - Class 3 → Weak Long (+1)
        - Class 4 → Strong Long (+1)

Example usage:
    generator = SignalGenerator(
        strategy_name="momentum_classifier_v1",
        asset="BTC/USDT",
        confidence_threshold=0.6,
        position_size_pct=10.0,  # 10% of bankroll per trade
        cooldown_bars=4,
    )

    # Convert prediction to signal (with current bankroll)
    # Mode is auto-detected from prediction.probabilities
    signal = generator.generate_signal(prediction, current_bankroll=9500.0)

    if signal:
        # Signal was generated (confidence threshold met, not duplicate)
        coordinator.process_signal(signal)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from execution.inference.predictor import Prediction
from portfolio.signal import StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generator.

    Attributes:
        strategy_name: Name of the strategy
        asset: Trading pair (e.g., 'BTC/USDT')
        confidence_threshold: Minimum confidence to generate signal (0.0-1.0)
        position_size_pct: Position size as percentage of current bankroll (0-100)
        cooldown_bars: Minimum bars between signals (prevents spam)
        allow_neutral_signals: If True, emit neutral signals (close positions)
    """

    strategy_name: str
    asset: str
    confidence_threshold: float = 0.6
    position_size_pct: float = 10.0  # 10% of bankroll per trade
    cooldown_bars: int = 4
    allow_neutral_signals: bool = True

    def __post_init__(self):
        """Validate config."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, "
                f"got {self.confidence_threshold}"
            )
        if not 0.0 < self.position_size_pct <= 100.0:
            raise ValueError(
                f"position_size_pct must be between 0 and 100, "
                f"got {self.position_size_pct}"
            )
        if self.cooldown_bars < 0:
            raise ValueError(
                f"cooldown_bars must be non-negative, "
                f"got {self.cooldown_bars}"
            )


class SignalGenerator:
    """
    Converts model predictions into trading signals.

    This class handles:
    - Confidence threshold filtering
    - Signal deduplication (don't emit same signal repeatedly)
    - Cooldown periods (minimum time between signals)
    - Position size calculation
    - StrategySignal creation

    Attributes:
        config: SignalGeneratorConfig
        _last_signal: Last signal emitted (for deduplication)
        _last_signal_timestamp: Timestamp of last signal (for cooldown)
        _signals_generated: Total count of signals generated
    """

    def __init__(self, config: SignalGeneratorConfig):
        """
        Initialize signal generator.

        Args:
            config: SignalGeneratorConfig instance
        """
        self.config = config
        self._last_signal: Optional[int] = None  # -1, 0, 1
        self._last_signal_timestamp: Optional[int] = None  # Unix ms
        self._signals_generated: int = 0

        logger.info(
            f"Initialized SignalGenerator for {config.strategy_name}",
            extra={
                "asset": config.asset,
                "confidence_threshold": config.confidence_threshold,
                "position_size_pct": config.position_size_pct,
                "cooldown_bars": config.cooldown_bars,
            }
        )

    def generate_signal(
        self,
        prediction: Prediction,
        candle_interval_ms: int = 900000,  # 15 minutes default
        has_open_positions: bool = False,
        current_bankroll: float = 10000.0,
    ) -> Optional[StrategySignal]:
        """
        Generate a trading signal from a prediction.

        Args:
            prediction: Prediction from ModelPredictor
            candle_interval_ms: Interval between candles in milliseconds
            has_open_positions: True if there are currently open positions
            current_bankroll: Current strategy bankroll in USD (for position sizing)

        Returns:
            StrategySignal if conditions met, None otherwise

        Conditions for generating signal:
        1. Confidence meets threshold
        2. Signal is different from last signal (deduplication)
           - UNLESS we have open positions and signal is NEUTRAL (must close positions!)
        3. Cooldown period has elapsed since last signal
        4. If neutral signal, check allow_neutral_signals config
        """
        # Detect number of classes and convert prediction to signal direction
        num_classes = self._detect_num_classes(prediction)
        signal_direction = self._prediction_to_signal(prediction.prediction, num_classes)

        # Check if neutral signals are allowed
        if signal_direction == 0 and not self.config.allow_neutral_signals:
            logger.info("Signal suppressed: Neutral signal not allowed")
            return None

        # Check confidence threshold
        if prediction.confidence < self.config.confidence_threshold:
            logger.info(
                f"Signal suppressed: Confidence {prediction.confidence:.2%} "
                f"< threshold {self.config.confidence_threshold:.2%}"
            )
            return None

        # Check deduplication (same signal as last)
        # EXCEPTION: If we have open positions and signal is NEUTRAL, we MUST emit it to close positions
        is_duplicate = self._last_signal is not None and signal_direction == self._last_signal
        is_closing_signal = has_open_positions and signal_direction == 0

        if is_duplicate and not is_closing_signal:
            signal_name = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}[signal_direction]
            logger.info(
                f"Signal suppressed: {signal_name} is duplicate of last signal"
            )
            return None

        # Log if we're emitting a duplicate NEUTRAL to close positions
        if is_duplicate and is_closing_signal:
            logger.info(
                f"Emitting duplicate NEUTRAL signal to close {has_open_positions} open position(s)"
            )

        # Check cooldown period
        if not self._is_cooldown_elapsed(prediction.timestamp, candle_interval_ms):
            bars_since_last = self._bars_since_last_signal(
                prediction.timestamp,
                candle_interval_ms
            )
            logger.info(
                f"Signal suppressed: Cooldown period "
                f"({bars_since_last}/{self.config.cooldown_bars} bars elapsed)"
            )
            return None

        # Calculate position size as percentage of current bankroll
        position_size_usd = current_bankroll * (self.config.position_size_pct / 100.0)

        # All checks passed - generate signal
        signal = StrategySignal(
            strategy_name=self.config.strategy_name,
            asset=self.config.asset,
            signal=signal_direction,
            confidence=prediction.confidence,
            timestamp=prediction.datetime,
            position_size_usd=position_size_usd,
        )

        # Update state
        self._last_signal = signal_direction
        self._last_signal_timestamp = prediction.timestamp
        self._signals_generated += 1

        logger.info(
            f"Generated signal #{self._signals_generated}: "
            f"{signal.direction_name.upper()} "
            f"(confidence: {signal.confidence:.2%}, size: ${position_size_usd:.2f})",
            extra={
                "signal": signal_direction,
                "confidence": prediction.confidence,
                "timestamp": prediction.timestamp,
                "position_size_usd": position_size_usd,
                "bankroll": current_bankroll,
            }
        )

        return signal

    def _detect_num_classes(self, prediction: Prediction) -> int:
        """
        Detect the number of classification classes from prediction.

        Detection logic:
        - If probabilities dict has 'neutral' key with value 0.0 → binary (2-class)
        - If max prediction class is 4 → 5-class
        - Otherwise → 3-class (default)

        Args:
            prediction: Prediction object with probabilities dict

        Returns:
            Number of classes (2, 3, or 5)
        """
        # Check for binary model (neutral probability is 0.0)
        if prediction.probabilities.get('neutral', 0.0) == 0.0:
            return 2

        # Check for 5-class model (has 'strong_short', 'weak_short', etc.)
        if 'strong_short' in prediction.probabilities or 'weak_long' in prediction.probabilities:
            return 5

        # Check by max prediction class value
        if prediction.prediction >= 4:
            return 5

        # Default to 3-class
        return 3

    def _prediction_to_signal(self, prediction_class: int, num_classes: int = 3) -> int:
        """
        Convert prediction class to signal direction.

        Supports multiple classification modes:

        Binary (2-class) - aggressive, always in position:
        - Class 0 → Short (-1)
        - Class 1 → Long (+1)

        3-class - standard, can stay flat:
        - Class 0 → Short (-1)
        - Class 1 → Neutral (0)
        - Class 2 → Long (+1)

        5-class - graduated confidence:
        - Class 0 → Strong Short (-1)
        - Class 1 → Weak Short (-1)
        - Class 2 → Neutral (0)
        - Class 3 → Weak Long (+1)
        - Class 4 → Strong Long (+1)

        Args:
            prediction_class: Prediction class from model
            num_classes: Number of classification classes (2, 3, or 5)

        Returns:
            Signal direction: -1=short, 0=neutral, 1=long
        """
        if num_classes == 2:
            # Binary: 0=short, 1=long (no neutral)
            mapping = {
                0: -1,  # short
                1: 1,   # long
            }
            return mapping.get(prediction_class, 0)

        elif num_classes == 5:
            # 5-class: strong/weak short, neutral, weak/strong long
            mapping = {
                0: -1,  # strong short
                1: -1,  # weak short
                2: 0,   # neutral
                3: 1,   # weak long
                4: 1,   # strong long
            }
            return mapping.get(prediction_class, 0)

        else:
            # 3-class (default): short, neutral, long
            mapping = {
                0: -1,  # short
                1: 0,   # neutral
                2: 1,   # long
            }
            return mapping.get(prediction_class, 0)

    def _is_cooldown_elapsed(
        self,
        current_timestamp: int,
        candle_interval_ms: int,
    ) -> bool:
        """
        Check if cooldown period has elapsed since last signal.

        Args:
            current_timestamp: Current timestamp (Unix ms)
            candle_interval_ms: Interval between candles in milliseconds

        Returns:
            True if cooldown elapsed or no previous signal
        """
        if self._last_signal_timestamp is None:
            return True  # No previous signal

        bars_since_last = self._bars_since_last_signal(
            current_timestamp,
            candle_interval_ms
        )

        return bars_since_last >= self.config.cooldown_bars

    def _bars_since_last_signal(
        self,
        current_timestamp: int,
        candle_interval_ms: int,
    ) -> int:
        """
        Calculate number of bars since last signal.

        Args:
            current_timestamp: Current timestamp (Unix ms)
            candle_interval_ms: Interval between candles in milliseconds

        Returns:
            Number of bars since last signal
        """
        if self._last_signal_timestamp is None:
            return float('inf')  # No previous signal

        time_diff_ms = current_timestamp - self._last_signal_timestamp
        return int(time_diff_ms / candle_interval_ms)

    def reset(self) -> None:
        """Reset signal generator state (useful for testing or restarting)."""
        self._last_signal = None
        self._last_signal_timestamp = None
        logger.info("Signal generator state reset")

    def get_stats(self) -> dict:
        """
        Get statistics about signal generation.

        Returns:
            Dict with statistics
        """
        return {
            "signals_generated": self._signals_generated,
            "last_signal": self._last_signal,
            "last_signal_timestamp": self._last_signal_timestamp,
            "strategy_name": self.config.strategy_name,
            "asset": self.config.asset,
            "confidence_threshold": self.config.confidence_threshold,
            "position_size_pct": self.config.position_size_pct,
        }
