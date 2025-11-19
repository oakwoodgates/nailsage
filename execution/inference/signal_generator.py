"""Signal generator that converts model predictions into trading signals.

This module provides:
- Confidence-based filtering
- Signal deduplication
- Cooldown periods between signals
- Position size calculation
- Integration with StrategySignal format

Example usage:
    generator = SignalGenerator(
        strategy_name="momentum_classifier_v1",
        asset="BTC/USDT",
        confidence_threshold=0.6,
        position_size_usd=10000.0,
        cooldown_bars=4,
    )

    # Convert prediction to signal
    signal = generator.generate_signal(prediction)

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
        position_size_usd: Position size in USD
        cooldown_bars: Minimum bars between signals (prevents spam)
        allow_neutral_signals: If True, emit neutral signals (close positions)
    """

    strategy_name: str
    asset: str
    confidence_threshold: float = 0.6
    position_size_usd: float = 10000.0
    cooldown_bars: int = 4
    allow_neutral_signals: bool = True

    def __post_init__(self):
        """Validate config."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, "
                f"got {self.confidence_threshold}"
            )
        if self.position_size_usd <= 0:
            raise ValueError(
                f"position_size_usd must be positive, "
                f"got {self.position_size_usd}"
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
                "position_size_usd": config.position_size_usd,
                "cooldown_bars": config.cooldown_bars,
            }
        )

    def generate_signal(
        self,
        prediction: Prediction,
        candle_interval_ms: int = 900000,  # 15 minutes default
    ) -> Optional[StrategySignal]:
        """
        Generate a trading signal from a prediction.

        Args:
            prediction: Prediction from ModelPredictor
            candle_interval_ms: Interval between candles in milliseconds

        Returns:
            StrategySignal if conditions met, None otherwise

        Conditions for generating signal:
        1. Confidence meets threshold
        2. Signal is different from last signal (deduplication)
        3. Cooldown period has elapsed since last signal
        4. If neutral signal, check allow_neutral_signals config
        """
        # Convert prediction class to signal direction
        signal_direction = self._prediction_to_signal(prediction.prediction)

        # Check if neutral signals are allowed
        if signal_direction == 0 and not self.config.allow_neutral_signals:
            logger.debug("Neutral signal suppressed (allow_neutral_signals=False)")
            return None

        # Check confidence threshold
        if prediction.confidence < self.config.confidence_threshold:
            logger.debug(
                f"Prediction confidence {prediction.confidence:.2%} "
                f"below threshold {self.config.confidence_threshold:.2%}"
            )
            return None

        # Check deduplication (same signal as last)
        if self._last_signal is not None and signal_direction == self._last_signal:
            logger.debug(
                f"Signal {signal_direction} is duplicate of last signal, skipping"
            )
            return None

        # Check cooldown period
        if not self._is_cooldown_elapsed(prediction.timestamp, candle_interval_ms):
            bars_since_last = self._bars_since_last_signal(
                prediction.timestamp,
                candle_interval_ms
            )
            logger.debug(
                f"Cooldown period not elapsed "
                f"({bars_since_last}/{self.config.cooldown_bars} bars)"
            )
            return None

        # All checks passed - generate signal
        signal = StrategySignal(
            strategy_name=self.config.strategy_name,
            asset=self.config.asset,
            signal=signal_direction,
            confidence=prediction.confidence,
            timestamp=prediction.datetime,
            position_size_usd=self.config.position_size_usd,
        )

        # Update state
        self._last_signal = signal_direction
        self._last_signal_timestamp = prediction.timestamp
        self._signals_generated += 1

        logger.info(
            f"Generated signal #{self._signals_generated}: "
            f"{signal.direction_name.upper()} "
            f"(confidence: {signal.confidence:.2%})",
            extra={
                "signal": signal_direction,
                "confidence": prediction.confidence,
                "timestamp": prediction.timestamp,
            }
        )

        return signal

    def _prediction_to_signal(self, prediction_class: int) -> int:
        """
        Convert prediction class to signal direction.

        Args:
            prediction_class: 0=short, 1=neutral, 2=long

        Returns:
            Signal direction: -1=short, 0=neutral, 1=long
        """
        mapping = {
            0: -1,  # short
            1: 0,   # neutral
            2: 1,   # long
        }
        return mapping[prediction_class]

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
        }
