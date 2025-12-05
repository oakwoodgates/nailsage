"""Signal generation and filtering pipeline."""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from config.strategy import StrategyConfig
from utils.logger import get_training_logger

logger = get_training_logger()


class SignalPipeline:
    """
    Handles signal generation and filtering for training and backtesting.

    This class provides unified signal processing logic used by both
    training pipelines and live execution. It handles:

    - Prediction to signal conversion
    - Confidence-based filtering
    - Trade cooldown periods
    - Signal deduplication

    Attributes:
        config: Strategy configuration
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize signal pipeline.

        Args:
            config: Strategy configuration
        """
        self.config = config

        logger.info("Initialized SignalPipeline")

    def convert_predictions_to_signals(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        num_classes: int = 3
    ) -> np.ndarray:
        """
        Convert model predictions to trading signals.

        Args:
            predictions: Model class predictions
            probabilities: Prediction probabilities (n_samples x n_classes)
            num_classes: Number of classes (2 or 3)

        Returns:
            Trading signals (-1=short, 0=neutral, 1=long)
        """
        signals = np.zeros_like(predictions, dtype=int)

        # Map predictions to signals based on number of classes
        if num_classes == 2:
            # Binary: 0=short, 1=long
            signals[predictions == 1] = 1   # Long
            signals[predictions == 0] = -1  # Short
        else:
            # 3-class: 0=short, 1=neutral, 2=long
            signals[predictions == 2] = 1   # Long
            signals[predictions == 0] = -1  # Short
            # predictions == 1 stays 0 (neutral)

        # Apply confidence filtering
        confidence_threshold = getattr(self.config.target, 'confidence_threshold', 0.0)
        if probabilities is not None and confidence_threshold > 0:
            signals = self._apply_confidence_filtering(
                signals, probabilities, confidence_threshold
            )

        return signals

    def _apply_confidence_filtering(
        self,
        signals: np.ndarray,
        probabilities: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Apply confidence threshold filtering to signals.

        Args:
            signals: Trading signals
            probabilities: Prediction probabilities
            threshold: Minimum confidence threshold

        Returns:
            Filtered signals
        """
        max_probs = probabilities.max(axis=1)
        low_confidence_mask = max_probs < threshold

        filtered_signals = signals.copy()
        filtered_signals[low_confidence_mask] = 0

        suppressed_count = low_confidence_mask.sum()
        total_signals = (signals != 0).sum()

        if suppressed_count > 0:
            logger.info(
                f"Suppressed {suppressed_count}/{total_signals} signals "
                f"due to confidence < {threshold:.2%}"
            )

        return filtered_signals

    def apply_trade_cooldown(
        self,
        signals: np.ndarray,
        min_bars: int
    ) -> np.ndarray:
        """
        Apply cooldown between trades - suppress signals within min_bars of last trade.

        Args:
            signals: Array of signals (-1, 0, 1)
            min_bars: Minimum bars between trades (0 = no cooldown)

        Returns:
            Signals with cooldown applied
        """
        if min_bars <= 0:
            return signals

        result = signals.copy()
        last_trade_idx = -min_bars - 1  # Initialize to allow first trade

        for i in range(len(signals)):
            if signals[i] != 0:  # Non-neutral signal
                if i - last_trade_idx > min_bars:
                    # Enough bars have passed, allow trade
                    last_trade_idx = i
                else:
                    # Still in cooldown, suppress signal
                    result[i] = 0

        suppressed_by_cooldown = ((signals != 0) & (result == 0)).sum()
        if suppressed_by_cooldown > 0:
            logger.info(
                f"Suppressed {suppressed_by_cooldown} signals due to cooldown ({min_bars} bars)"
            )

        return result

    def process_signals_for_backtest(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        num_classes: int = 3
    ) -> np.ndarray:
        """
        Process predictions into signals for backtesting.

        This combines prediction conversion, confidence filtering, and cooldown.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
            num_classes: Number of classes

        Returns:
            Final trading signals for backtesting
        """
        # Convert predictions to signals
        signals = self.convert_predictions_to_signals(
            predictions, probabilities, num_classes
        )

        # Apply trade cooldown if configured
        min_bars_between_trades = self.config.backtest.min_bars_between_trades if self.config.backtest else 0
        if min_bars_between_trades > 0:
            signals = self.apply_trade_cooldown(signals, min_bars_between_trades)

        return signals

    def get_signal_statistics(self, signals: np.ndarray) -> dict:
        """
        Get statistics about generated signals.

        Args:
            signals: Array of trading signals

        Returns:
            Dictionary with signal statistics
        """
        total_signals = len(signals)
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()

        return {
            "total_signals": total_signals,
            "long_signals": int(long_signals),
            "short_signals": int(short_signals),
            "neutral_signals": int(neutral_signals),
            "long_percentage": float(long_signals / total_signals) if total_signals > 0 else 0.0,
            "short_percentage": float(short_signals / total_signals) if total_signals > 0 else 0.0,
            "neutral_percentage": float(neutral_signals / total_signals) if total_signals > 0 else 0.0,
        }
