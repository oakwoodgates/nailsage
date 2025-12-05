"""Tests for the signal pipeline."""

import numpy as np
import pytest

from config.strategy import StrategyConfig
from training.signal_pipeline import SignalPipeline


class TestSignalPipeline:
    """Test the signal generation and filtering pipeline."""

    def test_convert_predictions_to_signals_binary(self):
        """Test signal conversion for binary classification."""
        # Create minimal config for binary classification
        config = self._create_minimal_config(confidence_threshold=0.0)

        pipeline = SignalPipeline(config)

        # Test binary predictions (0=short, 1=long)
        predictions = np.array([0, 1, 0, 1])
        probabilities = np.array([
            [0.7, 0.3],  # short
            [0.2, 0.8],  # long
            [0.6, 0.4],  # short
            [0.1, 0.9],  # long
        ])

        signals = pipeline.convert_predictions_to_signals(predictions, probabilities, num_classes=2)

        expected = np.array([-1, 1, -1, 1])  # short, long, short, long
        np.testing.assert_array_equal(signals, expected)

    def test_convert_predictions_to_signals_3class(self):
        """Test signal conversion for 3-class classification."""
        config = self._create_minimal_config(confidence_threshold=0.0)

        pipeline = SignalPipeline(config)

        # Test 3-class predictions (0=short, 1=neutral, 2=long)
        predictions = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.array([
            [0.8, 0.1, 0.1],  # short
            [0.1, 0.8, 0.1],  # neutral
            [0.1, 0.1, 0.8],  # long
            [0.7, 0.2, 0.1],  # short
            [0.2, 0.6, 0.2],  # neutral
            [0.1, 0.2, 0.7],  # long
        ])

        signals = pipeline.convert_predictions_to_signals(predictions, probabilities, num_classes=3)

        expected = np.array([-1, 0, 1, -1, 0, 1])  # short, neutral, long, ...
        np.testing.assert_array_equal(signals, expected)

    def test_confidence_filtering(self):
        """Test that low confidence predictions are filtered out."""
        config = self._create_minimal_config(confidence_threshold=0.6)

        pipeline = SignalPipeline(config)

        predictions = np.array([0, 1, 2])
        probabilities = np.array([
            [0.8, 0.1, 0.1],  # confidence = 0.8 > 0.6, should pass
            [0.4, 0.4, 0.2],  # confidence = 0.4 < 0.6, should be filtered
            [0.1, 0.1, 0.8],  # confidence = 0.8 > 0.6, should pass
        ])

        signals = pipeline.convert_predictions_to_signals(predictions, probabilities, num_classes=3)

        expected = np.array([-1, 0, 1])  # short, filtered to neutral, long
        np.testing.assert_array_equal(signals, expected)

    def test_trade_cooldown(self):
        """Test trade cooldown functionality."""
        config = self._create_minimal_config()

        pipeline = SignalPipeline(config)

        # Signals: long, short, neutral, long (with cooldown of 2 bars)
        signals = np.array([1, -1, 0, 1])

        # Apply cooldown of 2 bars minimum between trades
        filtered_signals = pipeline.apply_trade_cooldown(signals, min_bars=2)

        # Should suppress the short signal (position 1) because it's within 2 bars of the long signal
        expected = np.array([1, 0, 0, 1])  # long, suppressed, neutral, long
        np.testing.assert_array_equal(filtered_signals, expected)

    def test_signal_statistics(self):
        """Test signal statistics calculation."""
        config = self._create_minimal_config()

        pipeline = SignalPipeline(config)

        signals = np.array([1, -1, 0, 1, -1, 0, 1])

        stats = pipeline.get_signal_statistics(signals)

        assert stats["total_signals"] == 7
        assert stats["long_signals"] == 3
        assert stats["short_signals"] == 2
        assert stats["neutral_signals"] == 2
        assert abs(stats["long_percentage"] - 0.42857) < 0.01
        assert abs(stats["short_percentage"] - 0.28571) < 0.01
        assert abs(stats["neutral_percentage"] - 0.28571) < 0.01

    def test_regression_target_rejected(self):
        """Regression targets should raise when generating signals."""
        class RegressionConfig:
            def __init__(self):
                self.target = type('Target', (), {
                    'confidence_threshold': 0.0,
                    'classes': None,
                    'type': 'regression'
                })()
                self.backtest = type('Backtest', (), {
                    'min_bars_between_trades': 0
                })()

        pipeline = SignalPipeline(RegressionConfig())
        with pytest.raises(ValueError):
            pipeline.process_signals_for_backtest(
                predictions=np.array([0.1, -0.2]),
                probabilities=None,
                num_classes=3
            )

    def _create_minimal_config(self, confidence_threshold=0.0):
        """Create a minimal StrategyConfig for testing."""
        class MinimalConfig:
            def __init__(self, confidence_threshold=0.0):
                self.target = type('Target', (), {
                    'confidence_threshold': confidence_threshold,
                    'classes': 3
                })()
                self.backtest = type('Backtest', (), {
                    'min_bars_between_trades': 0
                })()

        return MinimalConfig(confidence_threshold=confidence_threshold)
