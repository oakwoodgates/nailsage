"""Unit tests for backtest position sizing functionality."""

import pytest
import pandas as pd

from training.validation.backtest import BacktestEngine
from config.backtest import BacktestConfig


class TestConfidencePositionSizing:
    """Test suite for confidence-based position sizing."""

    @pytest.fixture
    def backtest_config(self):
        """Create backtest config."""
        return BacktestConfig(
            initial_capital=10000.0,
            taker_fee=0.001,  # 0.1%
            slippage_bps=5,
            max_leverage=2,
            max_position_size=1.0,
            enable_leverage=True,
        )

    @pytest.fixture
    def sample_data_with_signals(self):
        """Create sample data with signals."""
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100,
        }, index=dates)
        return df

    def test_position_sizing_without_confidence(self, backtest_config, sample_data_with_signals):
        """Test position sizing without confidence (should use full size)."""
        engine = BacktestEngine(backtest_config)

        # Create signals without confidence
        signals = pd.Series([0] * 10 + [1] * 5 + [0] * 85, index=sample_data_with_signals.index)

        # Run backtest without confidences
        metrics = engine.run(sample_data_with_signals, signals, confidences=None)

        # Should execute trades
        assert len(engine.trades) > 0

    def test_position_sizing_with_confidence(self, backtest_config, sample_data_with_signals):
        """Test position sizing with confidence values."""
        engine = BacktestEngine(backtest_config)

        # Create signals
        signals = pd.Series([0] * 10 + [1] * 5 + [0] * 85, index=sample_data_with_signals.index)

        # Create confidences (varying from 0.5 to 1.0)
        confidences = pd.Series([0.5] * 10 + [0.75] * 5 + [0.6] * 85, index=sample_data_with_signals.index)

        # Run backtest with confidences
        metrics = engine.run(sample_data_with_signals, signals, confidences=confidences)

        # Should execute trades
        assert len(engine.trades) > 0

    def test_confidence_scaling_calculation(self, backtest_config):
        """Test confidence scaling formula."""
        engine = BacktestEngine(backtest_config)

        # Test different confidence values
        # Formula: (confidence - 0.5) * 2
        # confidence <= 0.5 -> no scaling (full size)
        # 0.75 -> 50% of base size
        # 1.0 -> 100% of base size

        size_at_50 = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=0.5
        )
        size_no_conf = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=None
        )
        # At confidence=0.5, should be same as no confidence (full size)
        assert size_at_50 == pytest.approx(size_no_conf)

        size_at_75 = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=0.75
        )
        size_at_100 = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=1.0
        )

        # 0.75 confidence should be 50% of 1.0 confidence
        assert size_at_75 == pytest.approx(size_at_100 * 0.5, rel=0.01)

    def test_confidence_below_threshold(self, backtest_config):
        """Test that confidence below 0.5 results in full size (no scaling)."""
        engine = BacktestEngine(backtest_config)

        # Confidence below 0.5 should return full size (no scaling applied)
        size = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=0.3
        )
        size_no_conf = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=None
        )
        assert size == pytest.approx(size_no_conf)

    def test_confidence_exactly_at_threshold(self, backtest_config):
        """Test confidence exactly at 0.5 threshold."""
        engine = BacktestEngine(backtest_config)

        size = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=0.5
        )
        size_no_conf = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=None
        )
        # Threshold, returns full size (no scaling)
        assert size == pytest.approx(size_no_conf)

    def test_confidence_full_conviction(self, backtest_config):
        """Test confidence at 1.0 (full conviction)."""
        engine = BacktestEngine(backtest_config)

        size = engine._calculate_position_size(
            capital=10000.0,
            price=100.0,
            confidence=1.0
        )

        # Should use full position size with leverage
        # max_position_size=1.0, leverage=2
        expected_max_value = 10000.0 * 1.0 * 2  # $20,000
        expected_size = expected_max_value / 100.0  # 200 units
        assert size == pytest.approx(expected_size)
