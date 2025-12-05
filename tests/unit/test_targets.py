"""Unit tests for target creation functions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from training.targets import create_binary_target, create_3class_target


class TestBinaryTarget:
    """Test suite for binary classification target."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        # Create data with clear trends
        prices = [100.0]
        for i in range(99):
            change = np.random.uniform(-1, 1)
            prices.append(prices[-1] * (1 + change/100))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * 100,
        })
        return df

    def test_binary_target_basic(self, sample_data):
        """Test basic binary target creation."""
        target = create_binary_target(
            df=sample_data,
            lookahead_bars=4,
            threshold_pct=0.5,
        )

        # Should return Series
        assert isinstance(target, pd.Series)
        assert len(target) == len(sample_data)

        # Should only have 0 (short) and 1 (long), no 2 or NaN at end
        valid_values = target.dropna()
        assert set(valid_values.unique()).issubset({0, 1})

    def test_binary_target_no_threshold(self, sample_data):
        """Test binary target with threshold=0 (all moves)."""
        target = create_binary_target(
            df=sample_data,
            lookahead_bars=4,
            threshold_pct=0.0,
        )

        # With no threshold, should classify all moves
        valid_values = target.dropna()
        assert len(valid_values) > 0
        assert all(v in [0, 1] for v in valid_values)

    def test_binary_target_high_threshold(self, sample_data):
        """Test binary target with high threshold (fewer signals)."""
        target = create_binary_target(
            df=sample_data,
            lookahead_bars=4,
            threshold_pct=2.0,  # 2% threshold
        )

        # With high threshold, might have NaN (neutral moves)
        # But valid values should still only be 0 or 1
        valid_values = target.dropna()
        if len(valid_values) > 0:
            assert set(valid_values.unique()).issubset({0, 1})

    def test_binary_target_lookahead_bars(self, sample_data):
        """Test binary target with different lookahead periods."""
        target_2bars = create_binary_target(
            df=sample_data,
            lookahead_bars=2,
            threshold_pct=0.5,
        )

        target_8bars = create_binary_target(
            df=sample_data,
            lookahead_bars=8,
            threshold_pct=0.5,
        )

        # Both should be valid
        assert len(target_2bars) == len(sample_data)
        assert len(target_8bars) == len(sample_data)

        # Longer lookahead should have at least 8 NaN at end
        # (may have more due to threshold filtering)
        assert target_8bars.isna().sum() >= 8

    def test_binary_vs_3class_target(self, sample_data):
        """Test difference between binary and 3-class targets."""
        binary = create_binary_target(
            df=sample_data,
            lookahead_bars=4,
            threshold_pct=0.5,
        )

        three_class = create_3class_target(
            df=sample_data,
            lookahead_bars=4,
            threshold_pct=0.5,
        )

        # Binary should only have 0 and 1
        assert set(binary.dropna().unique()).issubset({0, 1})

        # 3-class should have 0, 1, and 2
        assert set(three_class.dropna().unique()).issubset({0, 1, 2})

    def test_binary_target_uptrend(self):
        """Test binary target on uptrend data."""
        # Create clear uptrend
        dates = pd.date_range('2023-01-01', periods=50, freq='15min')
        prices = [100 * (1.01 ** i) for i in range(50)]

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * 50,
        })

        target = create_binary_target(df, lookahead_bars=4, threshold_pct=0.5)

        # In uptrend, should have mostly 1 (long)
        valid_values = target.dropna()
        if len(valid_values) > 0:
            long_ratio = (valid_values == 1).sum() / len(valid_values)
            assert long_ratio > 0.5  # More longs than shorts

    def test_binary_target_downtrend(self):
        """Test binary target on downtrend data."""
        # Create clear downtrend
        dates = pd.date_range('2023-01-01', periods=50, freq='15min')
        prices = [100 * (0.99 ** i) for i in range(50)]

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * 50,
        })

        target = create_binary_target(df, lookahead_bars=4, threshold_pct=0.5)

        # In downtrend, should have mostly 0 (short)
        valid_values = target.dropna()
        if len(valid_values) > 0:
            short_ratio = (valid_values == 0).sum() / len(valid_values)
            assert short_ratio > 0.5  # More shorts than longs
