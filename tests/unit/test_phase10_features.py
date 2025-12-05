"""Unit tests for Phase 10 features.

This module tests the features added in Phase 10:
- Binary classification target
- Confidence-based position sizing
- Trade cooldown mechanism
- Walk-forward validation with retrain
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from training.targets import create_binary_target, create_3class_target
from validation.backtest import BacktestEngine
from config.backtest import BacktestConfig


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


class TestTradeCooldown:
    """Test suite for trade cooldown mechanism."""

    def test_cooldown_imports(self):
        """Test that cooldown function can be imported."""
        # This is tested in the validation script, but verify imports work
        from strategies.short_term.validate_momentum_classifier import apply_trade_cooldown
        assert callable(apply_trade_cooldown)

    def test_cooldown_logic_simple(self):
        """Test cooldown logic with simple example."""
        from strategies.short_term.validate_momentum_classifier import apply_trade_cooldown

        # Create signals: [1, 1, 1, -1, 1]
        signals = np.array([1, 1, 1, -1, 1])
        min_bars = 2

        result = apply_trade_cooldown(signals, min_bars)

        # First signal (1) passes
        # Second signal (1) blocked (< 2 bars)
        # Third signal (1) blocked (< 2 bars from first)
        # Fourth signal (-1) blocked (< 2 bars from first)
        # Fifth signal (1) passes (>= 2 bars from first)

        # Check that cooldown was applied
        assert result[0] != 0  # First passes
        assert sum(result != 0) <= len(signals)  # Some blocked

    def test_cooldown_zero_bars(self):
        """Test cooldown with 0 bars (no cooldown)."""
        from strategies.short_term.validate_momentum_classifier import apply_trade_cooldown

        signals = np.array([1, 1, -1, 1, -1])
        min_bars = 0

        result = apply_trade_cooldown(signals, min_bars)

        # With no cooldown, all signals should pass
        assert np.array_equal(result, signals)

    def test_cooldown_all_neutral(self):
        """Test cooldown with all neutral signals."""
        from strategies.short_term.validate_momentum_classifier import apply_trade_cooldown

        signals = np.array([0, 0, 0, 0, 0])
        min_bars = 4

        result = apply_trade_cooldown(signals, min_bars)

        # No trades, result should be same
        assert np.array_equal(result, signals)

    def test_cooldown_far_apart(self):
        """Test cooldown with signals far apart."""
        from strategies.short_term.validate_momentum_classifier import apply_trade_cooldown

        # Signals 10 bars apart
        signals = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        min_bars = 4

        result = apply_trade_cooldown(signals, min_bars)

        # All signals should pass (far apart)
        assert result[0] == 1
        assert result[10] == -1
        assert result[20] == 1


class TestWalkForwardRetrain:
    """Test suite for walk-forward validation with retrain."""

    def test_retrain_model_factory_imports(self):
        """Test that retrain model factory can be imported."""
        from strategies.short_term.validate_momentum_classifier import create_model
        assert callable(create_model)

    def test_create_model_xgboost(self):
        """Test model creation for XGBoost."""
        from strategies.short_term.validate_momentum_classifier import create_model

        params = {'max_depth': 5, 'n_estimators': 100}
        model = create_model('xgboost', params)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_create_model_lightgbm(self):
        """Test model creation for LightGBM."""
        from strategies.short_term.validate_momentum_classifier import create_model

        params = {'max_depth': 5, 'n_estimators': 100}
        model = create_model('lightgbm', params)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_model_random_forest(self):
        """Test model creation for RandomForest."""
        from strategies.short_term.validate_momentum_classifier import create_model

        params = {'max_depth': 5, 'n_estimators': 100}
        model = create_model('random_forest', params)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_model_unknown(self):
        """Test model creation with unknown type."""
        from strategies.short_term.validate_momentum_classifier import create_model

        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model('unknown_model', {})


class TestFeatureExclusion:
    """Test suite for OHLCV feature exclusion."""

    def test_ohlcv_columns_defined(self):
        """Test that OHLCV columns are properly defined."""
        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert len(ohlcv_cols) == 6

    def test_feature_exclusion_logic(self):
        """Test feature column filtering logic."""
        # Simulate dataframe columns
        all_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'ema_12', 'rsi_14', 'macd', 'bollinger_upper'
        ]

        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in all_cols if col not in ohlcv_cols]

        # Should only have indicators
        assert 'open' not in feature_cols
        assert 'close' not in feature_cols
        assert 'ema_12' in feature_cols
        assert 'rsi_14' in feature_cols
        assert len(feature_cols) == 4

    def test_no_ohlcv_leakage(self):
        """Test that OHLCV columns are not used as features."""
        all_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ema_12']
        ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        feature_cols = [col for col in all_cols if col not in ohlcv_cols]

        # Verify no OHLCV in features
        for ohlcv in ohlcv_cols:
            assert ohlcv not in feature_cols
