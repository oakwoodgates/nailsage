"""Integration tests for ML pipeline (Feature → Model → Signal → Backtest).

These tests verify that multiple components work together correctly,
without involving the paper trading module.

Test Coverage:
- Feature computation + Model prediction
- Model prediction + Signal generation
- End-to-end backtest flow
- Model registry + Predictor integration
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from config.backtest import BacktestConfig
from validation.backtest import BacktestEngine
from config.feature import FeatureConfig
from execution.inference.predictor import ModelPredictor
from execution.inference.signal_generator import SignalGenerator, SignalGeneratorConfig
from features.engine import FeatureEngine
from models.registry import ModelMetadata, ModelRegistry
from portfolio.signal import StrategySignal
from training.targets import create_binary_target


@pytest.fixture
def sample_price_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    # Generate realistic price data with trend
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='4H'),
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n),
        'num_trades': np.random.randint(100, 1000, n),
    }

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def feature_config():
    """Create basic feature configuration."""
    return FeatureConfig(
        indicators=[
            {"name": "sma", "params": {"period": 20}},
            {"name": "rsi", "params": {"period": 14}},
            {"name": "returns", "params": {"period": 1}},
        ]
    )


@pytest.fixture
def backtest_config():
    """Create backtest configuration."""
    return BacktestConfig(
        initial_capital=10000.0,
        taker_fee=0.001,
        slippage_bps=5,
        enable_confidence_sizing=True,
    )


class TestFeatureEnginePipeline:
    """Test FeatureEngine integration with model prediction."""

    def test_feature_computation_produces_valid_features(
        self, sample_price_data, feature_config
    ):
        """Test that FeatureEngine produces valid features for model."""
        engine = FeatureEngine(feature_config)

        # Compute features
        features = engine.compute_features(sample_price_data)

        # Verify output
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_price_data)

        # Check OHLCV excluded (Phase 10 feature)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            assert col not in features.columns

        # Check expected features exist
        assert 'sma_20' in features.columns
        assert 'rsi_14' in features.columns
        assert 'returns_1' in features.columns

        # Verify no inf/nan in features (except at edges)
        valid_rows = ~features.isna().any(axis=1)
        assert valid_rows.sum() > 100  # Should have valid data

    def test_features_compatible_with_sklearn_model(
        self, sample_price_data, feature_config
    ):
        """Test that computed features work with sklearn models."""
        engine = FeatureEngine(feature_config)

        # Compute features and target
        features = engine.compute_features(sample_price_data)
        target = create_binary_target(
            sample_price_data,
            lookahead_bars=4,
            threshold_pct=1.0
        )

        # Align features and target
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]

        assert len(X) > 50, "Need sufficient valid samples"

        # Train sklearn model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Verify model can predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})  # Binary predictions

        # Verify probabilities
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestModelPredictionPipeline:
    """Test Model prediction with FeatureEngine integration."""

    @pytest.fixture
    def mock_trained_model(self, sample_price_data, feature_config):
        """Create and train a real model for testing."""
        engine = FeatureEngine(feature_config)
        features = engine.compute_features(sample_price_data)
        target = create_binary_target(
            sample_price_data,
            lookahead_bars=4,
            threshold_pct=1.0
        )

        # Get valid data
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]

        # Train model
        model = RandomForestClassifier(
            n_estimators=20,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)

        return model, features.columns.tolist()

    @pytest.fixture
    def mock_registry(self, mock_trained_model, feature_config):
        """Create mock registry with trained model."""
        model, feature_names = mock_trained_model

        metadata = ModelMetadata(
            model_id="test_integration_model",
            strategy_name="test_strategy",
            strategy_timeframe="short_term",
            model_type="random_forest",
            version="v1",
            trained_at="2024-01-01T00:00:00",
            model_artifact_path="models/test.pkl",
            validation_metrics={"accuracy": 0.6},
            training_dataset_path="data/test.parquet",
            training_date_range=("2024-01-01", "2024-03-01"),
            validation_date_range=("2024-03-01", "2024-04-01"),
            feature_config=feature_config.model_dump(),
            model_config={"n_estimators": 20},
            target_config={"lookahead_bars": 4},
        )

        registry = MagicMock(spec=ModelRegistry)
        registry.get_model.return_value = metadata
        registry.load_model_artifact.return_value = model

        return registry

    @pytest.mark.asyncio
    async def test_predictor_end_to_end(
        self, sample_price_data, feature_config, mock_registry
    ):
        """Test ModelPredictor with real FeatureEngine."""
        engine = FeatureEngine(feature_config)

        # Create predictor
        predictor = ModelPredictor(
            model_id="test_integration_model",
            registry=mock_registry,
            feature_engine=engine,
        )

        # Load model
        await predictor.load_model()
        assert predictor.is_loaded()

        # Compute features
        features_df = engine.compute_features(sample_price_data)

        # Make prediction
        timestamp = int(sample_price_data.index[-1].timestamp() * 1000)
        prediction = await predictor.predict(features_df, timestamp)

        # Verify prediction
        assert prediction is not None
        assert prediction.signal in [-1, 0, 1]
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.timestamp == timestamp
        assert prediction.model_id == "test_integration_model"


class TestSignalGenerationPipeline:
    """Test signal generation from model predictions."""

    def test_prediction_to_signal_conversion(self):
        """Test converting model predictions to trading signals."""
        from execution.inference.predictor import Prediction

        # Create signal generator
        config = SignalGeneratorConfig(
            strategy_name="test_strategy",
            asset="BTC/USDT",
            confidence_threshold=0.6,
            position_size_usd=10000,
            cooldown_bars=0,  # No cooldown for this test
        )
        generator = SignalGenerator(config)

        # Test high-confidence LONG prediction → signal generated
        prediction_long = Prediction(
            signal=1,
            confidence=0.75,
            timestamp=1000,
            model_id="test",
        )
        signal = generator.generate_signal(prediction_long)

        assert signal is not None
        assert isinstance(signal, StrategySignal)
        assert signal.signal == 1  # LONG
        assert signal.confidence == 0.75
        assert signal.strategy_name == "test_strategy"

        # Test low-confidence prediction → no signal
        prediction_low = Prediction(
            signal=1,
            confidence=0.4,  # Below threshold
            timestamp=2000,
            model_id="test",
        )
        signal = generator.generate_signal(prediction_low)
        assert signal is None  # Filtered by confidence threshold

    def test_signal_deduplication(self):
        """Test that duplicate signals are filtered."""
        from execution.inference.predictor import Prediction

        config = SignalGeneratorConfig(
            strategy_name="test_strategy",
            asset="BTC/USDT",
            confidence_threshold=0.5,
            position_size_usd=10000,
            cooldown_bars=0,
        )
        generator = SignalGenerator(config)

        prediction = Prediction(
            signal=1,
            confidence=0.7,
            timestamp=1000,
            model_id="test",
        )

        # First signal should be generated
        signal1 = generator.generate_signal(prediction)
        assert signal1 is not None

        # Duplicate signal should be blocked
        signal2 = generator.generate_signal(prediction)
        assert signal2 is None  # Deduplicated


class TestBacktestIntegration:
    """Test end-to-end backtest with real components."""

    def test_backtest_with_binary_target(
        self, sample_price_data, backtest_config
    ):
        """Test complete backtest flow with binary target."""
        # Create binary target (Phase 10)
        target = create_binary_target(
            sample_price_data,
            lookahead_bars=4,
            threshold_pct=1.5,
        )

        # Create signals (simulate model predictions)
        signals = pd.Series(0, index=sample_price_data.index)

        # Add some signals where target is 1
        target_idx = target[target == 1].index
        if len(target_idx) > 0:
            # Take every other target signal to avoid overtrading
            signals.loc[target_idx[::2]] = 1

        # Run backtest
        engine = BacktestEngine(backtest_config)
        results = engine.run_backtest(
            df=sample_price_data,
            signals=signals,
        )

        # Verify results structure
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'num_trades' in results
        assert 'win_rate' in results

        # Verify reasonable results
        assert results['num_trades'] > 0, "Should have executed trades"
        assert -1.0 <= results['total_return'] <= 10.0, "Return should be reasonable"
        assert 0.0 <= results['win_rate'] <= 1.0, "Win rate should be [0, 1]"

    def test_backtest_with_confidence_sizing(
        self, sample_price_data
    ):
        """Test backtest with confidence-based position sizing."""
        # Enable confidence sizing
        config = BacktestConfig(
            initial_capital=10000.0,
            taker_fee=0.001,
            slippage_bps=5,
            enable_confidence_sizing=True,  # Phase 10 feature
        )

        # Create signals with varying confidence
        signals = pd.Series(0, index=sample_price_data.index)
        signals.iloc[10] = 1  # LONG signal
        signals.iloc[20] = 1  # LONG signal
        signals.iloc[30] = 1  # LONG signal

        # Add confidence scores
        confidence = pd.Series(0.5, index=sample_price_data.index)
        confidence.iloc[10] = 0.6  # Low confidence
        confidence.iloc[20] = 0.8  # Medium confidence
        confidence.iloc[30] = 0.95  # High confidence

        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest(
            df=sample_price_data,
            signals=signals,
            confidence=confidence,
        )

        # Verify confidence affected position sizing
        assert results['num_trades'] > 0
        # With confidence sizing, not all signals may result in trades
        # (some may be too small due to low confidence)


class TestModelRegistryIntegration:
    """Test ModelRegistry integration with other components."""

    def test_registry_stores_and_loads_metadata(self, tmp_path):
        """Test storing and retrieving model metadata."""
        # Create registry with temp directory
        registry = ModelRegistry(models_dir=tmp_path)

        # Create sample metadata
        metadata = ModelMetadata(
            model_id="test_model_123",
            strategy_name="test_strategy",
            strategy_timeframe="short_term",
            model_type="random_forest",
            version="v1",
            trained_at="2024-01-01T00:00:00",
            model_artifact_path=str(tmp_path / "test_model.pkl"),
            validation_metrics={"accuracy": 0.75},
            training_dataset_path="data/test.parquet",
            training_date_range=("2024-01-01", "2024-03-01"),
            validation_date_range=("2024-03-01", "2024-04-01"),
            feature_config={"indicators": []},
            model_config={"n_estimators": 100},
            target_config={"lookahead_bars": 4},
        )

        # Save metadata
        registry.save_model_metadata(metadata)

        # Retrieve metadata
        loaded = registry.get_model("test_model_123")

        assert loaded is not None
        assert loaded.model_id == metadata.model_id
        assert loaded.strategy_name == metadata.strategy_name
        assert loaded.validation_metrics == metadata.validation_metrics

    def test_registry_filters_by_strategy(self, tmp_path):
        """Test filtering models by strategy name."""
        registry = ModelRegistry(models_dir=tmp_path)

        # Save multiple models with different strategies
        for i, strategy in enumerate(['strategy_a', 'strategy_b', 'strategy_a']):
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                strategy_name=strategy,
                strategy_timeframe="short_term",
                model_type="random_forest",
                version="v1",
                trained_at=f"2024-01-0{i+1}T00:00:00",
                model_artifact_path=str(tmp_path / f"model_{i}.pkl"),
                validation_metrics={"accuracy": 0.7 + i*0.1},
                training_dataset_path="data/test.parquet",
                training_date_range=("2024-01-01", "2024-03-01"),
                validation_date_range=("2024-03-01", "2024-04-01"),
                feature_config={"indicators": []},
                model_config={"n_estimators": 100},
                target_config={"lookahead_bars": 4},
            )
            registry.save_model_metadata(metadata)

        # Get latest model for strategy_a
        latest = registry.get_latest_model(
            strategy_name="strategy_a",
            strategy_timeframe="short_term"
        )

        assert latest is not None
        assert latest.strategy_name == "strategy_a"
        # Should get model_2 (latest for strategy_a)
        assert latest.model_id == "model_2"


# Run integration tests with verbose output
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
