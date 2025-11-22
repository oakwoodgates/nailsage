"""Unit tests for ModelPredictor."""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

from execution.inference.predictor import ModelPredictor, Prediction
from models.metadata import ModelMetadata
from features.engine import FeatureEngine
from models.registry import ModelRegistry


@pytest.fixture
def mock_metadata():
    """Create mock model metadata."""
    return ModelMetadata(
        model_id="test_model_123",
        strategy_name="test_strategy",
        strategy_timeframe="short_term",
        model_type="xgboost",
        version="v1",
        trained_at="2023-11-14T12:00:00",
        model_artifact_path="models/trained/test_model_123.pkl",
        validation_metrics={"accuracy": 0.75, "f1_score": 0.70},
        training_dataset_path="data/raw/test_data.parquet",
        training_date_range=("2023-01-01", "2023-10-31"),
        validation_date_range=("2023-11-01", "2023-11-14"),
        feature_config={"indicators": []},
        model_config={"max_depth": 5},
        target_config={"lookahead_bars": 4},
    )


@pytest.fixture
def mock_registry(mock_metadata):
    """Create mock model registry."""
    registry = MagicMock(spec=ModelRegistry)
    registry.get_model.return_value = mock_metadata
    return registry


@pytest.fixture
def mock_feature_engine():
    """Create mock feature engine."""
    engine = MagicMock(spec=FeatureEngine)
    engine.get_max_lookback.return_value = 200

    # Mock compute_features to return DataFrame with features
    def compute_features_mock(df):
        features_df = df.copy()
        features_df['ema_12'] = 100.0
        features_df['rsi_14'] = 50.0
        features_df['macd'] = 1.5
        features_df['bollinger_upper'] = 110.0
        features_df['bollinger_lower'] = 90.0
        return features_df

    engine.compute_features.side_effect = compute_features_mock
    return engine


@pytest.fixture
def mock_model():
    """Create mock sklearn model."""
    model = MagicMock()
    # Mock predict_proba to return class probabilities
    model.predict_proba.return_value = np.array([
        [0.15, 0.10, 0.75]  # short=15%, neutral=10%, long=75%
    ])
    return model


@pytest.fixture
def sample_candle_data():
    """Create sample candle data."""
    timestamps = pd.date_range('2023-11-14', periods=500, freq='15min')
    data = {
        'timestamp': timestamps,
        'open': np.random.uniform(99, 101, 500),
        'high': np.random.uniform(100, 102, 500),
        'low': np.random.uniform(98, 100, 500),
        'close': np.random.uniform(99, 101, 500),
        'volume': np.random.uniform(1000, 2000, 500),
    }
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    return df


class TestModelPredictor:
    """Test suite for ModelPredictor class."""

    def test_init(self, mock_registry, mock_feature_engine):
        """Test predictor initialization."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        assert predictor.model_id == "test_model_123"
        assert predictor.registry == mock_registry
        assert predictor.feature_engine == mock_feature_engine
        assert predictor.model is None
        assert predictor.metadata is None
        assert predictor._last_prediction is None
        assert len(predictor._prediction_cache) == 0

    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_registry, mock_feature_engine, mock_metadata, mock_model):
        """Test successful model loading."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Mock Path.exists to return True
        with patch('pathlib.Path.exists', return_value=True):
            # Mock joblib.load to return mock model
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        assert predictor.model is not None
        assert predictor.metadata == mock_metadata
        assert predictor.is_loaded() is True

    @pytest.mark.asyncio
    async def test_load_model_not_found_in_registry(self, mock_feature_engine):
        """Test model loading when model not in registry."""
        registry = MagicMock(spec=ModelRegistry)
        registry.get_model.return_value = None  # Model not found

        predictor = ModelPredictor(
            model_id="nonexistent_model",
            registry=registry,
            feature_engine=mock_feature_engine,
        )

        with pytest.raises(ValueError, match="Model .* not found in registry"):
            await predictor.load_model()

    @pytest.mark.asyncio
    async def test_load_model_artifact_not_found(self, mock_registry, mock_feature_engine):
        """Test model loading when artifact file not found."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Mock Path.exists to return False
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Model artifact not found"):
                await predictor.load_model()

    @pytest.mark.asyncio
    async def test_predict_success(self, mock_registry, mock_feature_engine, mock_metadata, mock_model, sample_candle_data):
        """Test successful prediction."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        # Make prediction
        current_timestamp = 1700000000000
        prediction = await predictor.predict(sample_candle_data, current_timestamp)

        assert isinstance(prediction, Prediction)
        assert prediction.timestamp == current_timestamp
        assert prediction.model_id == "test_model_123"
        assert prediction.prediction == 2  # Long (argmax of [0.15, 0.10, 0.75])
        assert prediction.confidence == 0.75
        assert prediction.probabilities['long'] == 0.75
        assert prediction.probabilities['short'] == 0.15
        assert prediction.probabilities['neutral'] == 0.10

    @pytest.mark.asyncio
    async def test_predict_model_not_loaded(self, mock_registry, mock_feature_engine, sample_candle_data):
        """Test prediction when model not loaded."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Try to predict without loading model
        with pytest.raises(ValueError, match="Model not loaded"):
            await predictor.predict(sample_candle_data, 1700000000000)

    @pytest.mark.asyncio
    async def test_predict_empty_dataframe(self, mock_registry, mock_feature_engine, mock_model):
        """Test prediction with empty DataFrame."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        # Try to predict with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty candle DataFrame"):
            await predictor.predict(empty_df, 1700000000000)

    @pytest.mark.asyncio
    async def test_predict_insufficient_data(self, mock_registry, mock_feature_engine, mock_model):
        """Test prediction with insufficient data."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        # Create DataFrame with less than max_lookback candles
        small_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-11-14', periods=50, freq='15min'),
            'open': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50,
        }).set_index('timestamp')

        with pytest.raises(ValueError, match="Insufficient data: need at least 200"):
            await predictor.predict(small_df, 1700000000000)

    @pytest.mark.asyncio
    async def test_predict_with_nan_features(self, mock_registry, mock_model):
        """Test prediction when features contain NaN."""
        # Create feature engine that returns NaN
        engine = MagicMock(spec=FeatureEngine)
        engine.get_max_lookback.return_value = 10

        def compute_features_nan(df):
            features_df = df.copy()
            features_df['ema_12'] = np.nan  # NaN feature
            features_df['rsi_14'] = 50.0
            return features_df

        engine.compute_features.side_effect = compute_features_nan

        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=engine,
        )

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-11-14', periods=20, freq='15min'),
            'open': [100] * 20,
            'high': [101] * 20,
            'low': [99] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20,
        }).set_index('timestamp')

        with pytest.raises(ValueError, match="Insufficient data for feature computation.*NaN"):
            await predictor.predict(df, 1700000000000)

    @pytest.mark.asyncio
    async def test_prediction_caching(self, mock_registry, mock_feature_engine, mock_model, sample_candle_data):
        """Test prediction caching."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        current_timestamp = 1700000000000

        # First prediction
        prediction1 = await predictor.predict(sample_candle_data, current_timestamp)
        assert mock_model.predict_proba.call_count == 1

        # Second prediction with same timestamp (should use cache)
        prediction2 = await predictor.predict(sample_candle_data, current_timestamp)
        assert mock_model.predict_proba.call_count == 1  # No additional call

        # Verify same prediction returned
        assert prediction1.timestamp == prediction2.timestamp
        assert prediction1.prediction == prediction2.prediction
        assert prediction1.confidence == prediction2.confidence

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, mock_registry, mock_feature_engine, mock_model, sample_candle_data):
        """Test cache size limit enforcement."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )
        predictor._max_cache_size = 10  # Small cache for testing

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        # Make predictions for 15 different timestamps
        for i in range(15):
            timestamp = 1700000000000 + (i * 900000)  # 15 min apart
            await predictor.predict(sample_candle_data, timestamp)

        # Cache should be limited to 10
        assert len(predictor._prediction_cache) == 10

    @pytest.mark.asyncio
    async def test_get_last_prediction(self, mock_registry, mock_feature_engine, mock_model, sample_candle_data):
        """Test getting last prediction."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # No prediction yet
        assert predictor.get_last_prediction() is None

        # Load model and make prediction
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        prediction = await predictor.predict(sample_candle_data, 1700000000000)

        # Get last prediction
        last = predictor.get_last_prediction()
        assert last is not None
        assert last.timestamp == prediction.timestamp
        assert last.prediction == prediction.prediction

    def test_clear_cache(self, mock_registry, mock_feature_engine):
        """Test clearing prediction cache."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Manually add some cached predictions
        predictor._prediction_cache[1700000000000] = MagicMock()
        predictor._prediction_cache[1700000900000] = MagicMock()
        predictor._prediction_cache[1700001800000] = MagicMock()

        assert len(predictor._prediction_cache) == 3

        # Clear cache
        predictor.clear_cache()

        assert len(predictor._prediction_cache) == 0

    def test_get_model_info_not_loaded(self, mock_registry, mock_feature_engine):
        """Test getting model info when model not loaded."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        info = predictor.get_model_info()
        assert info is None

    @pytest.mark.asyncio
    async def test_get_model_info_loaded(self, mock_registry, mock_feature_engine, mock_model):
        """Test getting model info when model is loaded."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            with patch('joblib.load', return_value=mock_model):
                await predictor.load_model()

        info = predictor.get_model_info()
        assert info is not None
        assert info["model_id"] == "test_model_123"
        assert info["strategy_name"] == "test_strategy"
        assert info["model_type"] == "xgboost"
        assert info["validation_metrics"]["accuracy"] == 0.75

    def test_is_loaded(self, mock_registry, mock_feature_engine, mock_model):
        """Test is_loaded check."""
        predictor = ModelPredictor(
            model_id="test_model_123",
            registry=mock_registry,
            feature_engine=mock_feature_engine,
        )

        assert predictor.is_loaded() is False

        # Manually set model
        predictor.model = mock_model

        assert predictor.is_loaded() is True


class TestPrediction:
    """Test suite for Prediction dataclass."""

    def test_prediction_init(self):
        """Test Prediction initialization."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test_model",
            prediction=2,
            confidence=0.75,
            probabilities={'short': 0.15, 'neutral': 0.10, 'long': 0.75},
        )

        assert prediction.timestamp == 1700000000000
        assert prediction.model_id == "test_model"
        assert prediction.prediction == 2
        assert prediction.confidence == 0.75

    def test_get_signal_short(self):
        """Test get_signal for short prediction."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=0,
            confidence=0.75,
            probabilities={'short': 0.75, 'neutral': 0.15, 'long': 0.10},
        )

        assert prediction.get_signal() == "SHORT"

    def test_get_signal_neutral(self):
        """Test get_signal for neutral prediction."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=1,
            confidence=0.75,
            probabilities={'short': 0.15, 'neutral': 0.75, 'long': 0.10},
        )

        assert prediction.get_signal() == "NEUTRAL"

    def test_get_signal_long(self):
        """Test get_signal for long prediction."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=2,
            confidence=0.75,
            probabilities={'short': 0.15, 'neutral': 0.10, 'long': 0.75},
        )

        assert prediction.get_signal() == "LONG"

    def test_repr(self):
        """Test Prediction __repr__."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=2,
            confidence=0.75,
            probabilities={'short': 0.15, 'neutral': 0.10, 'long': 0.75},
        )

        repr_str = repr(prediction)
        assert "LONG" in repr_str
        assert "75.00%" in repr_str or "0.75" in repr_str
        assert "1700000000000" in repr_str
