"""Model predictor for live inference.

This module provides async prediction on streaming candle data by:
- Loading trained models from registry
- Computing features on sliding windows
- Running inference asynchronously (CPU-bound work)
- Caching predictions to avoid redundant computation

Example usage:
    # Initialize predictor
    predictor = ModelPredictor(
        model_id="2e30bea4e8f93845_20251108_153045_a3f9c2",
        registry=ModelRegistry(),
        feature_engine=FeatureEngine(config),
    )

    # Load model
    await predictor.load_model()

    # Run prediction on candle data
    candle_df = buffer.to_dataframe(n=500)
    prediction = await predictor.predict(candle_df, current_timestamp)
"""

import asyncio
import joblib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from features.engine import FeatureEngine
from models.registry import ModelRegistry
from models.metadata import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Prediction result from model.

    Attributes:
        timestamp: Unix milliseconds of the candle that was predicted
        datetime: Python datetime of the candle
        model_id: Model identifier from registry
        prediction: Predicted class (0=short, 1=neutral, 2=long)
        confidence: Maximum probability across all classes
        probabilities: Probability for each class
    """

    timestamp: int
    datetime: datetime
    model_id: str
    prediction: int
    confidence: float
    probabilities: Dict[str, float]

    def get_signal(self) -> str:
        """Get signal as string (SHORT, NEUTRAL, LONG)."""
        return ["SHORT", "NEUTRAL", "LONG"][self.prediction]

    def __repr__(self) -> str:
        return (
            f"Prediction(timestamp={self.timestamp}, "
            f"signal={self.get_signal()}, "
            f"confidence={self.confidence:.2%})"
        )


class ModelPredictor:
    """
    Async predictor for trained models.

    This class handles:
    - Model loading from registry
    - Feature computation on streaming data
    - Async inference (CPU-bound work in thread pool)
    - Prediction caching

    Attributes:
        model_id: Model identifier from registry
        registry: ModelRegistry instance
        feature_engine: FeatureEngine for computing indicators
        model: Loaded sklearn/xgboost/lightgbm model (None until load_model())
        metadata: Model metadata from registry
    """

    def __init__(
        self,
        model_id: str,
        registry: ModelRegistry,
        feature_engine: FeatureEngine,
    ):
        """
        Initialize predictor.

        Args:
            model_id: Model ID from registry
            registry: ModelRegistry instance
            feature_engine: FeatureEngine for computing indicators
        """
        self.model_id = model_id
        self.registry = registry
        self.feature_engine = feature_engine
        self.model = None
        self.metadata: Optional[ModelMetadata] = None

        # Caching
        self._last_prediction: Optional[Prediction] = None
        self._prediction_cache: Dict[int, Prediction] = {}
        self._max_cache_size = 1000

        logger.info(f"Initialized ModelPredictor for model {model_id}")

    async def load_model(self) -> None:
        """
        Load model from registry.

        This is an async operation to avoid blocking the event loop
        during potentially slow I/O operations.

        Raises:
            ValueError: If model not found in registry
            FileNotFoundError: If model artifact file not found
        """
        logger.info(f"Loading model {self.model_id} from registry...")

        # Load metadata from registry (I/O operation, run in thread)
        self.metadata = await asyncio.to_thread(
            self.registry.get_model,
            self.model_id
        )

        if self.metadata is None:
            raise ValueError(f"Model {self.model_id} not found in registry")

        # Check if artifact file exists
        artifact_path = Path(self.metadata.model_artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}"
            )

        # Load model artifact (I/O operation, run in thread)
        logger.info(f"Loading model artifact from {artifact_path}")
        self.model = await asyncio.to_thread(
            joblib.load,
            artifact_path
        )

        logger.info(
            f"Model {self.model_id} loaded successfully",
            extra={
                "model_type": self.metadata.model_type,
                "strategy": self.metadata.strategy_name,
                "trained_at": self.metadata.trained_at,
            }
        )

    async def predict(
        self,
        candle_df: pd.DataFrame,
        current_timestamp: int,
    ) -> Prediction:
        """
        Run prediction on candle data.

        Args:
            candle_df: DataFrame with OHLCV data (index is datetime)
            current_timestamp: Current candle timestamp (Unix ms)

        Returns:
            Prediction with class and confidence

        Raises:
            ValueError: If model not loaded or insufficient data
        """
        if self.model is None:
            raise ValueError("Model not loaded - call load_model() first")

        # Check cache
        if current_timestamp in self._prediction_cache:
            logger.debug(f"Using cached prediction for timestamp {current_timestamp}")
            return self._prediction_cache[current_timestamp]

        # Validate input data
        if len(candle_df) == 0:
            raise ValueError("Empty candle DataFrame provided")

        # Check if we have enough data for indicators
        max_lookback = self.feature_engine.get_max_lookback()
        if len(candle_df) < max_lookback:
            raise ValueError(
                f"Insufficient data: need at least {max_lookback} candles, "
                f"got {len(candle_df)}"
            )

        # Compute features (CPU-bound, run in thread)
        logger.debug(f"Computing features for {len(candle_df)} candles")
        feature_df = await asyncio.to_thread(
            self.feature_engine.compute_features,
            candle_df
        )

        # Get latest features (last row)
        if len(feature_df) == 0:
            raise ValueError("No features computed - check candle data")

        # Get all columns except timestamp (model expects OHLCV + computed features)
        # The model was trained on: open, high, low, close, volume + all indicators
        exclude_cols = ['timestamp']
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]

        if not feature_cols:
            raise ValueError("No feature columns computed - check feature engine")

        # Extract features for prediction (last row only)
        latest_features = feature_df.iloc[-1:][feature_cols]

        # Check for NaN values
        if latest_features.isna().any().any():
            nan_cols = latest_features.columns[latest_features.isna().any()].tolist()
            logger.warning(
                f"NaN values in features: {nan_cols}. "
                f"Need more data for indicators."
            )
            raise ValueError(
                "Insufficient data for feature computation (NaN values present)"
            )

        # Run inference (CPU-bound, run in thread)
        logger.debug(f"Running inference on {len(latest_features.columns)} features")
        probabilities = await asyncio.to_thread(
            self.model.predict_proba,
            latest_features
        )

        # Extract prediction and confidence
        class_probs = probabilities[0]  # Get first (and only) row
        predicted_class = int(np.argmax(class_probs))
        confidence = float(np.max(class_probs))

        # Create prediction object
        prediction = Prediction(
            timestamp=current_timestamp,
            datetime=pd.to_datetime(current_timestamp, unit='ms'),
            model_id=self.model_id,
            prediction=predicted_class,
            confidence=confidence,
            probabilities={
                'short': float(class_probs[0]),
                'neutral': float(class_probs[1]),
                'long': float(class_probs[2]),
            }
        )

        # Cache prediction
        self._cache_prediction(current_timestamp, prediction)
        self._last_prediction = prediction

        logger.info(
            f"Prediction: {prediction.get_signal()} "
            f"(confidence: {confidence:.2%})"
        )

        return prediction

    def _cache_prediction(self, timestamp: int, prediction: Prediction) -> None:
        """
        Cache a prediction.

        Args:
            timestamp: Timestamp key
            prediction: Prediction to cache
        """
        self._prediction_cache[timestamp] = prediction

        # Limit cache size (remove oldest if over limit)
        if len(self._prediction_cache) > self._max_cache_size:
            oldest_timestamp = min(self._prediction_cache.keys())
            del self._prediction_cache[oldest_timestamp]
            logger.debug(f"Removed oldest cached prediction (timestamp: {oldest_timestamp})")

    def get_last_prediction(self) -> Optional[Prediction]:
        """
        Get last prediction made (useful for monitoring).

        Returns:
            Last Prediction or None if no predictions made yet
        """
        return self._last_prediction

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        cache_size = len(self._prediction_cache)
        self._prediction_cache.clear()
        logger.info(f"Cleared {cache_size} cached predictions")

    def get_model_info(self) -> Optional[Dict]:
        """
        Get model metadata information.

        Returns:
            Dict with model info or None if model not loaded
        """
        if self.metadata is None:
            return None

        return {
            "model_id": self.metadata.model_id,
            "strategy_name": self.metadata.strategy_name,
            "strategy_timeframe": self.metadata.strategy_timeframe,
            "model_type": self.metadata.model_type,
            "version": self.metadata.version,
            "trained_at": self.metadata.trained_at,
            "validation_metrics": self.metadata.validation_metrics,
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
