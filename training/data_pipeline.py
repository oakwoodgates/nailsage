"""Data pipeline for loading, preparing, and feature engineering."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd

from config.strategy import StrategyConfig
from data.loader import DataLoader
from features.engine import FeatureEngine
from training.targets import create_3class_target, create_binary_target
from utils.logger import get_training_logger

logger = get_training_logger()


class DataPipeline:
    """
    Handles data loading, preparation, and feature engineering for training.

    This class encapsulates all data-related operations:
    - Loading raw data from files
    - Data resampling and cleaning
    - Target variable creation
    - Feature computation and alignment
    - Data quality validation

    Attributes:
        config: Strategy configuration
        ohlcv_columns: Standard OHLCV column names
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize data pipeline.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.ohlcv_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']
        self.cache_enabled = getattr(config, "feature_cache_enabled", False)
        self.cache_dir = Path("data/processed/cache")

        logger.info("Initialized DataPipeline")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training.

        Returns:
            Tuple of (features_df, targets_series) ready for training
        """
        # Load raw data
        df = self._load_raw_data()
        self._validate_raw_schema(df)

        # Create target variable
        target_series = self._create_target_variable(df)

        # Initialize and compute features
        feature_engine = FeatureEngine(self.config.features)
        df_features = self._compute_features(df, feature_engine)
        self._validate_feature_schema(df_features)
        self._validate_numeric_finite(df_features)

        # Align features and targets
        df_clean, target_clean = self._align_features_and_targets(
            df_features, target_series
        )

        logger.info(f"Data preparation complete: {len(df_clean):,} samples")

        return df_clean, target_clean

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw OHLCV data from file."""
        logger.info(f"Loading data from {self.config.data_source}")

        data_path = Path("data/raw") / self.config.data_source
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load parquet file directly (matching original implementation)
        df = pd.read_parquet(data_path)

        logger.info(f"Loaded {len(df):,} rows")

        # Rename 'time' to 'timestamp' for consistency
        if 'time' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'time': 'timestamp'})

        # Resample if needed
        if self.config.resample_interval:
            logger.info(f"Resampling to {self.config.resample_interval}")
            df = self._resample_data(df, self.config.resample_interval)

        return df

    def _resample_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample data to specified interval."""
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')

        # Resample OHLCV data
        resampled = df_indexed.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'num_trades': 'sum',
        }).dropna()

        # Reset index to get timestamp back as column
        resampled = resampled.reset_index()

        logger.info(f"After resampling: {len(resampled):,} rows")

        return resampled

    def _validate_raw_schema(self, df: pd.DataFrame) -> None:
        """Validate that required OHLCV columns exist."""
        missing = [c for c in self.ohlcv_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        # Ensure timestamp is tz-naive
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("timestamp column must be datetime-like")
        if df['timestamp'].dt.tz is not None:
            raise ValueError("timestamp column must be timezone-naive")

    def _validate_feature_schema(self, df_features: pd.DataFrame) -> None:
        """Ensure timestamp is present and no duplicate columns."""
        if "timestamp" not in df_features.columns:
            raise ValueError("Feature DataFrame must contain 'timestamp' column.")
        if df_features.columns.duplicated().any():
            raise ValueError("Feature DataFrame contains duplicate column names.")

    def _validate_numeric_finite(self, df_features: pd.DataFrame) -> None:
        """Ensure numeric feature columns are finite (no inf/nan after dropna)."""
        numeric_cols = [c for c in df_features.columns if c not in self.ohlcv_columns and pd.api.types.is_numeric_dtype(df_features[c])]
        if not numeric_cols:
            return
        bad_mask = ~np.isfinite(df_features[numeric_cols]).all(axis=1)
        if bad_mask.any():
            raise ValueError(f"Non-finite values detected in feature columns: {bad_mask.sum()} rows")

    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable based on configured target type."""
        target_type = getattr(self.config.target, "type", None)
        num_classes = self.config.target.classes or 3

        # Normalize target type for compatibility
        target_type_normalized = (target_type or "").lower()
        if target_type_normalized in ("", "3class", "classification_3class", "classification"):
            target_type_normalized = "3class"
        elif target_type_normalized in ("binary", "2class", "classification_2class"):
            target_type_normalized = "binary"

        logger.info(f"Creating target variable (type={target_type_normalized or '3class'}, classes={num_classes})")

        if target_type_normalized == "binary" or num_classes == 2:
            return create_binary_target(
                df=df,
                lookahead_bars=self.config.target.lookahead_bars,
                threshold_pct=self.config.target.threshold_pct
            )
        if target_type_normalized == "3class" or num_classes == 3:
            return create_3class_target(
                df=df,
                lookahead_bars=self.config.target.lookahead_bars,
                threshold_pct=self.config.target.threshold_pct
            )

        if target_type_normalized == "regression":
            raise ValueError("Regression targets are not yet supported in this pipeline.")

        raise ValueError(f"Unsupported target type '{target_type}'. Provide 'binary', '3class', or 'regression'.")

    def _compute_features(self, df: pd.DataFrame, feature_engine: FeatureEngine) -> pd.DataFrame:
        """Compute features using the feature engine."""
        logger.info("Initializing feature engine...")
        logger.info("Computing features...")

        # Optional caching to speed up iterative runs
        cache_key = self._build_cache_key(df) if self.cache_enabled else None
        if cache_key:
            cached = self._maybe_load_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded features from cache: {cache_key}")
                return cached

        df_features = feature_engine.compute_features(df)

        if cache_key:
            self._write_cache(cache_key, df_features)

        return df_features

    def _build_cache_key(self, df: pd.DataFrame) -> str:
        """Build a deterministic cache key from data source and feature config."""
        payload = {
            "data_source": self.config.data_source,
            "resample_interval": self.config.resample_interval,
            "feature_config": getattr(self.config.features, "model_dump", lambda: self.config.features)(),
            "row_count": len(df),
            "source_checksum": self._source_checksum(),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _maybe_load_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached features if available."""
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def _write_cache(self, cache_key: str, df_features: pd.DataFrame) -> None:
        """Persist features cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        df_features.to_parquet(cache_path)
        logger.info(f"Wrote feature cache: {cache_path}")

    def _source_checksum(self) -> str:
        """Compute checksum of source file using size and mtime for invalidation."""
        data_path = Path("data/raw") / self.config.data_source
        if not data_path.exists():
            return "missing"
        stat = data_path.stat()
        payload = f"{stat.st_size}-{int(stat.st_mtime)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _align_features_and_targets(
        self,
        df_features: pd.DataFrame,
        target_series: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and targets, handling missing data."""
        # Drop rows with NaN values
        df_clean = df_features.dropna()

        # Align targets with features
        target_clean = target_series.reindex(df_clean.index).dropna()
        df_clean = df_clean.reindex(target_clean.index)

        logger.info(f"After cleaning: {len(df_clean):,} samples")
        logger.info(f"Target distribution: {target_clean.value_counts().to_dict()}")

        return df_clean, target_clean

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature column names, excluding OHLCV columns.

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        feature_cols = [col for col in df.columns if col not in self.ohlcv_columns]

        logger.info(f"Using {len(feature_cols)} feature columns")

        return feature_cols

    def prepare_validation_data(
        self,
        df_clean: pd.DataFrame,
        target_clean: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for validation period.

        Args:
            df_clean: Cleaned feature DataFrame
            target_clean: Cleaned target series

        Returns:
            Tuple of (validation_features, validation_targets)
        """
        # Filter validation period
        val_mask = (
            (df_clean['timestamp'] >= self.config.validation_start) &
            (df_clean['timestamp'] <= self.config.validation_end)
        )
        df_val = df_clean[val_mask]
        target_val = target_clean[val_mask]

        logger.info(f"Validation period: {self.config.validation_start} to {self.config.validation_end}")
        logger.info(f"Validation samples: {len(df_val):,}")

        return df_val, target_val
