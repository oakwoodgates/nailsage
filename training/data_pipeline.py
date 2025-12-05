"""Data pipeline for loading, preparing, and feature engineering."""

import logging
from pathlib import Path
from typing import Tuple, List

import pandas as pd

from config.strategy import StrategyConfig
from data.loader import DataLoader
from features.engine import FeatureEngine
from targets.classification import create_3class_target, create_binary_target
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

        logger.info("Initialized DataPipeline")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training.

        Returns:
            Tuple of (features_df, targets_series) ready for training
        """
        # Load raw data
        df = self._load_raw_data()

        # Create target variable
        target_series = self._create_target_variable(df)

        # Initialize and compute features
        feature_engine = FeatureEngine(self.config.features)
        df_features = self._compute_features(df, feature_engine)

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

    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for classification."""
        num_classes = self.config.target.classes or 3
        logger.info(f"Creating {num_classes}-class target variable...")

        if num_classes == 2:
            target_series = create_binary_target(
                df=df,
                lookahead_bars=self.config.target.lookahead_bars,
                threshold_pct=self.config.target.threshold_pct
            )
        else:
            target_series = create_3class_target(
                df=df,
                lookahead_bars=self.config.target.lookahead_bars,
                threshold_pct=self.config.target.threshold_pct
            )

        return target_series

    def _compute_features(self, df: pd.DataFrame, feature_engine: FeatureEngine) -> pd.DataFrame:
        """Compute features using the feature engine."""
        logger.info("Initializing feature engine...")
        logger.info("Computing features...")

        df_features = feature_engine.compute_features(df)

        return df_features

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
