"""Feature schema for validating model inputs at inference time.

This module ensures that features computed at inference exactly match
what the model was trained on, preventing shape mismatches and silent errors.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class FeatureSchema:
    """
    Schema defining expected features for a model.

    This is saved with model metadata during training and used to validate
    features at inference time.

    Attributes:
        feature_names: Ordered list of feature column names expected by model
        include_ohlcv: Whether OHLCV columns should be included in features
        include_volume: Whether volume column should be included (if include_ohlcv=True)
        include_num_trades: Whether num_trades should be included
        ohlcv_columns: The exact OHLCV column names used during training
    """

    feature_names: List[str]
    include_ohlcv: bool = False
    include_volume: bool = True
    include_num_trades: bool = False
    ohlcv_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame has required features.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If features don't match schema
        """
        # Check all required features exist
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}. "
                f"Expected: {self.feature_names}"
            )

        # Check for extra columns (warning only)
        extra = set(df.columns) - set(self.feature_names)
        if extra:
            # Filter out common non-feature columns
            non_feature_cols = {'timestamp', 'datetime', 'symbol'}
            extra = extra - non_feature_cols
            if extra:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"DataFrame has extra columns not in schema: {sorted(extra)}. "
                    f"These will be ignored."
                )

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from DataFrame in correct order.

        Args:
            df: DataFrame with computed features

        Returns:
            DataFrame with only required features in correct order

        Raises:
            ValueError: If required features are missing
        """
        # Validate first
        self.validate(df)

        # Extract in correct order
        return df[self.feature_names].copy()

    def check_nan_values(self, df: pd.DataFrame) -> List[str]:
        """
        Check for NaN values in required features.

        Args:
            df: DataFrame to check

        Returns:
            List of feature names with NaN values
        """
        nan_features = []
        for col in self.feature_names:
            if col in df.columns and df[col].isna().any():
                nan_features.append(col)
        return nan_features

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        include_ohlcv: bool = False,
        include_volume: bool = True,
        include_num_trades: bool = False,
        exclude_columns: Optional[List[str]] = None,
    ) -> "FeatureSchema":
        """
        Create FeatureSchema from a DataFrame.

        Args:
            df: DataFrame with features
            include_ohlcv: Include OHLCV columns in features
            include_volume: Include volume (if include_ohlcv=True)
            include_num_trades: Include num_trades column
            exclude_columns: Additional columns to exclude

        Returns:
            FeatureSchema instance
        """
        if exclude_columns is None:
            exclude_columns = []

        # Standard columns to exclude
        standard_exclude = ['timestamp', 'datetime', 'symbol']
        all_exclude = set(standard_exclude + exclude_columns)

        # Determine OHLCV columns
        ohlcv_base = ['open', 'high', 'low', 'close']
        ohlcv_columns = ohlcv_base.copy()
        if include_volume:
            ohlcv_columns.append('volume')

        # Build feature list
        feature_names = []

        for col in df.columns:
            # Skip excluded columns
            if col in all_exclude:
                continue

            # Handle OHLCV
            if col in ohlcv_base:
                if include_ohlcv:
                    feature_names.append(col)
            elif col == 'volume':
                if include_ohlcv and include_volume:
                    feature_names.append(col)
            elif col == 'num_trades':
                if include_num_trades:
                    feature_names.append(col)
            else:
                # All other columns are features (indicators)
                feature_names.append(col)

        return cls(
            feature_names=feature_names,
            include_ohlcv=include_ohlcv,
            include_volume=include_volume,
            include_num_trades=include_num_trades,
            ohlcv_columns=ohlcv_columns,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "feature_names": self.feature_names,
            "include_ohlcv": self.include_ohlcv,
            "include_volume": self.include_volume,
            "include_num_trades": self.include_num_trades,
            "ohlcv_columns": self.ohlcv_columns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        """Create from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"FeatureSchema("
            f"n_features={len(self.feature_names)}, "
            f"include_ohlcv={self.include_ohlcv})"
        )
