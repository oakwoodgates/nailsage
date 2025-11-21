"""Feature engineering engine for computing technical indicators."""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.feature import FeatureConfig
from features.base import BaseIndicator
from features.indicators.moving_average import SMA, EMA
from features.indicators.momentum import RSI, MACD, ROC
from features.indicators.volatility import BollingerBands, ATR
from features.indicators.volume import VolumeMA
from utils.logger import get_features_logger

logger = get_features_logger()


class FeatureEngine:
    """
    Feature engineering engine that computes technical indicators.

    Supports:
    - Dynamic computation based on configuration
    - Feature caching for performance
    - Lookback-aware calculations
    - Multiple indicators in parallel
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize FeatureEngine.

        Args:
            config: Feature configuration
        """
        self.config = config
        self.indicators: List[BaseIndicator] = []
        self._initialize_indicators()

        logger.info(
            f"Initialized FeatureEngine with {len(self.indicators)} indicators",
            extra_data={"cache_enabled": config.enable_cache},
        )

    def _initialize_indicators(self):
        """Initialize indicators based on configuration."""
        for ind_config in self.config.indicators:
            indicator = self._create_indicator_from_config(ind_config)
            if indicator:
                self.indicators.append(indicator)

    def _create_indicator_from_config(self, ind_config) -> Optional[BaseIndicator]:
        """
        Create indicator instance from config.

        Args:
            ind_config: IndicatorConfig object

        Returns:
            BaseIndicator instance or None if type not recognized
        """
        indicator_type = ind_config.type.upper()
        params = ind_config.params

        try:
            if indicator_type == "SMA":
                return SMA(**params)
            elif indicator_type == "EMA":
                return EMA(**params)
            elif indicator_type == "RSI":
                return RSI(**params)
            elif indicator_type == "MACD":
                return MACD(**params)
            elif indicator_type == "BOLLINGERBANDS":
                return BollingerBands(**params)
            elif indicator_type == "ATR":
                return ATR(**params)
            elif indicator_type == "VOLUMEMA":
                return VolumeMA(**params)
            elif indicator_type == "ROC":
                return ROC(**params)
            else:
                logger.warning(f"Unknown indicator type: {indicator_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create indicator {indicator_type}: {e}")
            return None

    def compute_features(
        self,
        df: pd.DataFrame,
        use_cache: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Compute all features for given DataFrame.

        Args:
            df: DataFrame with OHLCV data
            use_cache: Override cache setting (None = use config default)

        Returns:
            DataFrame with all feature columns added
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided to compute_features")
            return df

        use_cache = use_cache if use_cache is not None else self.config.enable_cache

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(df)
            cached_df = self._load_from_cache(cache_key)
            if cached_df is not None:
                logger.info("Loaded features from cache")
                return cached_df

        logger.info(f"Computing features for {len(df):,} rows")

        # Make a copy to avoid modifying original
        result_df = df.copy()

        # Compute each indicator
        for indicator in self.indicators:
            try:
                result_df = indicator.calculate(result_df)
            except Exception as e:
                logger.error(
                    f"Failed to compute indicator '{indicator.name}': {e}",
                    extra_data={"indicator": str(indicator)},
                )
                # Continue with other indicators
                continue

        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, result_df)

        feature_cols = [col for col in result_df.columns if col not in df.columns]
        logger.info(f"Computed {len(feature_cols)} feature columns")

        return result_df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names that will be generated.

        Returns:
            List of feature column names
        """
        # This is a simplified version - actual names depend on indicator implementation
        feature_names = []

        for indicator in self.indicators:
            # Most indicators use their name as the column prefix
            feature_names.append(indicator.name)

        return feature_names

    def get_max_lookback(self) -> int:
        """
        Get the maximum lookback period needed across all indicators.

        Returns:
            Maximum lookback period in bars
        """
        if not self.indicators:
            return self.config.max_lookback

        lookbacks = [ind.get_lookback_period() for ind in self.indicators]
        return max(lookbacks + [self.config.max_lookback])

    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """
        Generate cache key based on data and configuration.

        Args:
            df: DataFrame

        Returns:
            Cache key string
        """
        # Hash based on: data shape, first/last timestamp, config
        key_parts = [
            str(len(df)),
            str(df.index[0]) if len(df) > 0 else "",
            str(df.index[-1]) if len(df) > 0 else "",
            str(self.config.model_dump()),
        ]

        key_string = "_".join(key_parts)
        hash_key = hashlib.md5(key_string.encode()).hexdigest()

        return hash_key

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load features from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached DataFrame or None if not found
        """
        if not self.config.enable_cache:
            return None

        cache_file = self.config.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            return df
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """
        Save features to cache.

        Args:
            cache_key: Cache key
            df: DataFrame to cache
        """
        if not self.config.enable_cache:
            return

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear_cache(self):
        """Clear all cached features."""
        if not self.config.enable_cache:
            return

        cache_dir = Path(self.config.cache_dir)
        if not cache_dir.exists():
            return

        cache_files = list(cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {len(cache_files)} cached feature files")
