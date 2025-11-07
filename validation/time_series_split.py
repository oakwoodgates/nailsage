"""Time series splitting with data leakage prevention.

This module provides time-series aware data splitting that prevents
data leakage by ensuring strict temporal ordering and lookback validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import numpy as np

from utils.logger import get_validation_logger

logger = get_validation_logger()


@dataclass
class TimeSeriesSplit:
    """
    Represents a single train/validation split with metadata.

    Attributes:
        train_start: Start timestamp of training data
        train_end: End timestamp of training data
        val_start: Start timestamp of validation data
        val_end: End timestamp of validation data
        split_index: Index of this split in the sequence
    """
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    split_index: int

    def validate(self) -> bool:
        """
        Validate that this split has no temporal leakage.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Train must come before validation
        if self.train_end >= self.val_start:
            raise ValueError(
                f"Data leakage detected: train_end ({self.train_end}) >= "
                f"val_start ({self.val_start})"
            )

        # Start must be before end for both
        if self.train_start >= self.train_end:
            raise ValueError(f"Invalid train period: start >= end")

        if self.val_start >= self.val_end:
            raise ValueError(f"Invalid validation period: start >= end")

        return True

    def get_gap_duration(self) -> timedelta:
        """Get the gap duration between train and validation."""
        return self.val_start - self.train_end

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Split {self.split_index}: "
            f"Train[{self.train_start} -> {self.train_end}] "
            f"Val[{self.val_start} -> {self.val_end}] "
            f"(gap: {self.get_gap_duration()})"
        )


class TimeSeriesSplitter:
    """
    Time series data splitter with walk-forward validation.

    Prevents data leakage by:
    1. Ensuring strict temporal ordering (train before validation)
    2. Adding optional gaps between train and validation
    3. Validating lookback windows don't cross boundaries
    4. Supporting expanding or rolling window strategies
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        gap_bars: int = 0,
        expanding_window: bool = True,
        min_train_size: int = 1000,
    ):
        """
        Initialize TimeSeriesSplitter.

        Args:
            n_splits: Number of train/validation splits
            test_size: Fraction of data to use for validation in each split
            gap_bars: Number of bars to skip between train and validation (prevents leakage)
            expanding_window: If True, training window expands; if False, uses rolling window
            min_train_size: Minimum number of training samples required
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_bars = gap_bars
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size

        logger.info(
            "Initialized TimeSeriesSplitter",
            extra_data={
                "n_splits": n_splits,
                "test_size": test_size,
                "gap_bars": gap_bars,
                "expanding_window": expanding_window,
            },
        )

    def split(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> List[TimeSeriesSplit]:
        """
        Generate train/validation splits.

        Args:
            df: DataFrame with time series data
            timestamp_column: Name of timestamp column

        Returns:
            List of TimeSeriesSplit objects

        Raises:
            ValueError: If data is insufficient or splits are invalid
        """
        if len(df) < self.min_train_size:
            raise ValueError(
                f"Insufficient data: {len(df)} rows < {self.min_train_size} minimum"
            )

        # Ensure data is sorted by timestamp
        df = df.sort_values(timestamp_column).reset_index(drop=True)

        n_samples = len(df)
        val_size = int(n_samples * self.test_size)

        if val_size < 10:
            raise ValueError(f"Validation size too small: {val_size} samples")

        splits = []

        # Calculate split points
        for i in range(self.n_splits):
            if self.expanding_window:
                # Expanding window: training size grows
                train_end_idx = int(n_samples * (1 - self.test_size * (self.n_splits - i) / self.n_splits))
            else:
                # Rolling window: training size stays constant
                train_size = n_samples - val_size * (self.n_splits - i)
                train_end_idx = train_size

            # Ensure minimum training size
            if train_end_idx < self.min_train_size:
                continue

            # Add gap
            val_start_idx = train_end_idx + self.gap_bars

            # Validation end
            val_end_idx = val_start_idx + val_size

            # Check we have enough data
            if val_end_idx > n_samples:
                break

            # Get timestamps
            train_start = df.iloc[0][timestamp_column]
            train_end = df.iloc[train_end_idx - 1][timestamp_column]
            val_start = df.iloc[val_start_idx][timestamp_column]
            val_end = df.iloc[val_end_idx - 1][timestamp_column]

            # Create split
            split = TimeSeriesSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                split_index=i,
            )

            # Validate split
            try:
                split.validate()
            except ValueError as e:
                logger.error(f"Invalid split {i}: {e}")
                raise

            splits.append(split)

            logger.info(
                f"Created split {i}",
                extra_data={
                    "train_samples": train_end_idx,
                    "val_samples": val_size,
                    "gap_bars": self.gap_bars,
                },
            )

        if len(splits) == 0:
            raise ValueError("Could not generate any valid splits")

        logger.info(f"Generated {len(splits)} valid splits")
        return splits

    def get_train_val_indices(
        self,
        df: pd.DataFrame,
        split: TimeSeriesSplit,
        timestamp_column: str = "timestamp",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get train and validation indices for a split.

        Args:
            df: DataFrame
            split: TimeSeriesSplit object
            timestamp_column: Name of timestamp column

        Returns:
            Tuple of (train_indices, val_indices)
        """
        timestamps = df[timestamp_column]

        # Get train indices
        train_mask = (timestamps >= split.train_start) & (timestamps <= split.train_end)
        train_idx = np.where(train_mask)[0]

        # Get validation indices
        val_mask = (timestamps >= split.val_start) & (timestamps <= split.val_end)
        val_idx = np.where(val_mask)[0]

        return train_idx, val_idx

    def validate_lookback(
        self,
        df: pd.DataFrame,
        split: TimeSeriesSplit,
        lookback_bars: int,
        timestamp_column: str = "timestamp",
    ) -> bool:
        """
        Validate that lookback window doesn't cross split boundary.

        This is critical: if we're computing features for the first validation
        bar and those features need 200 bars of history, we need to ensure
        those 200 bars are all in the training data.

        Args:
            df: DataFrame
            split: TimeSeriesSplit object
            lookback_bars: Number of bars needed for lookback
            timestamp_column: Name of timestamp column

        Returns:
            True if valid

        Raises:
            ValueError: If lookback validation fails
        """
        timestamps = df[timestamp_column]

        # Find index of first validation bar
        val_start_idx = df[timestamps == split.val_start].index[0]

        # Check if we have enough history
        if val_start_idx < lookback_bars:
            raise ValueError(
                f"Insufficient history for validation: "
                f"need {lookback_bars} bars, have {val_start_idx}"
            )

        # Check that lookback doesn't include validation data
        lookback_start_idx = val_start_idx - lookback_bars
        lookback_start_time = df.iloc[lookback_start_idx][timestamp_column]

        if lookback_start_time > split.train_end:
            raise ValueError(
                f"Lookback window crosses split boundary: "
                f"lookback_start ({lookback_start_time}) > train_end ({split.train_end})"
            )

        logger.info(
            "Lookback validation passed",
            extra_data={
                "lookback_bars": lookback_bars,
                "val_start_idx": val_start_idx,
                "lookback_start_time": str(lookback_start_time),
            },
        )

        return True
