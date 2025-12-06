"""Time series splitting with data leakage prevention.

This module provides time-series aware data splitting that prevents
data leakage by ensuring strict temporal ordering and lookback validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

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
        min_train_size: int = 1000,
        min_val_size: int = 500,
        expanding_window: bool = True,  # kept for backward compatibility
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
        self.min_train_size = min_train_size
        self.min_val_size = min_val_size
        self.expanding_window = expanding_window

        logger.info(
            "Initialized TimeSeriesSplitter",
            extra_data={
                "n_splits": n_splits,
                "test_size": test_size,
                "gap_bars": gap_bars,
                "min_val_size": min_val_size,
                "expanding_window": expanding_window,
            },
        )

    def split(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
        persist_path: str = None,
        load_existing: bool = False,
        min_val_size: int = 500,
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
        if load_existing and persist_path:
            persisted = Path(persist_path)
            if persisted.exists():
                import json
                data = json.loads(persisted.read_text())
                return [
                    TimeSeriesSplit(
                        train_start=pd.to_datetime(s["train_start"]),
                        train_end=pd.to_datetime(s["train_end"]),
                        val_start=pd.to_datetime(s["val_start"]),
                        val_end=pd.to_datetime(s["val_end"]),
                        split_index=s["split_index"],
                    )
                    for s in data
                ]

        if len(df) < self.min_train_size:
            raise ValueError(
                f"Insufficient data: {len(df)} rows < {self.min_train_size} minimum"
            )

        # Ensure data is sorted by timestamp
        df = df.sort_values(timestamp_column).reset_index(drop=True)

        n_samples = len(df)
        val_size = int(n_samples * self.test_size)
        val_size = max(val_size, min_val_size)

        if val_size < 10:
            raise ValueError(f"Validation size too small: {val_size} samples")

        sklearn_splitter = SklearnTimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=val_size,
        )

        splits = []
        for i, (train_idx, val_idx) in enumerate(sklearn_splitter.split(df)):
            # Apply gap by truncating tail of train
            if self.gap_bars > 0:
                if len(train_idx) <= self.gap_bars:
                    continue
                train_idx = train_idx[:-self.gap_bars]

            if len(train_idx) < self.min_train_size:
                continue

            train_start = df.iloc[train_idx[0]][timestamp_column]
            train_end = df.iloc[train_idx[-1]][timestamp_column]
            val_start = df.iloc[val_idx[0]][timestamp_column]
            val_end = df.iloc[val_idx[-1]][timestamp_column]

            split = TimeSeriesSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                split_index=i,
            )

            split.validate()
            splits.append(split)

            logger.info(
                f"Created split {i}",
                extra_data={
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "gap_bars": self.gap_bars,
                },
            )

        if len(splits) == 0:
            raise ValueError("Could not generate any valid splits")

        if persist_path and splits:
            import json
            payload = [
                {
                    "train_start": str(s.train_start),
                    "train_end": str(s.train_end),
                    "val_start": str(s.val_start),
                    "val_end": str(s.val_end),
                    "split_index": s.split_index,
                }
                for s in splits
            ]
            Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
            Path(persist_path).write_text(json.dumps(payload, indent=2))

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
        if lookback_bars <= 0:
            return True

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
