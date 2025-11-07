"""Base classes for feature engineering."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseIndicator(ABC):
    """
    Base class for technical indicators.

    All indicators should inherit from this class and implement
    the calculate() method.
    """

    def __init__(self, name: str, **params):
        """
        Initialize indicator.

        Args:
            name: Indicator name
            **params: Indicator-specific parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator columns added
        """
        pass

    def get_required_columns(self) -> List[str]:
        """
        Get list of required columns for this indicator.

        Returns:
            List of column names needed
        """
        # Default: requires close price
        return ["close"]

    def get_lookback_period(self) -> int:
        """
        Get the lookback period needed for this indicator.

        This is the minimum number of bars required to calculate
        the indicator.

        Returns:
            Lookback period in bars
        """
        # Default: no specific lookback
        return 0

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required = self.get_required_columns()
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Indicator '{self.name}' requires columns: {missing}")

        return True

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}(name='{self.name}', {params_str})"
