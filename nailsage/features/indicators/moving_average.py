"""Moving average indicators."""

import pandas as pd

from nailsage.features.base import BaseIndicator


class SMA(BaseIndicator):
    """Simple Moving Average."""

    def __init__(self, window: int, price_column: str = "close"):
        """
        Initialize SMA indicator.

        Args:
            window: Number of periods for moving average
            price_column: Column to calculate SMA on (default: close)
        """
        super().__init__(name=f"sma_{window}", window=window, price_column=price_column)
        self.window = window
        self.price_column = price_column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with SMA column added
        """
        self.validate_dataframe(df)

        df = df.copy()
        df[self.name] = df[self.price_column].rolling(window=self.window, min_periods=self.window).mean()

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return [self.price_column]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window


class EMA(BaseIndicator):
    """Exponential Moving Average."""

    def __init__(self, window: int, price_column: str = "close"):
        """
        Initialize EMA indicator.

        Args:
            window: Number of periods for moving average
            price_column: Column to calculate EMA on (default: close)
        """
        super().__init__(name=f"ema_{window}", window=window, price_column=price_column)
        self.window = window
        self.price_column = price_column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with EMA column added
        """
        self.validate_dataframe(df)

        df = df.copy()
        df[self.name] = df[self.price_column].ewm(span=self.window, adjust=False, min_periods=self.window).mean()

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return [self.price_column]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window
