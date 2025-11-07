"""Volatility indicators (Bollinger Bands, ATR)."""

import pandas as pd
import numpy as np

from features.base import BaseIndicator


class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator."""

    def __init__(self, window: int = 20, num_std: float = 2.0, price_column: str = "close"):
        """
        Initialize Bollinger Bands indicator.

        Args:
            window: Number of periods for moving average
            num_std: Number of standard deviations for bands
            price_column: Column to calculate on (default: close)
        """
        super().__init__(
            name="bb",
            window=window,
            num_std=num_std,
            price_column=price_column,
        )
        self.window = window
        self.num_std = num_std
        self.price_column = price_column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Returns middle band (SMA), upper band, lower band, and bandwidth.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with Bollinger Band columns added
        """
        self.validate_dataframe(df)

        df = df.copy()

        # Middle band (SMA)
        sma = df[self.price_column].rolling(window=self.window, min_periods=self.window).mean()
        df["bb_middle"] = sma

        # Standard deviation
        std = df[self.price_column].rolling(window=self.window, min_periods=self.window).std()

        # Upper and lower bands
        df["bb_upper"] = sma + (std * self.num_std)
        df["bb_lower"] = sma - (std * self.num_std)

        # Bandwidth (useful for volatility analysis)
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # %B (position within bands)
        df["bb_pct"] = (df[self.price_column] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return [self.price_column]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window


class ATR(BaseIndicator):
    """Average True Range (volatility indicator)."""

    def __init__(self, window: int = 14):
        """
        Initialize ATR indicator.

        Args:
            window: Number of periods for ATR calculation
        """
        super().__init__(name=f"atr_{window}", window=window)
        self.window = window

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average True Range.

        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA of True Range

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with ATR column added
        """
        self.validate_dataframe(df)

        df = df.copy()

        # Calculate True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR (EMA of True Range)
        atr = true_range.ewm(span=self.window, adjust=False, min_periods=self.window).mean()
        df[self.name] = atr

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window + 1  # +1 for shift()
