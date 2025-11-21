"""Momentum indicators (RSI, MACD, ROC)."""

import pandas as pd
import numpy as np

from features.base import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index."""

    def __init__(self, window: int = 14, price_column: str = "close"):
        """
        Initialize RSI indicator.

        Args:
            window: Number of periods for RSI calculation
            price_column: Column to calculate RSI on (default: close)
        """
        super().__init__(name=f"rsi_{window}", window=window, price_column=price_column)
        self.window = window
        self.price_column = price_column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with RSI column added
        """
        self.validate_dataframe(df)

        df = df.copy()

        # Calculate price changes
        delta = df[self.price_column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate average gain and loss
        avg_gain = gain.ewm(span=self.window, adjust=False, min_periods=self.window).mean()
        avg_loss = loss.ewm(span=self.window, adjust=False, min_periods=self.window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df[self.name] = rsi

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return [self.price_column]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window + 1  # +1 for diff()


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, price_column: str = "close"):
        """
        Initialize MACD indicator.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            price_column: Column to calculate MACD on (default: close)
        """
        super().__init__(
            name="macd",
            fast=fast,
            slow=slow,
            signal=signal,
            price_column=price_column,
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.price_column = price_column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD, Signal, and Histogram.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with MACD columns added
        """
        self.validate_dataframe(df)

        df = df.copy()

        # Calculate fast and slow EMAs
        ema_fast = df[self.price_column].ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        ema_slow = df[self.price_column].ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()

        # MACD line
        macd_line = ema_fast - ema_slow
        df["macd"] = macd_line

        # Signal line
        signal_line = macd_line.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()
        df["macd_signal"] = signal_line

        # Histogram
        df["macd_hist"] = macd_line - signal_line

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return [self.price_column]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.slow + self.signal


class ROC(BaseIndicator):
    """Rate of Change (Momentum)."""

    def __init__(self, window: int = 10, price_column: str = "close"):
        """
        Initialize ROC indicator.

        Args:
            window: Number of periods to look back
            price_column: Column to calculate ROC on (default: close)
        """
        super().__init__(name=f"roc_{window}", window=window, price_column=price_column)
        self.window = window
        self.price_column = price_column

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Rate of Change.

        ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with ROC column added
        """
        self.validate_dataframe(df)

        df = df.copy()

        price = df[self.price_column]
        price_n_ago = price.shift(self.window)

        df[self.name] = ((price - price_n_ago) / price_n_ago) * 100

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return [self.price_column]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window
