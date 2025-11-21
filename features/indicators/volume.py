"""Volume-based indicators."""

import pandas as pd

from features.base import BaseIndicator


class VolumeMA(BaseIndicator):
    """Volume Moving Average."""

    def __init__(self, window: int = 20):
        """
        Initialize Volume MA indicator.

        Args:
            window: Number of periods for moving average
        """
        super().__init__(name=f"volume_ma_{window}", window=window)
        self.window = window

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Moving Average and Volume Ratio.

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with Volume MA columns added
        """
        self.validate_dataframe(df)

        df = df.copy()

        # Volume moving average
        volume_ma = df["volume"].rolling(window=self.window, min_periods=self.window).mean()
        df[self.name] = volume_ma

        # Volume ratio (current volume / average volume)
        df[f"volume_ratio_{self.window}"] = df["volume"] / volume_ma

        return df

    def get_required_columns(self) -> list[str]:
        """Get required columns."""
        return ["volume"]

    def get_lookback_period(self) -> int:
        """Get lookback period."""
        return self.window
