"""Feature engineering configuration."""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field

from config.base import BaseConfig


class FeatureConfig(BaseConfig):
    """
    Configuration for feature engineering.

    Defines which indicators to compute and their parameters.
    Each strategy can have its own feature configuration.
    """

    # Indicator parameters - Simple Moving Averages
    sma_windows: List[int] = Field(
        default=[5, 10, 20, 50, 100, 200],
        description="Windows for Simple Moving Average",
    )

    # Exponential Moving Averages
    ema_windows: List[int] = Field(
        default=[9, 12, 21, 26, 50],
        description="Windows for Exponential Moving Average",
    )

    # RSI
    rsi_window: int = Field(
        default=14,
        description="Window for RSI calculation",
        ge=2,
    )

    # MACD
    macd_fast: int = Field(default=12, description="MACD fast period", ge=2)
    macd_slow: int = Field(default=26, description="MACD slow period", ge=2)
    macd_signal: int = Field(default=9, description="MACD signal period", ge=2)

    # Bollinger Bands
    bb_window: int = Field(default=20, description="Bollinger Bands window", ge=2)
    bb_std: float = Field(
        default=2.0,
        description="Bollinger Bands standard deviations",
        ge=0.1,
    )

    # ATR (Average True Range)
    atr_window: int = Field(default=14, description="ATR window", ge=2)

    # Volume indicators
    volume_ma_window: int = Field(
        default=20,
        description="Volume moving average window",
        ge=2,
    )

    # Momentum indicators
    roc_window: int = Field(
        default=10,
        description="Rate of Change window",
        ge=1,
    )

    # Feature caching
    enable_cache: bool = Field(
        default=True,
        description="Enable feature caching to improve performance",
    )
    cache_dir: Path = Field(
        default=Path("features/cache"),
        description="Directory for feature cache",
    )

    # Lookback configuration
    max_lookback: int = Field(
        default=200,
        description="Maximum lookback window needed (for data loading)",
        ge=1,
    )

    # Custom indicators (for future extension)
    custom_indicators: Optional[Dict[str, Dict]] = Field(
        default=None,
        description="Custom indicator definitions",
    )

    def get_max_window(self) -> int:
        """
        Calculate the maximum window size needed across all indicators.

        This is used to determine the minimum data history required.

        Returns:
            Maximum window size
        """
        windows = []

        # Collect all window sizes
        if self.sma_windows:
            windows.extend(self.sma_windows)
        if self.ema_windows:
            windows.extend(self.ema_windows)

        windows.extend(
            [
                self.rsi_window,
                self.macd_slow,  # Slowest MACD component
                self.bb_window,
                self.atr_window,
                self.volume_ma_window,
                self.roc_window,
            ]
        )

        return max(windows) if windows else self.max_lookback
