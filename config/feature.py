"""Feature engineering configuration."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from config.base import BaseConfig


class IndicatorConfig(BaseConfig):
    """Configuration for a single indicator."""

    name: str = Field(description="Indicator name (e.g., 'ema_12')")
    type: str = Field(description="Indicator type (e.g., 'EMA', 'RSI', 'MACD')")
    params: dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")


class FeatureConfig(BaseConfig):
    """
    Configuration for feature engineering.

    Defines which indicators to compute and their parameters.
    Each strategy can have its own feature configuration.
    """

    # Flexible indicator list from YAML
    indicators: List[IndicatorConfig] = Field(
        description="List of indicators to compute",
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

        # Extract window sizes from indicator configs
        for ind in self.indicators:
            if "window" in ind.params:
                windows.append(ind.params["window"])
            if "slow" in ind.params:
                windows.append(ind.params["slow"])

        return max(windows) if windows else self.max_lookback
