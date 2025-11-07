"""Strategy configuration."""

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator

from config.base import BaseConfig
from config.data import DataConfig
from config.feature import FeatureConfig


class TargetType(str, Enum):
    """Type of prediction target."""

    CLASSIFICATION_3CLASS = "classification_3class"  # long/short/neutral
    CLASSIFICATION_2CLASS = "classification_2class"  # long/short
    REGRESSION = "regression"  # continuous return prediction


class TimeFrame(str, Enum):
    """Supported timeframes."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class StrategyConfig(BaseConfig):
    """
    Configuration for a trading strategy.

    Combines data, features, and strategy-specific parameters.
    Each strategy (short-term, long-term, etc.) will have its own config.
    """

    # Strategy identification
    name: str = Field(
        description="Strategy name (e.g., 'btc_spot_short_term')",
    )
    description: Optional[str] = Field(
        default=None,
        description="Strategy description",
    )

    # Data configuration
    data: DataConfig = Field(
        description="Data loading and validation configuration",
    )

    # Feature configuration
    features: FeatureConfig = Field(
        description="Feature engineering configuration",
    )

    # Timeframe
    timeframe: TimeFrame = Field(
        description="Trading timeframe (e.g., '15m', '1h', '1d')",
    )

    # Target definition
    target_type: TargetType = Field(
        default=TargetType.CLASSIFICATION_3CLASS,
        description="Type of prediction target",
    )
    target_horizon_bars: int = Field(
        default=1,
        description="Number of bars ahead to predict",
        ge=1,
    )
    target_threshold: float = Field(
        default=0.005,  # 0.5%
        description="Threshold for classification (e.g., 0.5% move)",
        gt=0.0,
    )
    neutral_zone: Optional[float] = Field(
        default=None,
        description="Neutral zone threshold (for 3-class). If None, uses target_threshold",
        gt=0.0,
    )

    # Model parameters
    model_type: str = Field(
        default="xgboost",
        description="Model type (xgboost, lightgbm, random_forest)",
    )
    model_params: Optional[dict] = Field(
        default=None,
        description="Model-specific hyperparameters",
    )

    # Training parameters
    train_test_split_ratio: float = Field(
        default=0.7,
        description="Ratio of data to use for training vs testing",
        gt=0.0,
        lt=1.0,
    )
    validation_windows: int = Field(
        default=5,
        description="Number of walk-forward validation windows",
        ge=1,
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate strategy name format."""
        if not v or len(v) < 3:
            raise ValueError("Strategy name must be at least 3 characters")
        # Convert to lowercase with underscores
        return v.lower().replace("-", "_").replace(" ", "_")

    @field_validator("neutral_zone")
    @classmethod
    def validate_neutral_zone(cls, v: Optional[float], info) -> Optional[float]:
        """Validate neutral zone is smaller than target threshold."""
        if v is not None:
            target_threshold = info.data.get("target_threshold")
            if target_threshold and v > target_threshold:
                raise ValueError("neutral_zone must be <= target_threshold")
        return v

    def get_neutral_threshold(self) -> float:
        """
        Get the neutral zone threshold.

        Returns target_threshold if neutral_zone is not set.
        """
        return self.neutral_zone if self.neutral_zone is not None else self.target_threshold

    def get_model_params_with_defaults(self) -> dict:
        """
        Get model parameters with sensible defaults for the model type.

        Returns:
            Dictionary of model parameters
        """
        defaults = {
            "xgboost": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "objective": "multi:softprob" if "classification" in self.target_type else "reg:squarederror",
                "random_state": 42,
            },
            "lightgbm": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "objective": "multiclass" if "classification" in self.target_type else "regression",
                "random_state": 42,
            },
            "random_forest": {
                "max_depth": 10,
                "n_estimators": 100,
                "random_state": 42,
            },
        }

        base_params = defaults.get(self.model_type, {})
        if self.model_params:
            base_params.update(self.model_params)

        return base_params
