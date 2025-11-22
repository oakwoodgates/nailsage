"""Strategy configuration."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, field_validator, model_validator

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


class DataSection(BaseConfig):
    """Data section of strategy config."""

    source_file: str = Field(description="Path to source data file")
    resample_interval: Optional[str] = Field(default=None, description="Resample interval (e.g., '15min')")
    train_start: str = Field(description="Training start date (YYYY-MM-DD)")
    train_end: str = Field(description="Training end date (YYYY-MM-DD)")
    validation_start: str = Field(description="Validation start date (YYYY-MM-DD)")
    validation_end: str = Field(description="Validation end date (YYYY-MM-DD)")


class TargetSection(BaseConfig):
    """Target section of strategy config."""

    type: str = Field(description="Target type (classification/regression)")
    classes: Optional[int] = Field(default=None, description="Number of classes for classification")
    lookahead_bars: int = Field(default=1, description="Bars to look ahead", ge=1)
    threshold_pct: float = Field(default=0.5, description="Threshold percentage", gt=0.0)
    class_weights: Optional[dict[int, float]] = Field(
        default=None,
        description="Class weights for imbalanced classification (e.g., {0: 5.0, 1: 1.0, 2: 5.0})"
    )
    confidence_threshold: float = Field(
        default=0.0,
        description="Minimum prediction confidence to generate signal (0.0-1.0). 0 = no filtering.",
        ge=0.0,
        le=1.0
    )


class ModelSection(BaseConfig):
    """Model section of strategy config."""

    type: str = Field(description="Model type (xgboost, lightgbm, etc.)")
    params: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")


class ValidationSection(BaseConfig):
    """Validation section of strategy config."""

    method: str = Field(default="walk_forward", description="Validation method")
    n_splits: int = Field(default=4, description="Number of splits", ge=2)
    expanding_window: bool = Field(default=True, description="Use expanding window")
    gap_bars: int = Field(default=0, description="Gap bars between splits", ge=0)


class BacktestSection(BaseConfig):
    """Backtest section of strategy config."""

    transaction_cost_pct: float = Field(default=0.04, description="Transaction cost %", ge=0.0)
    slippage_bps: int = Field(default=2, description="Slippage in basis points", ge=0)
    leverage: int = Field(default=1, description="Leverage", ge=1)
    capital: float = Field(default=10000, description="Starting capital", gt=0.0)
    min_bars_between_trades: int = Field(
        default=0,
        description="Minimum bars between trades (cooldown). 0 = no cooldown.",
        ge=0
    )


class RiskSection(BaseConfig):
    """Risk section of strategy config."""

    max_position_size_pct: float = Field(default=100, description="Max position size %", gt=0.0)
    stop_loss_pct: float = Field(default=2.0, description="Stop loss %", gt=0.0)
    take_profit_pct: float = Field(default=3.0, description="Take profit %", gt=0.0)


class StrategyConfig(BaseConfig):
    """
    Configuration for a trading strategy.

    Combines data, features, and strategy-specific parameters.
    Each strategy (short-term, long-term, etc.) will have its own config.
    """

    # Strategy identification
    strategy_name: str = Field(
        description="Strategy name (e.g., 'momentum_classifier')",
    )
    version: str = Field(
        description="Strategy version (e.g., 'v1')",
    )
    strategy_timeframe: str = Field(
        description="Strategy timeframe category (short_term, medium_term, long_term)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Strategy description",
    )

    # Main sections
    data: DataSection = Field(
        description="Data configuration",
    )
    features: FeatureConfig = Field(
        description="Feature engineering configuration",
    )
    target: TargetSection = Field(
        description="Target variable configuration",
    )
    model: ModelSection = Field(
        description="Model configuration",
    )
    validation: Optional[ValidationSection] = Field(
        default=None,
        description="Validation configuration",
    )
    backtest: Optional[BacktestSection] = Field(
        default=None,
        description="Backtest configuration",
    )
    risk: Optional[RiskSection] = Field(
        default=None,
        description="Risk management configuration",
    )

    @field_validator("strategy_name")
    @classmethod
    def validate_strategy_name(cls, v: str) -> str:
        """Validate strategy name format."""
        if not v or len(v) < 3:
            raise ValueError("Strategy name must be at least 3 characters")
        # Convert to lowercase with underscores
        return v.lower().replace("-", "_").replace(" ", "_")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not v.startswith("v"):
            return f"v{v}"
        return v

    # Helper properties for accessing nested config
    @property
    def data_source(self) -> str:
        """Get data source file path."""
        return self.data.source_file

    @property
    def resample_interval(self) -> Optional[str]:
        """Get resample interval."""
        return self.data.resample_interval

    @property
    def train_start(self) -> str:
        """Get training start date."""
        return self.data.train_start

    @property
    def train_end(self) -> str:
        """Get training end date."""
        return self.data.train_end

    @property
    def validation_start(self) -> str:
        """Get validation start date."""
        return self.data.validation_start

    @property
    def validation_end(self) -> str:
        """Get validation end date."""
        return self.data.validation_end

    @property
    def target_lookahead_bars(self) -> int:
        """Get target lookahead bars."""
        return self.target.lookahead_bars

    @property
    def target_threshold_pct(self) -> float:
        """Get target threshold percentage."""
        return self.target.threshold_pct

    @property
    def model_type_str(self) -> str:
        """Get model type as string."""
        return self.model.type

    @property
    def model_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return self.model.params

    @property
    def data_config(self) -> DataConfig:
        """Get DataConfig for loader compatibility."""
        # Create a minimal DataConfig from strategy data section
        from pathlib import Path

        return DataConfig(
            data_dir=Path("data/raw"),
            symbol="BTC/USDT",  # Default symbol, can be overridden
        )

    @property
    def feature_config(self) -> FeatureConfig:
        """Get feature configuration."""
        return self.features
