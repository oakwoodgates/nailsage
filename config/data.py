"""Data configuration for OHLCV loading and validation."""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator

from config.base import BaseConfig


class DataFormat(str, Enum):
    """Supported data formats."""

    PARQUET = "parquet"
    CSV = "csv"


class DataConfig(BaseConfig):
    """
    Configuration for data loading and validation.

    Defines how to load OHLCV data, validation parameters,
    and quality thresholds.
    """

    # Data source
    data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory containing raw OHLCV data",
    )
    format: DataFormat = Field(
        default=DataFormat.PARQUET,
        description="Data format (parquet or csv)",
    )
    symbol: str = Field(
        description="Trading symbol (e.g., 'BTC/USD', 'BTC-PERP')",
        examples=["BTC/USD", "BTC-PERP", "ETH/USD"],
    )

    # Schema
    timestamp_column: str = Field(
        default="timestamp",
        description="Name of timestamp column",
    )
    open_column: str = Field(default="open", description="Name of open price column")
    high_column: str = Field(default="high", description="Name of high price column")
    low_column: str = Field(default="low", description="Name of low price column")
    close_column: str = Field(default="close", description="Name of close price column")
    volume_column: str = Field(default="volume", description="Name of volume column")
    num_trades_column: Optional[str] = Field(
        default="num_trades",
        description="Name of number of trades column (optional)",
    )

    # Validation parameters
    allow_gaps: bool = Field(
        default=False,
        description="Allow gaps in time series (if False, will raise error)",
    )
    max_gap_seconds: int = Field(
        default=300,  # 5 minutes
        description="Maximum allowed gap between bars in seconds",
    )
    min_volume: float = Field(
        default=0.0,
        description="Minimum volume threshold (bars below will be flagged)",
    )
    outlier_threshold_std: float = Field(
        default=5.0,
        description="Standard deviations for price outlier detection",
    )

    # Data quality
    min_data_quality_score: float = Field(
        default=0.95,
        description="Minimum data quality score (0-1) to pass validation",
        ge=0.0,
        le=1.0,
    )

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: Path) -> Path:
        """Ensure data directory exists or can be created."""
        if isinstance(v, str):
            v = Path(v)
        # Don't create here, just validate it's a valid path
        return v

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        if not v or len(v) < 3:
            raise ValueError("Symbol must be at least 3 characters")
        return v.upper()

    def get_columns(self) -> List[str]:
        """
        Get list of all expected column names.

        Returns:
            List of column names in order: timestamp, OHLCV, optional extras
        """
        cols = [
            self.timestamp_column,
            self.open_column,
            self.high_column,
            self.low_column,
            self.close_column,
            self.volume_column,
        ]

        if self.num_trades_column:
            cols.append(self.num_trades_column)

        return cols
