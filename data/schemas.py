"""Data schemas and models."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class OHLCVSchema(BaseModel):
    """
    Schema definition for OHLCV data.

    Validates that DataFrames contain required columns
    and have correct data types.
    """

    timestamp: str = Field(default="timestamp", description="Timestamp column name")
    open: str = Field(default="open", description="Open price column name")
    high: str = Field(default="high", description="High price column name")
    low: str = Field(default="low", description="Low price column name")
    close: str = Field(default="close", description="Close price column name")
    volume: str = Field(default="volume", description="Volume column name")
    num_trades: Optional[str] = Field(default=None, description="Number of trades column name")

    def get_required_columns(self) -> List[str]:
        """Get list of required column names."""
        cols = [self.timestamp, self.open, self.high, self.low, self.close, self.volume]
        if self.num_trades:
            cols.append(self.num_trades)
        return cols

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
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        return True


@dataclass
class DataQualityIssue:
    """Represents a data quality issue found during validation."""

    severity: str  # "error", "warning", "info"
    category: str  # "gap", "outlier", "missing", "invalid"
    message: str
    timestamp: Optional[datetime] = None
    details: Optional[dict] = None


@dataclass
class DataQualityReport:
    """
    Report of data quality validation.

    Contains overall quality score and list of issues found.
    """

    quality_score: float  # 0-1, where 1 is perfect
    total_rows: int
    issues: List[DataQualityIssue]
    start_time: datetime
    end_time: datetime
    gaps_count: int = 0
    outliers_count: int = 0
    missing_count: int = 0

    @property
    def is_valid(self) -> bool:
        """Check if data quality is acceptable (no errors)."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    def get_errors(self) -> List[DataQualityIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == "error"]

    def get_warnings(self) -> List[DataQualityIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    def summary(self) -> str:
        """
        Get human-readable summary of data quality.

        Returns:
            Summary string
        """
        errors = len(self.get_errors())
        warnings = len(self.get_warnings())

        summary = f"Data Quality Report\n"
        summary += f"==================\n"
        summary += f"Quality Score: {self.quality_score:.2%}\n"
        summary += f"Total Rows: {self.total_rows:,}\n"
        summary += f"Time Range: {self.start_time} to {self.end_time}\n"
        summary += f"\n"
        summary += f"Issues Found:\n"
        summary += f"  Errors: {errors}\n"
        summary += f"  Warnings: {warnings}\n"
        summary += f"  Gaps: {self.gaps_count}\n"
        summary += f"  Outliers: {self.outliers_count}\n"
        summary += f"  Missing Values: {self.missing_count}\n"

        if errors > 0:
            summary += f"\nERRORS:\n"
            for issue in self.get_errors():
                summary += f"  - {issue.message}\n"

        if warnings > 0 and warnings <= 10:  # Only show first 10 warnings
            summary += f"\nWARNINGS (showing first 10):\n"
            for issue in self.get_warnings()[:10]:
                summary += f"  - {issue.message}\n"

        return summary
