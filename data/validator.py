"""Data validation for OHLCV data quality."""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from config.data import DataConfig
from data.schemas import DataQualityIssue, DataQualityReport
from utils.logger import get_data_logger

logger = get_data_logger()


class DataValidator:
    """
    Validates OHLCV data quality.

    Checks for:
    - Gaps in time series
    - Price outliers
    - Missing values
    - Invalid OHLC relationships (high < low, etc.)
    - Volume anomalies
    """

    def __init__(self, config: DataConfig):
        """
        Initialize DataValidator.

        Args:
            config: Data configuration with validation parameters
        """
        self.config = config
        logger.info("Initialized DataValidator")

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate DataFrame and generate quality report.

        Args:
            df: DataFrame to validate

        Returns:
            DataQualityReport with issues and quality score
        """
        logger.info(f"Validating {len(df):,} rows of data")

        issues: List[DataQualityIssue] = []

        # Check for missing values
        missing_issues = self._check_missing_values(df)
        issues.extend(missing_issues)

        # Check for gaps
        gap_issues = self._check_gaps(df)
        issues.extend(gap_issues)

        # Check for outliers
        outlier_issues = self._check_outliers(df)
        issues.extend(outlier_issues)

        # Check OHLC validity
        ohlc_issues = self._check_ohlc_validity(df)
        issues.extend(ohlc_issues)

        # Check volume
        volume_issues = self._check_volume(df)
        issues.extend(volume_issues)

        # Calculate quality score
        quality_score = self._calculate_quality_score(df, issues)

        # Count issue types
        gaps_count = len([i for i in issues if i.category == "gap"])
        outliers_count = len([i for i in issues if i.category == "outlier"])
        missing_count = len([i for i in issues if i.category == "missing"])

        # Create report
        report = DataQualityReport(
            quality_score=quality_score,
            total_rows=len(df),
            issues=issues,
            start_time=df[self.config.timestamp_column].min(),
            end_time=df[self.config.timestamp_column].max(),
            gaps_count=gaps_count,
            outliers_count=outliers_count,
            missing_count=missing_count,
        )

        logger.info(
            f"Validation complete: Quality score = {quality_score:.2%}",
            extra_data={
                "errors": len(report.get_errors()),
                "warnings": len(report.get_warnings()),
                "gaps": gaps_count,
                "outliers": outliers_count,
            },
        )

        return report

    def _check_missing_values(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for missing values in critical columns."""
        issues = []

        critical_cols = [
            self.config.timestamp_column,
            self.config.open_column,
            self.config.high_column,
            self.config.low_column,
            self.config.close_column,
            self.config.volume_column,
        ]

        for col in critical_cols:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count > 0:
                severity = "error" if missing_count > len(df) * 0.01 else "warning"
                issues.append(
                    DataQualityIssue(
                        severity=severity,
                        category="missing",
                        message=f"Column '{col}' has {missing_count} missing values ({missing_count/len(df):.1%})",
                        details={"column": col, "count": int(missing_count)},
                    )
                )

        return issues

    def _check_gaps(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for gaps in time series."""
        issues = []

        if len(df) < 2:
            return issues

        timestamp_col = self.config.timestamp_column
        timestamps = df[timestamp_col]

        # Calculate time differences
        time_diffs = timestamps.diff().dropna()

        # Find gaps larger than expected
        max_gap = timedelta(seconds=self.config.max_gap_seconds)
        gaps = time_diffs[time_diffs > max_gap]

        if len(gaps) > 0:
            severity = "error" if not self.config.allow_gaps else "warning"

            for idx, gap in gaps.items():
                gap_seconds = gap.total_seconds()
                timestamp = df.loc[idx, timestamp_col]

                issues.append(
                    DataQualityIssue(
                        severity=severity,
                        category="gap",
                        message=f"Gap of {gap_seconds/60:.1f} minutes at {timestamp}",
                        timestamp=timestamp,
                        details={"gap_seconds": gap_seconds},
                    )
                )

            # Summary issue
            total_gap_time = gaps.sum().total_seconds()
            issues.append(
                DataQualityIssue(
                    severity=severity,
                    category="gap",
                    message=f"Found {len(gaps)} gaps totaling {total_gap_time/3600:.1f} hours",
                    details={"gap_count": len(gaps), "total_gap_hours": total_gap_time / 3600},
                )
            )

        return issues

    def _check_outliers(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for price outliers using statistical methods."""
        issues = []

        price_col = self.config.close_column

        if price_col not in df.columns or len(df) < 10:
            return issues

        prices = df[price_col].dropna()

        # Calculate returns
        returns = prices.pct_change().dropna()

        if len(returns) == 0:
            return issues

        # Find outliers using z-score
        mean_return = returns.mean()
        std_return = returns.std()
        threshold = self.config.outlier_threshold_std

        outliers = returns[np.abs(returns - mean_return) > threshold * std_return]

        if len(outliers) > 0:
            for idx, ret in outliers.items():
                if idx in df.index:
                    timestamp = df.loc[idx, self.config.timestamp_column]
                    price = df.loc[idx, price_col]

                    issues.append(
                        DataQualityIssue(
                            severity="warning",
                            category="outlier",
                            message=f"Price outlier: {ret:.2%} return at {timestamp} (price: {price})",
                            timestamp=timestamp,
                            details={"return": float(ret), "price": float(price)},
                        )
                    )

            # Limit number of outlier issues to avoid spam
            if len(issues) > 20:
                issues = issues[:20]
                issues.append(
                    DataQualityIssue(
                        severity="warning",
                        category="outlier",
                        message=f"... and {len(outliers) - 20} more outliers (showing first 20)",
                        details={"total_outliers": len(outliers)},
                    )
                )

        return issues

    def _check_ohlc_validity(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check that OHLC relationships are valid (high >= low, etc.)."""
        issues = []

        # Check high >= low
        invalid_hl = df[df[self.config.high_column] < df[self.config.low_column]]
        if len(invalid_hl) > 0:
            for idx, row in invalid_hl.iterrows():
                issues.append(
                    DataQualityIssue(
                        severity="error",
                        category="invalid",
                        message=f"High < Low at {row[self.config.timestamp_column]}",
                        timestamp=row[self.config.timestamp_column],
                        details={
                            "high": float(row[self.config.high_column]),
                            "low": float(row[self.config.low_column]),
                        },
                    )
                )

        # Check open and close within high/low range
        invalid_o = df[
            (df[self.config.open_column] > df[self.config.high_column])
            | (df[self.config.open_column] < df[self.config.low_column])
        ]
        if len(invalid_o) > 0:
            issues.append(
                DataQualityIssue(
                    severity="error",
                    category="invalid",
                    message=f"Found {len(invalid_o)} bars where open price is outside high/low range",
                    details={"count": len(invalid_o)},
                )
            )

        invalid_c = df[
            (df[self.config.close_column] > df[self.config.high_column])
            | (df[self.config.close_column] < df[self.config.low_column])
        ]
        if len(invalid_c) > 0:
            issues.append(
                DataQualityIssue(
                    severity="error",
                    category="invalid",
                    message=f"Found {len(invalid_c)} bars where close price is outside high/low range",
                    details={"count": len(invalid_c)},
                )
            )

        return issues

    def _check_volume(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for volume anomalies."""
        issues = []

        volume_col = self.config.volume_column

        # Check for negative volume
        negative_vol = df[df[volume_col] < 0]
        if len(negative_vol) > 0:
            issues.append(
                DataQualityIssue(
                    severity="error",
                    category="invalid",
                    message=f"Found {len(negative_vol)} bars with negative volume",
                    details={"count": len(negative_vol)},
                )
            )

        # Check for zero/very low volume
        low_vol = df[df[volume_col] < self.config.min_volume]
        if len(low_vol) > 0:
            issues.append(
                DataQualityIssue(
                    severity="warning",
                    category="invalid",
                    message=f"Found {len(low_vol)} bars with volume < {self.config.min_volume}",
                    details={"count": len(low_vol), "threshold": self.config.min_volume},
                )
            )

        return issues

    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[DataQualityIssue]) -> float:
        """
        Calculate overall data quality score (0-1).

        Args:
            df: DataFrame
            issues: List of issues found

        Returns:
            Quality score between 0 and 1
        """
        if len(df) == 0:
            return 0.0

        # Start with perfect score
        score = 1.0

        # Deduct points for each issue based on severity
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")

        # Errors are more severe
        error_penalty = min(0.5, (error_count / len(df)) * 10)  # Max 50% penalty
        warning_penalty = min(0.2, (warning_count / len(df)) * 5)  # Max 20% penalty

        score -= error_penalty
        score -= warning_penalty

        return max(0.0, score)
