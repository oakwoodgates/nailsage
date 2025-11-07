"""Data loading for OHLCV data from various sources."""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from config.data import DataConfig, DataFormat
from data.schemas import OHLCVSchema
from utils.logger import get_data_logger

logger = get_data_logger()


class DataLoader:
    """
    Loads OHLCV data from files (Parquet or CSV).

    Handles schema validation, type conversion, and timestamp parsing.
    """

    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader.

        Args:
            config: Data configuration
        """
        self.config = config
        self.schema = OHLCVSchema(
            timestamp=config.timestamp_column,
            open=config.open_column,
            high=config.high_column,
            low=config.low_column,
            close=config.close_column,
            volume=config.volume_column,
            num_trades=config.num_trades_column,
        )

        logger.info(
            f"Initialized DataLoader for {config.symbol}",
            extra_data={"format": config.format, "data_dir": str(config.data_dir)},
        )

    def load(
        self,
        filename: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from file.

        Args:
            filename: Filename to load (if None, uses symbol to find file)
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)

        Returns:
            DataFrame with OHLCV data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is invalid
        """
        # Determine file path
        if filename is None:
            # Auto-detect based on symbol
            filename = self._get_default_filename()

        file_path = self.config.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(
            f"Loading data from {file_path}",
            extra_data={"start_date": start_date, "end_date": end_date},
        )

        # Load based on format
        if self.config.format == DataFormat.PARQUET:
            df = self._load_parquet(file_path)
        elif self.config.format == DataFormat.CSV:
            df = self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported data format: {self.config.format}")

        # Validate schema
        self.schema.validate_dataframe(df)

        # Process DataFrame
        df = self._process_dataframe(df)

        # Filter by date range if specified
        if start_date or end_date:
            df = self._filter_by_date(df, start_date, end_date)

        logger.info(
            f"Loaded {len(df):,} rows",
            extra_data={
                "start": str(df[self.config.timestamp_column].min()),
                "end": str(df[self.config.timestamp_column].max()),
            },
        )

        return df

    def _get_default_filename(self) -> str:
        """
        Get default filename based on symbol.

        Returns:
            Filename string
        """
        # Convert symbol to filename-friendly format
        # BTC/USD -> btc_usd
        # BTC-PERP -> btc_perp
        symbol_clean = (
            self.config.symbol.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
        )

        if self.config.format == DataFormat.PARQUET:
            return f"{symbol_clean}.parquet"
        else:
            return f"{symbol_clean}.csv"

    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            DataFrame
        """
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}", extra_data={"path": str(file_path)})
            raise

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}", extra_data={"path": str(file_path)})
            raise

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame: convert types, parse timestamps, sort.

        Args:
            df: Raw DataFrame

        Returns:
            Processed DataFrame
        """
        df = df.copy()

        # Parse timestamp column
        timestamp_col = self.config.timestamp_column
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            # Try to parse as datetime
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except Exception as e:
                logger.error(
                    f"Failed to parse timestamp column: {e}",
                    extra_data={"column": timestamp_col},
                )
                raise ValueError(f"Cannot parse timestamp column '{timestamp_col}': {e}")

        # Ensure timezone-aware (UTC)
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
        else:
            df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")

        # Convert price columns to float
        price_cols = [
            self.config.open_column,
            self.config.high_column,
            self.config.low_column,
            self.config.close_column,
        ]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert volume to float
        df[self.config.volume_column] = pd.to_numeric(df[self.config.volume_column], errors="coerce")

        # Convert num_trades if present
        if self.config.num_trades_column and self.config.num_trades_column in df.columns:
            df[self.config.num_trades_column] = pd.to_numeric(
                df[self.config.num_trades_column], errors="coerce"
            )

        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Check for NaN values in critical columns
        critical_cols = [timestamp_col] + price_cols + [self.config.volume_column]
        nan_counts = df[critical_cols].isna().sum()
        if nan_counts.any():
            logger.warning(
                "Found NaN values in data",
                extra_data={"nan_counts": nan_counts[nan_counts > 0].to_dict()},
            )

        return df

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: DataFrame to filter
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Filtered DataFrame
        """
        timestamp_col = self.config.timestamp_column

        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize("UTC")
            df = df[df[timestamp_col] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize("UTC")
            df = df[df[timestamp_col] <= end_dt]

        return df.reset_index(drop=True)

    def get_latest_timestamp(self, filename: Optional[str] = None) -> pd.Timestamp:
        """
        Get the latest timestamp in the data file without loading all data.

        Args:
            filename: Optional filename (uses default if None)

        Returns:
            Latest timestamp
        """
        if filename is None:
            filename = self._get_default_filename()

        file_path = self.config.data_dir / filename

        if self.config.format == DataFormat.PARQUET:
            # Use pyarrow to read just the timestamp column
            table = pq.read_table(file_path, columns=[self.config.timestamp_column])
            df = table.to_pandas()
        else:
            # For CSV, read just the timestamp column
            df = pd.read_csv(file_path, usecols=[self.config.timestamp_column])

        df[self.config.timestamp_column] = pd.to_datetime(df[self.config.timestamp_column])
        return df[self.config.timestamp_column].max()
