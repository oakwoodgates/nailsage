"""Data metadata tracking for provenance and reproducibility."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetMetadata:
    """
    Metadata for a trading dataset.

    Tracks all information needed for reproducibility and provenance.
    """

    # Asset information
    asset: str  # e.g., "BTC"
    quote: str  # e.g., "USDT"
    exchange: str  # e.g., "Binance"
    market_type: str  # e.g., "spot", "perps", "futures"

    # Time information
    interval: str  # e.g., "1m", "15m", "1h"
    start_date: str  # ISO format: "2024-01-01T00:00:00Z"
    end_date: str  # ISO format: "2024-12-31T23:59:59Z"
    num_bars: int  # Total number of bars in dataset

    # Data quality
    data_quality_score: float  # 0-1 score from DataValidator
    has_gaps: bool  # Whether data has gaps
    gap_count: int  # Number of gaps found

    # Source information
    source_file: str  # Path to data file
    file_format: str  # "parquet" or "csv"
    file_size_mb: float  # File size in megabytes

    # Metadata
    created_at: str  # When metadata was created
    columns: list[str]  # Column names in dataset

    # Optional fields
    notes: Optional[str] = None  # Any additional notes

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path | str) -> None:
        """
        Save metadata to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "DatasetMetadata":
        """
        Load metadata from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            DatasetMetadata instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def load(cls, path: Path | str) -> "DatasetMetadata":
        """
        Load metadata from JSON file (alias for from_json).

        Args:
            path: Path to JSON file

        Returns:
            DatasetMetadata instance
        """
        return cls.from_json(path)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        asset: str,
        quote: str,
        exchange: str,
        market_type: str,
        interval: str,
        source_file: str,
        data_quality_score: float = 1.0,
        has_gaps: bool = False,
        gap_count: int = 0,
        notes: Optional[str] = None,
    ) -> "DatasetMetadata":
        """
        Create metadata from a DataFrame.

        Args:
            df: DataFrame with OHLCV data
            asset: Asset symbol (e.g., "BTC")
            quote: Quote currency (e.g., "USDT")
            exchange: Exchange name (e.g., "Binance")
            market_type: Market type (e.g., "spot", "perps")
            interval: Data interval (e.g., "1m", "15m")
            source_file: Path to source data file
            data_quality_score: Quality score from validator
            has_gaps: Whether data has gaps
            gap_count: Number of gaps
            notes: Optional notes

        Returns:
            DatasetMetadata instance
        """
        # Extract time range
        if "timestamp" in df.columns:
            start_date = pd.to_datetime(df["timestamp"].min()).isoformat()
            end_date = pd.to_datetime(df["timestamp"].max()).isoformat()
        else:
            # Use index if it's a datetime index
            start_date = df.index.min().isoformat()
            end_date = df.index.max().isoformat()

        # Get file info
        file_path = Path(source_file)
        file_format = file_path.suffix.lstrip(".")
        file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0

        return cls(
            asset=asset.upper(),
            quote=quote.upper(),
            exchange=exchange.lower(),
            market_type=market_type.lower(),
            interval=interval.lower(),
            start_date=start_date,
            end_date=end_date,
            num_bars=len(df),
            data_quality_score=data_quality_score,
            has_gaps=has_gaps,
            gap_count=gap_count,
            source_file=str(source_file),
            file_format=file_format,
            file_size_mb=round(file_size_mb, 2),
            created_at=datetime.utcnow().isoformat() + "Z",
            columns=list(df.columns),
            notes=notes,
        )

    def get_metadata_filename(self) -> str:
        """
        Generate standard metadata filename.

        Returns:
            Filename string (e.g., "binance_btc_usdt_perps_1m.metadata.json")
        """
        base = f"{self.exchange}_{self.asset}_{self.quote}_{self.market_type}_{self.interval}"
        return f"{base}.metadata.json"

    def summary(self) -> str:
        """
        Get human-readable summary.

        Returns:
            Formatted summary string
        """
        summary = "Dataset Metadata\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Asset:        {self.asset}/{self.quote}\n"
        summary += f"Exchange:     {self.exchange.title()}\n"
        summary += f"Market Type:  {self.market_type.title()}\n"
        summary += f"Interval:     {self.interval}\n"
        summary += f"\n"
        summary += f"Date Range:   {self.start_date} to {self.end_date}\n"
        summary += f"Num Bars:     {self.num_bars:,}\n"
        summary += f"\n"
        summary += f"Quality:      {self.data_quality_score:.2%}\n"
        summary += f"Has Gaps:     {self.has_gaps}\n"
        summary += f"Gap Count:    {self.gap_count}\n"
        summary += f"\n"
        summary += f"Source File:  {self.source_file}\n"
        summary += f"File Format:  {self.file_format}\n"
        summary += f"File Size:    {self.file_size_mb:.2f} MB\n"
        summary += f"\n"
        summary += f"Columns:      {', '.join(self.columns)}\n"

        if self.notes:
            summary += f"\nNotes: {self.notes}\n"

        return summary


def get_metadata_path(data_file_path: Path | str) -> Path:
    """
    Get metadata file path for a data file.

    Adds .metadata.json to the data file name.

    Args:
        data_file_path: Path to data file

    Returns:
        Path to metadata file

    Example:
        >>> get_metadata_path("data/raw/btc_usdt.parquet")
        Path("data/raw/btc_usdt.metadata.json")
    """
    data_file_path = Path(data_file_path)
    # Remove extension and add .metadata.json
    base_name = data_file_path.stem
    return data_file_path.parent / f"{base_name}.metadata.json"


def load_metadata(data_file_path: Path | str) -> Optional[DatasetMetadata]:
    """
    Load metadata for a data file.

    Args:
        data_file_path: Path to data file

    Returns:
        DatasetMetadata if found, None otherwise
    """
    metadata_path = get_metadata_path(data_file_path)

    if not metadata_path.exists():
        return None

    return DatasetMetadata.from_json(metadata_path)


def save_metadata(metadata: DatasetMetadata, data_file_path: Path | str) -> Path:
    """
    Save metadata for a data file.

    Args:
        metadata: DatasetMetadata instance
        data_file_path: Path to data file

    Returns:
        Path to saved metadata file
    """
    metadata_path = get_metadata_path(data_file_path)
    metadata.to_json(metadata_path)
    return metadata_path
