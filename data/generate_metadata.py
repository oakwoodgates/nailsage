"""
Generate metadata for data files in data/raw directory.

This script scans the data/raw directory, loads each data file,
validates it, and generates corresponding metadata files.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config.data import DataConfig
from data.loader import DataLoader
from data.validator import DataValidator
from data.metadata import DatasetMetadata, save_metadata, get_metadata_path
from utils.logger import setup_logger, get_data_logger

# Setup logging
setup_logger(level=20)  # INFO level
logger = get_data_logger()


def extract_info_from_filename(filename: str) -> dict:
    """
    Extract metadata from filename.

    Supports multiple formats:
    1. Standard: {exchange}_{asset}_{quote}_{market_type}_{interval}.{ext}
       Example: binance_btc_usdt_perps_1m.parquet
    2. With prefix: merged_{exchange}_{asset}_{quote}_{market_type}_{interval}_{timestamp}.{ext}
       Example: merged_binance_BTC_USDT_perps_1m_20251108_033030.parquet

    Args:
        filename: Filename to parse

    Returns:
        Dictionary with extracted info
    """
    # Remove extension
    name = Path(filename).stem

    # Split by underscore
    parts = name.split("_")

    # Handle "merged_" prefix or similar
    if parts[0].lower() in ["merged", "combined", "concat", "aggregated"]:
        parts = parts[1:]  # Skip prefix

    # Extract core components (ignore timestamp suffixes)
    # Look for common interval patterns: 1m, 5m, 15m, 1h, 4h, 1d
    interval_idx = None
    for i, part in enumerate(parts):
        if part.lower() in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
            interval_idx = i
            break

    if interval_idx is not None and interval_idx >= 4:
        # We found the interval, use positions before it
        return {
            "exchange": parts[0],
            "asset": parts[1],
            "quote": parts[2],
            "market_type": parts[3],
            "interval": parts[interval_idx],
        }
    elif len(parts) >= 5:
        # Fallback to first 5 parts
        return {
            "exchange": parts[0],
            "asset": parts[1],
            "quote": parts[2],
            "market_type": parts[3],
            "interval": parts[4],
        }
    else:
        # Fallback: prompt user or use defaults
        logger.warning(f"Filename {filename} doesn't match expected pattern")
        return {
            "exchange": "unknown",
            "asset": "unknown",
            "quote": "unknown",
            "market_type": "unknown",
            "interval": "1m",
        }


def generate_metadata_for_file(
    file_path: Path,
    asset: str = None,
    quote: str = None,
    exchange: str = None,
    market_type: str = None,
    interval: str = None,
    force: bool = False,
) -> Path:
    """
    Generate metadata for a single data file.

    Args:
        file_path: Path to data file
        asset: Asset symbol (if None, extracted from filename)
        quote: Quote currency (if None, extracted from filename)
        exchange: Exchange name (if None, extracted from filename)
        market_type: Market type (if None, extracted from filename)
        interval: Data interval (if None, extracted from filename)
        force: If True, regenerate even if metadata exists

    Returns:
        Path to generated metadata file
    """
    # Check if metadata already exists
    metadata_path = get_metadata_path(file_path)

    if metadata_path.exists() and not force:
        logger.info(f"Metadata already exists for {file_path.name}, skipping")
        return metadata_path

    logger.info(f"Generating metadata for {file_path.name}")

    # Extract info from filename if not provided
    file_info = extract_info_from_filename(file_path.name)

    asset = asset or file_info["asset"]
    quote = quote or file_info["quote"]
    exchange = exchange or file_info["exchange"]
    market_type = market_type or file_info["market_type"]
    interval = interval or file_info["interval"]

    # Create config for loading
    config = DataConfig(
        data_dir=file_path.parent,
        symbol=f"{asset}/{quote}",
        format="parquet" if file_path.suffix == ".parquet" else "csv",
    )

    # Load data
    loader = DataLoader(config)
    df = loader.load(filename=file_path.name)

    logger.info(f"Loaded {len(df):,} rows from {file_path.name}")

    # Validate data
    validator = DataValidator(config)
    quality_report = validator.validate(df)

    logger.info(f"Data quality score: {quality_report.quality_score:.2%}")

    # Create metadata
    metadata = DatasetMetadata.from_dataframe(
        df=df,
        asset=asset,
        quote=quote,
        exchange=exchange,
        market_type=market_type,
        interval=interval,
        source_file=str(file_path),
        data_quality_score=quality_report.quality_score,
        has_gaps=quality_report.gaps_count > 0,
        gap_count=quality_report.gaps_count,
        notes=f"Generated from {file_path.name}",
    )

    # Save metadata
    saved_path = save_metadata(metadata, file_path)
    logger.info(f"Saved metadata to {saved_path}")

    # Print summary
    print("\n" + metadata.summary())

    return saved_path


def generate_metadata_for_directory(
    directory: Path = Path("data/raw"),
    pattern: str = "*.*",
    force: bool = False,
) -> list[Path]:
    """
    Generate metadata for all data files in a directory.

    Args:
        directory: Directory to scan
        pattern: File pattern to match (e.g., "*.parquet", "*.csv")
        force: If True, regenerate even if metadata exists

    Returns:
        List of paths to generated metadata files
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all data files
    data_files = []
    for ext in ["parquet", "csv"]:
        data_files.extend(list(directory.glob(f"*.{ext}")))

    if not data_files:
        logger.warning(f"No data files found in {directory}")
        return []

    logger.info(f"Found {len(data_files)} data file(s) in {directory}")

    # Generate metadata for each file
    metadata_files = []
    for data_file in data_files:
        try:
            metadata_path = generate_metadata_for_file(data_file, force=force)
            metadata_files.append(metadata_path)
        except Exception as e:
            logger.error(f"Failed to generate metadata for {data_file}: {e}")
            continue

    logger.info(f"Generated metadata for {len(metadata_files)} file(s)")
    return metadata_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate metadata for data files")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing data files",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Generate metadata for a specific file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate metadata even if it exists",
    )
    parser.add_argument(
        "--asset",
        type=str,
        help="Asset symbol (e.g., BTC)",
    )
    parser.add_argument(
        "--quote",
        type=str,
        help="Quote currency (e.g., USDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        help="Exchange name (e.g., binance)",
    )
    parser.add_argument(
        "--market-type",
        type=str,
        help="Market type (e.g., spot, perps)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        help="Data interval (e.g., 1m, 15m)",
    )

    args = parser.parse_args()

    if args.file:
        # Generate metadata for a specific file
        generate_metadata_for_file(
            args.file,
            asset=args.asset,
            quote=args.quote,
            exchange=args.exchange,
            market_type=args.market_type,
            interval=args.interval,
            force=args.force,
        )
    else:
        # Generate metadata for all files in directory
        generate_metadata_for_directory(
            directory=args.dir,
            force=args.force,
        )
