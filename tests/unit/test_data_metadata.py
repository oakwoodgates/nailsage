"""Unit tests for data metadata tracking."""

import json
from pathlib import Path

import pandas as pd
import pytest

from data.metadata import DatasetMetadata, get_metadata_path, load_metadata, save_metadata


@pytest.mark.unit
def test_dataset_metadata_creation():
    """Test creating DatasetMetadata."""
    metadata = DatasetMetadata(
        asset="BTC",
        quote="USDT",
        exchange="binance",
        market_type="perps",
        interval="1m",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-31T23:59:59Z",
        num_bars=44640,
        data_quality_score=0.99,
        has_gaps=False,
        gap_count=0,
        source_file="data/raw/binance_btc_usdt_perps_1m.parquet",
        file_format="parquet",
        file_size_mb=10.5,
        created_at="2025-01-01T00:00:00Z",
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    assert metadata.asset == "BTC"
    assert metadata.quote == "USDT"
    assert metadata.exchange == "binance"
    assert metadata.market_type == "perps"
    assert metadata.interval == "1m"
    assert metadata.num_bars == 44640
    assert metadata.data_quality_score == 0.99
    assert not metadata.has_gaps


@pytest.mark.unit
def test_dataset_metadata_to_dict():
    """Test converting metadata to dictionary."""
    metadata = DatasetMetadata(
        asset="BTC",
        quote="USDT",
        exchange="binance",
        market_type="perps",
        interval="1m",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-31T23:59:59Z",
        num_bars=44640,
        data_quality_score=0.99,
        has_gaps=False,
        gap_count=0,
        source_file="data/raw/binance_btc_usdt_perps_1m.parquet",
        file_format="parquet",
        file_size_mb=10.5,
        created_at="2025-01-01T00:00:00Z",
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    data_dict = metadata.to_dict()

    assert isinstance(data_dict, dict)
    assert data_dict["asset"] == "BTC"
    assert data_dict["quote"] == "USDT"
    assert data_dict["num_bars"] == 44640


@pytest.mark.unit
def test_dataset_metadata_save_load(tmp_path):
    """Test saving and loading metadata."""
    metadata = DatasetMetadata(
        asset="BTC",
        quote="USDT",
        exchange="binance",
        market_type="perps",
        interval="1m",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-31T23:59:59Z",
        num_bars=44640,
        data_quality_score=0.99,
        has_gaps=False,
        gap_count=0,
        source_file="data/raw/binance_btc_usdt_perps_1m.parquet",
        file_format="parquet",
        file_size_mb=10.5,
        created_at="2025-01-01T00:00:00Z",
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # Save
    save_path = tmp_path / "test_metadata.json"
    metadata.to_json(save_path)

    assert save_path.exists()

    # Load
    loaded = DatasetMetadata.load(save_path)

    assert loaded.asset == metadata.asset
    assert loaded.quote == metadata.quote
    assert loaded.num_bars == metadata.num_bars
    assert loaded.data_quality_score == metadata.data_quality_score


@pytest.mark.unit
def test_dataset_metadata_from_dataframe(sample_ohlcv_data, tmp_path):
    """Test creating metadata from DataFrame."""
    # Save sample data to a file
    data_file = tmp_path / "test_data.parquet"
    sample_ohlcv_data.to_parquet(data_file)

    metadata = DatasetMetadata.from_dataframe(
        df=sample_ohlcv_data,
        asset="BTC",
        quote="USDT",
        exchange="binance",
        market_type="spot",
        interval="1h",
        source_file=str(data_file),
    )

    assert metadata.asset == "BTC"
    assert metadata.quote == "USDT"
    assert metadata.num_bars == len(sample_ohlcv_data)
    assert metadata.file_format == "parquet"
    assert "timestamp" in metadata.columns
    assert "close" in metadata.columns


@pytest.mark.unit
def test_get_metadata_path():
    """Test metadata path generation."""
    data_path = Path("data/raw/btc_usdt_1m.parquet")
    metadata_path = get_metadata_path(data_path)

    assert metadata_path == Path("data/raw/btc_usdt_1m.metadata.json")
    assert metadata_path.suffix == ".json"


@pytest.mark.unit
def test_save_and_load_metadata_helpers(tmp_path, sample_ohlcv_data):
    """Test save_metadata and load_metadata helper functions."""
    # Create metadata
    data_file = tmp_path / "test_data.parquet"
    sample_ohlcv_data.to_parquet(data_file)

    metadata = DatasetMetadata.from_dataframe(
        df=sample_ohlcv_data,
        asset="ETH",
        quote="USDT",
        exchange="coinbase",
        market_type="spot",
        interval="5m",
        source_file=str(data_file),
    )

    # Save using helper
    saved_path = save_metadata(metadata, data_file)
    assert saved_path.exists()
    assert saved_path.name == "test_data.metadata.json"

    # Load using helper
    loaded = load_metadata(data_file)
    assert loaded is not None
    assert loaded.asset == "ETH"
    assert loaded.exchange == "coinbase"


@pytest.mark.unit
def test_metadata_summary():
    """Test metadata summary generation."""
    metadata = DatasetMetadata(
        asset="BTC",
        quote="USDT",
        exchange="binance",
        market_type="perps",
        interval="15m",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-31T23:59:59Z",
        num_bars=2976,
        data_quality_score=0.985,
        has_gaps=True,
        gap_count=3,
        source_file="data/raw/binance_btc_usdt_perps_15m.parquet",
        file_format="parquet",
        file_size_mb=2.5,
        created_at="2025-01-01T00:00:00Z",
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        notes="Test dataset",
    )

    summary = metadata.summary()

    assert isinstance(summary, str)
    assert "BTC/USDT" in summary
    assert "Binance" in summary
    assert "2,976" in summary
    assert "98.50%" in summary
    assert "Test dataset" in summary
