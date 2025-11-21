"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    data = {
        "timestamp": dates,
        "open": 100.0 + pd.Series(range(100)) * 0.1,
        "high": 101.0 + pd.Series(range(100)) * 0.1,
        "low": 99.0 + pd.Series(range(100)) * 0.1,
        "close": 100.5 + pd.Series(range(100)) * 0.1,
        "volume": 1000.0 + pd.Series(range(100)) * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ohlcv_with_gaps():
    """Generate sample OHLCV data with gaps for testing."""
    # Create 100 hours of data but skip some hours to create gaps
    dates = pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC")
    # Remove some rows to create gaps
    dates = dates[[i for i in range(120) if i not in [10, 11, 50, 51, 52]]]

    data = {
        "timestamp": dates,
        "open": 100.0 + pd.Series(range(len(dates))) * 0.1,
        "high": 101.0 + pd.Series(range(len(dates))) * 0.1,
        "low": 99.0 + pd.Series(range(len(dates))) * 0.1,
        "close": 100.5 + pd.Series(range(len(dates))) * 0.1,
        "volume": 1000.0 + pd.Series(range(len(dates))) * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for data files."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary directories for model storage."""
    models_dir = tmp_path / "models"
    trained_dir = models_dir / "trained"
    metadata_dir = models_dir / "metadata"

    trained_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    return {"models": models_dir, "trained": trained_dir, "metadata": metadata_dir}
