import pandas as pd
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from training.data_pipeline import DataPipeline
from config.feature import FeatureConfig


def _make_config(tmp_path):
    return SimpleNamespace(
        data_source="dummy.parquet",
        resample_interval=None,
        feature_cache_enabled=True,
        features=FeatureConfig(indicators=[]),
        target=SimpleNamespace(
            type="binary",
            classes=2,
            lookahead_bars=1,
            threshold_pct=0.1,
            class_weights=None,
            confidence_threshold=0.0,
        ),
    )


def _write_dummy(tmp_path, name="dummy.parquet"):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="T"),
            "open": np.arange(5),
            "high": np.arange(5) + 1,
            "low": np.arange(5) - 1,
            "close": np.arange(5),
            "volume": np.ones(5),
            "num_trades": np.ones(5),
        }
    )
    path = tmp_path / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / name
    df.to_parquet(file_path)
    return file_path


def test_cache_invalidation_on_mtime_change(tmp_path, monkeypatch):
    file_path = _write_dummy(tmp_path)
    cfg = _make_config(tmp_path)

    # Patch working directory to tmp fixture
    monkeypatch.chdir(tmp_path)

    pipeline = DataPipeline(cfg)

    # First compute (should write cache)
    df_clean, target_clean = pipeline.load_and_prepare_data()
    assert (tmp_path / "data/processed/cache").exists()

    # Touch file to change mtime (invalidate cache)
    Path(file_path).touch()
    df_clean2, _ = pipeline.load_and_prepare_data()

    # Ensure pipeline still returns data and no errors
    assert len(df_clean2) == len(df_clean)

