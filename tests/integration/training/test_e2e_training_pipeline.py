import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.strategy import StrategyConfig
from training.pipeline import TrainingPipeline


pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_E2E_TRAINING"),
    reason="Set RUN_E2E_TRAINING=1 to run end-to-end training pipeline test",
)


def test_train_pipeline_end_to_end(monkeypatch, tmp_path):
    # Arrange: synthetic OHLCV data
    n = 300
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="T"),
            "open": np.linspace(100, 101, n),
            "high": np.linspace(100.5, 101.5, n),
            "low": np.linspace(99.5, 100.5, n),
            "close": np.linspace(100, 102, n),
            "volume": np.ones(n),
            "num_trades": np.ones(n),
        }
    )
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "fixture.parquet"
    df.to_parquet(data_path)

    # Strategy config YAML
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = strategies_dir / "fixture_strategy.yaml"
    yaml_path.write_text(
        f"""
strategy_name: fixture_strategy
version: v1
strategy_timeframe: test
description: E2E fixture
data:
  source_file: fixture.parquet
  resample_interval: null
  train_start: "2021-01-01"
  train_end: "2021-01-02"
  validation_start: "2021-01-01"
  validation_end: "2021-01-02"
features:
  indicators: []
target:
  type: binary
  classes: 2
  lookahead_bars: 1
  threshold_pct: 0.1
model:
  type: extra_trees
  params: {{}}
validation:
  method: walk_forward
  n_splits: 2
  expanding_window: true
  gap_bars: 0
backtest:
  transaction_cost_pct: 0.04
  slippage_bps: 0
  leverage: 1
  capital: 10000
  min_bars_between_trades: 0
"""
    )

    # Use temp workspace
    monkeypatch.chdir(tmp_path)

    cfg = StrategyConfig.from_yaml(str(yaml_path))
    pipeline = TrainingPipeline(cfg)
    # Loosen splitter requirements for small fixture
    pipeline.validator.splitter.min_train_size = 50
    pipeline.validator.splitter.test_size = 0.2

    # Act
    result = pipeline.run(train_only=True, save_results=False)

    # Assert
    assert result.model_id
    assert result.training_metrics.get("train_accuracy") is not None

