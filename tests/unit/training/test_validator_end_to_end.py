import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.dummy import DummyClassifier

from training.validator import Validator


def _make_dummy_config():
    return SimpleNamespace(
        strategy_name="dummy",
        version="v1",
        strategy_timeframe="test",
        description=None,
        data_source="dummy.parquet",
        resample_interval=None,
        target=SimpleNamespace(
            type="binary",
            classes=2,
            lookahead_bars=1,
            threshold_pct=0.1,
            class_weights=None,
            confidence_threshold=0.0,
        ),
        features={},
        validation=SimpleNamespace(method="walk_forward", n_splits=3, expanding_window=True, gap_bars=0),
        backtest=SimpleNamespace(
            transaction_cost_pct=0.04,
            slippage_bps=0,
            leverage=1,
            capital=10000,
            min_bars_between_trades=0,
        ),
        model=SimpleNamespace(type="dummy", params={}),
    )


def test_validator_walk_forward_trains_per_split(monkeypatch):
    cfg = _make_dummy_config()
    validator = Validator(cfg)
    validator.splitter.min_train_size = 50
    validator.splitter.test_size = 0.1

    # Patch model factory to deterministic dummy classifier
    monkeypatch.setattr(validator, "_create_model", lambda: DummyClassifier(strategy="most_frequent"))

    # Create synthetic dataset with required columns
    n = 1100
    ts = pd.date_range("2020-01-01", periods=n, freq="T")
    df_clean = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.linspace(100, 101, n),
            "high": np.linspace(100.5, 101.5, n),
            "low": np.linspace(99.5, 100.5, n),
            "close": np.linspace(100, 102, n),
            "volume": np.ones(n),
            "num_trades": np.ones(n),
            "feat1": np.sin(np.linspace(0, 10, n)),
        }
    )
    target_clean = pd.Series((np.arange(n) % 2), index=df_clean.index)

    feature_cols = [c for c in df_clean.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", "num_trades"]]

    results = validator.run_validation(None, df_clean, target_clean, feature_cols)

    assert results["n_splits"] > 0
    assert len(results["splits"]) == results["n_splits"]
    # Accuracy should be defined between 0 and 1
    for split in results["splits"]:
        assert 0.0 <= split["val_metrics"]["accuracy"] <= 1.0

