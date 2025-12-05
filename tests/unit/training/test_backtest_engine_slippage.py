import pandas as pd
import pytest
from types import SimpleNamespace

from config.backtest import BacktestConfig
from training.validation.backtest import BacktestEngine
from training.backtest_pipeline import BacktestPipeline


def test_slippage_applied_only_on_execution():
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "num_trades": 1,
            }
        ]
    ).set_index("timestamp")

    cfg = BacktestConfig(slippage_bps=50)  # 0.50%
    engine = BacktestEngine(cfg)
    bar = df.iloc[0]

    mtm_price = engine._get_mark_to_market_price(bar, "close")
    exec_price = engine._get_execution_price(bar, "close", apply_slippage=True)

    assert mtm_price == bar["close"]
    assert exec_price == pytest.approx(bar["close"] * 1.005)


def test_backtest_pipeline_resolves_binary_classes():
    dummy_config = SimpleNamespace(
        strategy_name="dummy",
        version="v1",
        strategy_timeframe="test",
        description=None,
        data_source="dummy.parquet",
        resample_interval=None,
        target=SimpleNamespace(
            type="binary",
            classes=None,
            lookahead_bars=1,
            threshold_pct=0.5,
            class_weights=None,
            confidence_threshold=0.0,
        ),
        features={},
        validation=None,
        backtest=None,
        model=SimpleNamespace(type="lightgbm", params={}),
    )

    pipeline = BacktestPipeline(dummy_config)
    assert pipeline._resolve_num_classes() == 2


def test_backtest_pipeline_rejects_regression_targets():
    dummy_config = SimpleNamespace(
        strategy_name="dummy",
        version="v1",
        strategy_timeframe="test",
        description=None,
        data_source="dummy.parquet",
        resample_interval=None,
        target=SimpleNamespace(
            type="regression",
            classes=None,
            lookahead_bars=1,
            threshold_pct=0.5,
            class_weights=None,
            confidence_threshold=0.0,
        ),
        features={},
        validation=None,
        backtest=None,
        model=SimpleNamespace(type="lightgbm", params={}),
    )

    pipeline = BacktestPipeline(dummy_config)
    with pytest.raises(ValueError):
        pipeline._resolve_num_classes()

