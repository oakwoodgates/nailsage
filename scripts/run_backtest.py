"""Run backtest on a trained model with historical data.

Usage:
    # Run backtest (default - saves results to JSON)
    python scripts/run_backtest.py --config configs/strategies/dev_scalper_1m_v1.yaml --model-id 33b9a1937aacaa4d_20251126_152519_f8835d

    # Run backtest without saving JSON
    python scripts/run_backtest.py --config configs/strategies/dev_scalper_1m_v1.yaml --model-id MODEL_ID --no-json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from config.backtest import BacktestConfig
from config.strategy import StrategyConfig
from features.engine import FeatureEngine
from models import ModelRegistry
from models.feature_schema import FeatureSchema
from targets.classification import create_3class_target
from validation.backtest import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def apply_trade_cooldown(signals: pd.Series, min_bars: int) -> pd.Series:
    """
    Apply cooldown between trades - suppress signals within min_bars of last trade.

    Args:
        signals: Series of signals (-1, 0, 1)
        min_bars: Minimum bars between trades (0 = no cooldown)

    Returns:
        Signals with cooldown applied
    """
    if min_bars <= 0:
        return signals

    result = signals.copy()
    signal_array = signals.values
    last_trade_idx = -min_bars - 1  # Initialize to allow first trade

    for i in range(len(signal_array)):
        if signal_array[i] != 0:  # Non-neutral signal
            if i - last_trade_idx > min_bars:
                # Enough bars have passed, allow trade
                last_trade_idx = i
            else:
                # Still in cooldown, suppress signal
                result.iloc[i] = 0

    return result


def load_and_prepare_data(config: StrategyConfig):
    """Load and prepare data for backtesting."""
    logger.info(f"Loading data from {config.data_source}")

    data_path = Path("data/raw") / config.data_source
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Resample if needed
    if config.resample_interval:
        logger.info(f"Resampling to {config.resample_interval}")
        df = df.resample(config.resample_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'num_trades': 'sum',
        }).dropna()
        logger.info(f"After resampling: {len(df):,} rows")

    return df


def run_backtest(config: StrategyConfig, model_id: str, save_json: bool = True):
    """Run backtest on historical data.

    Args:
        config: Strategy configuration
        model_id: Model ID to backtest
        save_json: If True, save results to JSON file
    """

    # Load data
    df = load_and_prepare_data(config)

    # Create target for the full dataset
    logger.info("Creating targets...")
    target_series = create_3class_target(
        df=df,
        lookahead_bars=config.target_lookahead_bars,
        threshold_pct=config.target_threshold_pct,
    )

    # Initialize feature engine
    logger.info("Initializing feature engine...")
    feature_engine = FeatureEngine(config.features)

    # Compute features
    logger.info("Computing features...")
    df_features = feature_engine.compute_features(df)

    # Align features and targets
    df_clean = df_features.dropna()
    target_clean = target_series.reindex(df_clean.index).dropna()
    df_clean = df_clean.reindex(target_clean.index)

    logger.info(f"After cleaning: {len(df_clean):,} samples")

    # Load model
    logger.info(f"Loading model {model_id}...")
    registry = ModelRegistry()
    metadata = registry.get_model(model_id)

    if metadata is None:
        raise ValueError(f"Model not found: {model_id}")

    model_path = Path(metadata.model_artifact_path)
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Load feature schema
    feature_schema = FeatureSchema.from_dict(metadata.feature_schema)
    logger.info(f"Feature schema: {len(feature_schema.feature_names)} features")

    # Filter validation period
    val_start = config.validation_start
    val_end = config.validation_end

    val_mask = (df_clean['timestamp'] >= val_start) & (df_clean['timestamp'] <= val_end)
    df_val = df_clean[val_mask]
    target_val = target_clean[val_mask]

    logger.info(f"Validation period: {val_start} to {val_end}")
    logger.info(f"Validation samples: {len(df_val):,}")

    # Prepare features (exclude OHLCV)
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']
    feature_cols = [col for col in df_val.columns if col not in ohlcv_cols]
    X_val = df_val[feature_cols]

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.predict(X_val)
    probas = model.predict_proba(X_val)
    confidences = pd.Series(probas.max(axis=1), index=df_val.index)

    # Convert predictions to signals: 0->-1 (short), 1->0 (neutral), 2->1 (long)
    signal_mapping = {0: -1, 1: 0, 2: 1}
    raw_signals = pd.Series([signal_mapping[p] for p in predictions], index=df_val.index)

    # Apply confidence filtering if threshold specified
    confidence_threshold = getattr(config.target, 'confidence_threshold', 0.0)
    if confidence_threshold > 0:
        logger.info(f"Applying confidence threshold: {confidence_threshold:.2f}")
        low_confidence_mask = confidences < confidence_threshold
        signals = raw_signals.copy()
        signals[low_confidence_mask] = 0

        suppressed_count = low_confidence_mask.sum()
        total_signals = (raw_signals != 0).sum()
        logger.info(f"Suppressed {suppressed_count}/{total_signals} signals ({suppressed_count/total_signals*100:.1f}%) due to low confidence")
    else:
        signals = raw_signals

    # Apply trade cooldown if specified
    min_bars_between_trades = config.backtest.min_bars_between_trades
    if min_bars_between_trades > 0:
        logger.info(f"Applying trade cooldown: {min_bars_between_trades} bars minimum between trades")
        signals_before_cooldown = signals.copy()
        signals = apply_trade_cooldown(signals, min_bars_between_trades)

        suppressed_by_cooldown = ((signals_before_cooldown != 0) & (signals == 0)).sum()
        total_signals_pre_cooldown = (signals_before_cooldown != 0).sum()
        logger.info(f"Suppressed {suppressed_by_cooldown}/{total_signals_pre_cooldown} signals ({suppressed_by_cooldown/total_signals_pre_cooldown*100:.1f}%) due to cooldown")

    final_trade_count = (signals != 0).sum()
    logger.info(f"Final signals for backtesting: {final_trade_count}")

    # Create backtest config
    backtest_config = BacktestConfig(
        initial_capital=config.backtest.capital,
        taker_fee=config.backtest.transaction_cost_pct / 100.0,  # Convert % to decimal
        slippage_bps=config.backtest.slippage_bps,
        enable_leverage=config.backtest.leverage > 1,
        max_leverage=float(config.backtest.leverage),
    )

    # Run backtest
    logger.info("Running backtest...")
    backtest_engine = BacktestEngine(backtest_config)

    results = backtest_engine.run(
        df=df_val,
        signals=signals,
        price_column="close",
        confidences=confidences,
    )

    # Display results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"\nModel: {model_id}")
    print(f"Strategy: {config.strategy_name} v{config.version}")
    print(f"Period: {val_start} to {val_end}")
    print(f"Initial Capital: ${backtest_config.initial_capital:,.2f}")
    print(f"Leverage: {backtest_config.max_leverage}x")
    print(f"\nTransaction Costs:")
    print(f"  Fee: {backtest_config.taker_fee * 100:.3f}%")
    print(f"  Slippage: {backtest_config.slippage_bps} bps")
    print(f"\nSignal Filtering:")
    print(f"  Confidence Threshold: {confidence_threshold:.2f}")
    print(f"  Trade Cooldown: {min_bars_between_trades} bars")

    metrics = results
    print(f"\n{'PERFORMANCE METRICS':-^70}")
    print(f"Total Return: {metrics.total_return * 100:.2f}%")
    print(f"Annual Return: {metrics.annual_return * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown * 100:.2f}%")
    print(f"Win Rate: {metrics.win_rate * 100:.2f}%")

    print(f"\n{'TRADING STATISTICS':-^70}")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Winning Trades: {metrics.winning_trades}")
    print(f"Losing Trades: {metrics.losing_trades}")
    print(f"Average Win: ${metrics.avg_win:.2f}")
    print(f"Average Loss: ${metrics.avg_loss:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")

    print(f"\n{'RISK METRICS':-^70}")
    print(f"Volatility: {metrics.volatility * 100:.2f}%")
    print(f"Downside Volatility: {metrics.downside_volatility * 100:.2f}%")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")

    print("="*70)

    # Save to JSON if requested
    if save_json:
        results_dir = Path("results/backtest")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{config.strategy_name}_{config.version}_{timestamp}.json"

        output = {
            "model_id": model_id,
            "config": {
                "strategy_name": config.strategy_name,
                "version": config.version,
                "period": {
                    "start": str(val_start),
                    "end": str(val_end)
                },
                "initial_capital": backtest_config.initial_capital,
                "leverage": backtest_config.max_leverage,
                "transaction_cost_pct": backtest_config.taker_fee * 100,
                "slippage_bps": backtest_config.slippage_bps,
                "confidence_threshold": confidence_threshold,
                "min_bars_between_trades": min_bars_between_trades,
            },
            "metrics": {
                "total_return": float(metrics.total_return),
                "annual_return": float(metrics.annual_return),
                "sharpe_ratio": float(metrics.sharpe_ratio),
                "sortino_ratio": float(metrics.sortino_ratio),
                "max_drawdown": float(metrics.max_drawdown),
                "win_rate": float(metrics.win_rate),
                "total_trades": int(metrics.total_trades),
                "winning_trades": int(metrics.winning_trades),
                "losing_trades": int(metrics.losing_trades),
                "avg_win": float(metrics.avg_win),
                "avg_loss": float(metrics.avg_loss),
                "profit_factor": float(metrics.profit_factor),
                "volatility": float(metrics.volatility),
                "downside_volatility": float(metrics.downside_volatility),
                "calmar_ratio": float(metrics.calmar_ratio),
            }
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run backtest on trained model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config YAML file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID to backtest"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving results to JSON file (default: save to results/backtest/)"
    )

    args = parser.parse_args()

    # Load config
    config = StrategyConfig.from_yaml(args.config)

    # Run backtest
    try:
        results = run_backtest(config, args.model_id, save_json=not args.no_json)
        logger.info("Backtest completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
