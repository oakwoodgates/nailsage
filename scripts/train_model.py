"""Generic model training script with optional walk-forward validation.

Usage:
    # Train and validate (default - saves results to JSON)
    python scripts/train_model.py --config configs/strategies/dev_scalper_1m_v1.yaml

    # Train only (no validation)
    python scripts/train_model.py --config configs/strategies/dev_scalper_1m_v1.yaml --train-only

    # Train and validate without saving JSON
    python scripts/train_model.py --config configs/strategies/dev_scalper_1m_v1.yaml --no-json
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config.backtest import BacktestConfig
from config.strategy import StrategyConfig
from features.engine import FeatureEngine
from models import ModelRegistry, create_model_metadata
from models.feature_schema import FeatureSchema
from targets.classification import create_3class_target, create_binary_target
from validation.backtest import BacktestEngine
from validation.time_series_split import TimeSeriesSplitter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(model_type: str, params: dict):
    """
    Factory function to create ML model based on type.

    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'extra_trees')
        params: Model hyperparameters

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**params)

    elif model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    elif model_type == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**params)

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: xgboost, lightgbm, random_forest, extra_trees"
        )


def apply_trade_cooldown(signals: np.ndarray, min_bars: int) -> np.ndarray:
    """
    Apply cooldown between trades - suppress signals within min_bars of last trade.

    Args:
        signals: Array of signals (-1, 0, 1)
        min_bars: Minimum bars between trades (0 = no cooldown)

    Returns:
        Signals with cooldown applied
    """
    if min_bars <= 0:
        return signals

    result = signals.copy()
    last_trade_idx = -min_bars - 1  # Initialize to allow first trade

    for i in range(len(signals)):
        if signals[i] != 0:  # Non-neutral signal
            if i - last_trade_idx > min_bars:
                # Enough bars have passed, allow trade
                last_trade_idx = i
            else:
                # Still in cooldown, suppress signal
                result[i] = 0

    return result


def convert_predictions_to_signals(predictions, probabilities, confidence_threshold, num_classes):
    """
    Convert model predictions to trading signals with optional confidence filtering.

    Args:
        predictions: Model class predictions
        probabilities: Model prediction probabilities (n_samples x n_classes)
        confidence_threshold: Minimum confidence to generate signal (0.0-1.0)
        num_classes: Number of classes (2 or 3)

    Returns:
        signals: Array of trading signals (-1=short, 0=neutral, 1=long)
    """
    signals = np.zeros_like(predictions, dtype=int)

    # Map predictions to signals based on number of classes
    if num_classes == 2:
        # Binary: 0=short, 1=long
        signals[predictions == 1] = 1   # Long
        signals[predictions == 0] = -1  # Short
    else:
        # 3-class: 0=short, 1=neutral, 2=long
        signals[predictions == 2] = 1   # Long
        signals[predictions == 0] = -1  # Short
        # predictions == 1 stays 0 (neutral)

    # Apply confidence filtering
    if probabilities is not None and confidence_threshold > 0:
        max_probs = probabilities.max(axis=1)
        low_confidence_mask = max_probs < confidence_threshold
        signals[low_confidence_mask] = 0

    return signals


def load_and_prepare_data(config: StrategyConfig):
    """Load and prepare data for training."""
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


def train_and_validate(config: StrategyConfig, train_only: bool = False, save_json: bool = True):
    """
    Train model with optional walk-forward validation.

    Args:
        config: Strategy configuration
        train_only: If True, skip validation
        save_json: If True, save results to JSON file

    Returns:
        Tuple of (model_metadata, validation_results)
    """
    # Load data
    df = load_and_prepare_data(config)

    # Create target
    num_classes = config.target.classes or 3
    logger.info(f"Creating {num_classes}-class target variable...")

    if num_classes == 2:
        target_series = create_binary_target(
            df=df,
            lookahead_bars=config.target_lookahead_bars,
            threshold_pct=config.target_threshold_pct
        )
    else:
        target_series = create_3class_target(
            df=df,
            lookahead_bars=config.target_lookahead_bars,
            threshold_pct=config.target_threshold_pct
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
    logger.info(f"Target distribution: {target_clean.value_counts().to_dict()}")

    # Filter training period
    train_mask = (df_clean['timestamp'] >= config.train_start) & (df_clean['timestamp'] <= config.train_end)
    df_train = df_clean[train_mask]
    target_train = target_clean[train_mask]

    logger.info(f"Training period: {config.train_start} to {config.train_end}")
    logger.info(f"Training samples: {len(df_train):,}")

    # Prepare features (exclude OHLCV)
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_trades']
    feature_cols = [col for col in df_train.columns if col not in ohlcv_cols]
    X_train = df_train[feature_cols]
    y_train = target_train

    logger.info(f"Using {len(feature_cols)} feature columns")

    # Calculate sample weights if specified
    sample_weights = None
    if hasattr(config.target, 'class_weights') and config.target.class_weights:
        class_weights = config.target.class_weights
        sample_weights = np.array([class_weights[label] for label in y_train])
        logger.info(f"Using class weights: {class_weights}")

        # Log class distribution with weights
        for class_label in sorted(class_weights.keys()):
            count = (y_train == class_label).sum()
            weight = class_weights[class_label]
            logger.info(f"  Class {class_label}: {count:,} samples, weight={weight:.1f}, effective={count*weight:.1f}")

    # Train model
    model_type = config.model_type_str
    logger.info(f"Training {model_type} model...")

    model = create_model(model_type, config.model_params)

    # Prepare fit arguments
    fit_kwargs = {
        'X': X_train,
        'y': y_train,
    }

    if sample_weights is not None:
        fit_kwargs['sample_weight'] = sample_weights

    # Train
    model.fit(**fit_kwargs)

    # Evaluate on training set
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    # Save model (temporary, will be registered with proper name later)
    model_filename = f"models/trained/{config.strategy_name}_{config.version}_temp.joblib"
    Path(model_filename).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_filename)
    logger.info(f"Saved model to {model_filename}")

    # Create feature schema
    feature_schema = FeatureSchema(
        feature_names=feature_cols,
        ohlcv_columns=ohlcv_cols
    )

    validation_results = None

    # Run validation if requested
    if not train_only:
        logger.info("\n" + "="*70)
        logger.info("RUNNING WALK-FORWARD VALIDATION")
        logger.info("="*70)

        # Filter validation period
        val_mask = (df_clean['timestamp'] >= config.validation_start) & (df_clean['timestamp'] <= config.validation_end)
        df_val = df_clean[val_mask]
        target_val = target_clean[val_mask]

        logger.info(f"Validation period: {config.validation_start} to {config.validation_end}")
        logger.info(f"Validation samples: {len(df_val):,}")

        # Setup walk-forward validation
        validation_config = config.validation or {}
        method = validation_config.method if hasattr(validation_config, 'method') else 'walk_forward'
        n_splits = validation_config.n_splits if hasattr(validation_config, 'n_splits') else 4
        expanding_window = validation_config.expanding_window if hasattr(validation_config, 'expanding_window') else True

        logger.info(f"Validation method: {method}")
        logger.info(f"Number of splits: {n_splits}")
        logger.info(f"Window type: {'expanding' if expanding_window else 'sliding'}")

        # Create time series splitter
        splitter = TimeSeriesSplitter(
            n_splits=n_splits,
            expanding_window=expanding_window,
            test_size=1.0 / (n_splits + 1),
        )

        # Get splits
        splits = splitter.split(df_val, timestamp_column='timestamp')
        logger.info(f"Generated {len(splits)} validation splits")

        # Run validation on each split
        split_results = []
        for i, split_info in enumerate(splits, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Split {i}/{len(splits)}")
            logger.info(f"{'='*70}")

            # Get train/val indices for this split
            train_idx, val_idx = splitter.get_train_val_indices(df_val, split_info)

            X_split_val = df_val.iloc[val_idx][feature_cols]
            y_split_val = target_val.iloc[val_idx]

            logger.info(f"Split validation samples: {len(X_split_val):,}")

            # Generate predictions
            predictions = model.predict(X_split_val)
            probabilities = model.predict_proba(X_split_val)

            # Calculate accuracy
            val_acc = accuracy_score(y_split_val, predictions)
            logger.info(f"Validation accuracy: {val_acc:.4f}")

            # Convert to signals with confidence filtering
            confidence_threshold = getattr(config.target, 'confidence_threshold', 0.0)
            signals = convert_predictions_to_signals(
                predictions, probabilities, confidence_threshold, num_classes
            )

            # Apply trade cooldown
            min_bars_between_trades = config.backtest.min_bars_between_trades if config.backtest else 0
            if min_bars_between_trades > 0:
                signals = apply_trade_cooldown(signals, min_bars_between_trades)

            # Run backtest
            if config.backtest:
                logger.info("Running backtest on split...")

                backtest_config = BacktestConfig(
                    initial_capital=config.backtest.capital,
                    taker_fee=config.backtest.transaction_cost_pct / 100.0,
                    slippage_bps=config.backtest.slippage_bps,
                    enable_leverage=config.backtest.leverage > 1,
                    max_leverage=float(config.backtest.leverage),
                )

                backtest_engine = BacktestEngine(backtest_config)
                backtest_metrics = backtest_engine.run(
                    df=df_val.iloc[val_idx],
                    signals=pd.Series(signals, index=df_val.iloc[val_idx].index),
                    price_column="close",
                    confidences=pd.Series(probabilities.max(axis=1), index=df_val.iloc[val_idx].index),
                )

                logger.info(f"Split {i} Results:")
                logger.info(f"  Accuracy: {val_acc:.4f}")
                logger.info(f"  Total Return: {backtest_metrics.total_return * 100:.2f}%")
                logger.info(f"  Sharpe Ratio: {backtest_metrics.sharpe_ratio:.2f}")
                logger.info(f"  Max Drawdown: {backtest_metrics.max_drawdown * 100:.2f}%")
                logger.info(f"  Win Rate: {backtest_metrics.win_rate * 100:.2f}%")
                logger.info(f"  Total Trades: {backtest_metrics.total_trades}")

                split_results.append({
                    'split': i,
                    'period': {
                        'start': str(split_info.val_start),
                        'end': str(split_info.val_end)
                    },
                    'val_metrics': {
                        'accuracy': float(val_acc),
                    },
                    'backtest_metrics': {
                        'total_return': float(backtest_metrics.total_return),
                        'annual_return': float(backtest_metrics.annual_return),
                        'sharpe_ratio': float(backtest_metrics.sharpe_ratio),
                        'sortino_ratio': float(backtest_metrics.sortino_ratio),
                        'max_drawdown': float(backtest_metrics.max_drawdown),
                        'win_rate': float(backtest_metrics.win_rate),
                        'total_trades': int(backtest_metrics.total_trades),
                        'profit_factor': float(backtest_metrics.profit_factor),
                    }
                })

        # Calculate aggregate metrics
        if split_results:
            logger.info(f"\n{'='*70}")
            logger.info("AGGREGATE VALIDATION RESULTS")
            logger.info(f"{'='*70}")

            avg_val_acc = np.mean([r['val_metrics']['accuracy'] for r in split_results])
            avg_return = np.mean([r['backtest_metrics']['total_return'] for r in split_results])
            avg_sharpe = np.mean([r['backtest_metrics']['sharpe_ratio'] for r in split_results])
            avg_max_dd = np.mean([r['backtest_metrics']['max_drawdown'] for r in split_results])
            avg_win_rate = np.mean([r['backtest_metrics']['win_rate'] for r in split_results])
            total_trades = sum([r['backtest_metrics']['total_trades'] for r in split_results])

            positive_splits = sum(1 for r in split_results if r['backtest_metrics']['total_return'] > 0)
            consistency_pct = positive_splits / len(split_results)

            logger.info(f"Average Validation Accuracy: {avg_val_acc:.4f}")
            logger.info(f"Average Total Return: {avg_return * 100:.2f}%")
            logger.info(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
            logger.info(f"Average Max Drawdown: {avg_max_dd * 100:.2f}%")
            logger.info(f"Average Win Rate: {avg_win_rate * 100:.2f}%")
            logger.info(f"Total Trades (all splits): {total_trades}")
            logger.info(f"Consistency: {consistency_pct * 100:.1f}% ({positive_splits}/{len(split_results)} splits profitable)")
            logger.info(f"{'='*70}")

            validation_results = {
                'n_splits': len(split_results),
                'splits': split_results,
                'aggregate': {
                    'avg_val_accuracy': float(avg_val_acc),
                    'avg_total_return': float(avg_return),
                    'avg_sharpe_ratio': float(avg_sharpe),
                    'avg_max_drawdown': float(avg_max_dd),
                    'avg_win_rate': float(avg_win_rate),
                    'total_trades': int(total_trades),
                    'positive_splits': int(positive_splits),
                    'consistency_pct': float(consistency_pct),
                }
            }

    # Create model metadata
    logger.info("\nCreating model metadata...")

    metadata = create_model_metadata(
        strategy_name=config.strategy_name,
        strategy_timeframe=config.strategy_timeframe,
        version=config.version,
        training_dataset_path=str(Path("data/raw") / config.data_source),
        training_date_range=(config.train_start, config.train_end),
        validation_date_range=(config.validation_start, config.validation_end),
        model_type=config.model_type_str,
        feature_config=config.feature_config.model_dump(),
        model_config=config.model_params,
        target_config={
            "type": "classification",
            "classes": num_classes,
            "lookahead_bars": config.target_lookahead_bars,
            "threshold_pct": config.target_threshold_pct,
            "class_weights": config.target.class_weights if hasattr(config.target, 'class_weights') else None,
            "confidence_threshold": getattr(config.target, 'confidence_threshold', 0.0),
        },
        validation_metrics={
            "train_accuracy": float(train_acc),
            "avg_val_accuracy": float(validation_results['aggregate']['avg_val_accuracy']) if validation_results else 0.0,
        },
        model_artifact_path=model_filename,
        feature_schema=feature_schema.to_dict(),
        notes=f"Generic training script - {config.description or 'No description'}"
    )

    # Register model
    registry = ModelRegistry()
    registered_metadata = registry.register_model(
        model_artifact_path=Path(model_filename),
        metadata=metadata
    )

    logger.info(f"\n{'='*70}")
    logger.info("MODEL REGISTERED SUCCESSFULLY")
    logger.info(f"{'='*70}")
    logger.info(f"Model ID: {registered_metadata.model_id}")
    logger.info(f"Config Hash: {registered_metadata.get_config_hash()}")
    logger.info(f"Training Time: {registered_metadata.get_training_timestamp()}")
    logger.info(f"Artifact: {registered_metadata.model_artifact_path}")
    logger.info(f"{'='*70}")

    # Save results to JSON if requested
    if save_json and validation_results:
        results_dir = Path("results/training")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{config.strategy_name}_{config.version}_{timestamp}.json"

        output = {
            'model_id': registered_metadata.model_id,
            'config': {
                'strategy_name': config.strategy_name,
                'version': config.version,
                'training_period': {
                    'start': config.train_start,
                    'end': config.train_end
                },
                'validation_period': {
                    'start': config.validation_start,
                    'end': config.validation_end
                }
            },
            'training_metrics': {
                'train_accuracy': float(train_acc),
            },
            'validation_results': validation_results
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

    return registered_metadata, validation_results


def main():
    parser = argparse.ArgumentParser(description="Train model with optional validation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config YAML file"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip validation, only train model"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving results to JSON file (default: save to results/training/)"
    )

    args = parser.parse_args()

    # Load config
    config = StrategyConfig.from_yaml(args.config)

    # Train and validate
    try:
        metadata, validation_results = train_and_validate(
            config,
            train_only=args.train_only,
            save_json=not args.no_json
        )
        logger.info("\nTraining completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
