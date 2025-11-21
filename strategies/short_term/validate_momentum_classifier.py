"""Walk-forward validation and backtesting for momentum classifier strategy."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from config.backtest import BacktestConfig
from config.strategy import StrategyConfig
from data.loader import DataLoader
from features.engine import FeatureEngine
from models import ModelRegistry
from targets.classification import create_3class_target
from validation.backtest import BacktestEngine
from validation.time_series_split import TimeSeriesSplitter
from validation.walk_forward import WalkForwardValidator
from utils.logger import get_logger, setup_logger


def create_model(model_type: str, params: dict):
    """Factory function to create ML model based on type."""
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
        raise ValueError(f"Unsupported model type: {model_type}")

# Setup logging
setup_logger(level=20)  # INFO
logger = get_logger("validation")


def convert_predictions_to_signals(
    predictions: np.ndarray,
    probabilities: np.ndarray = None,
    confidence_threshold: float = 0.0
) -> np.ndarray:
    """
    Convert 3-class predictions to trading signals with optional confidence filtering.

    Args:
        predictions: Array of class predictions (0, 1, 2)
        probabilities: Array of prediction probabilities (n_samples, n_classes)
        confidence_threshold: Minimum confidence to generate signal (0.0-1.0)

    Returns:
        Array of signals: -1 (short), 0 (neutral), 1 (long)
    """
    signals = np.zeros_like(predictions, dtype=int)
    signals[predictions == 2] = 1   # Long
    signals[predictions == 0] = -1  # Short
    # predictions == 1 stays 0 (neutral)

    # Filter by confidence if threshold is set
    if probabilities is not None and confidence_threshold > 0:
        max_probs = probabilities.max(axis=1)
        low_confidence_mask = max_probs < confidence_threshold
        signals[low_confidence_mask] = 0  # Set to neutral if low confidence

    return signals


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    # Class distribution
    pred_dist = pd.Series(y_pred).value_counts(normalize=True).to_dict()

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'matthews_correlation': mcc,
        'precision_short': precision[0],
        'precision_neutral': precision[1],
        'precision_long': precision[2],
        'recall_short': recall[0],
        'recall_neutral': recall[1],
        'recall_long': recall[2],
        'f1_short': f1[0],
        'f1_neutral': f1[1],
        'f1_long': f1[2],
        'support_short': int(support[0]),
        'support_neutral': int(support[1]),
        'support_long': int(support[2]),
        'pred_pct_short': pred_dist.get(0, 0.0),
        'pred_pct_neutral': pred_dist.get(1, 0.0),
        'pred_pct_long': pred_dist.get(2, 0.0),
    }


def validate_strategy(
    config_path: str,
    model_id: str = None,
    n_splits: int = 4,
    save_results: bool = True,
    retrain: bool = False
):
    """
    Run walk-forward validation and backtesting on momentum classifier.

    Args:
        config_path: Path to strategy configuration YAML
        model_id: Model ID to load from registry (if None, uses latest). Ignored if retrain=True.
        n_splits: Number of walk-forward splits
        save_results: Whether to save results to file
        retrain: If True, retrain model on each split (true walk-forward). If False, use pre-trained model.
    """
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = StrategyConfig.from_yaml(config_path)

    # Load model from registry (only if not retraining)
    model = None
    model_metadata = None
    if not retrain:
        registry = ModelRegistry()
        if model_id:
            logger.info(f"Loading model {model_id} from registry")
            model_metadata = registry.get_model(model_id)
            if model_metadata is None:
                raise ValueError(f"Model not found: {model_id}")
        else:
            logger.info("Loading latest model from registry")
            models = registry.find_models(strategy_name=config.strategy_name)
            if not models:
                raise ValueError(f"No models found for strategy {config.strategy_name}")
            # Sort by training time to get latest
            models.sort(key=lambda m: m.trained_at, reverse=True)
            model_metadata = models[0]
            model_id = model_metadata.model_id

        logger.info(f"Using model: {model_id}")
        model = joblib.load(model_metadata.model_artifact_path)
    else:
        logger.info("Retrain mode: will train fresh model on each split")
        model_id = "retrained_per_split"

    # Load and validate data
    logger.info(f"Loading data from {config.data_config.data_dir / config.data_source}")
    loader = DataLoader(config.data_config)
    df = loader.load(filename=config.data_source)

    logger.info(f"Loaded {len(df):,} rows")

    # Resample if needed
    if config.resample_interval:
        logger.info(f"Resampling from 1m to {config.resample_interval}")
        df = df.set_index('timestamp').resample(config.resample_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        logger.info(f"After resampling: {len(df):,} rows")

    # Generate features
    logger.info("Computing features...")
    feature_engine = FeatureEngine(config.feature_config)
    df_features = feature_engine.compute_features(df)

    logger.info(f"Generated {len(df_features.columns)} total columns")

    # Create target variable
    logger.info("Creating target variable...")
    target = create_3class_target(
        df_features,
        lookahead_bars=config.target_lookahead_bars,
        threshold_pct=config.target_threshold_pct
    )

    # Remove rows with NaN
    valid_idx = target.notna() & df_features.notna().all(axis=1)
    df_clean = df_features[valid_idx].copy()
    target_clean = target[valid_idx].copy()

    logger.info(f"Clean dataset: {len(df_clean):,} rows")
    logger.info(f"Target distribution: {target_clean.value_counts().to_dict()}")

    # Prepare feature columns - exclude OHLCV to prevent data leakage
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_clean.columns if col not in ohlcv_cols]

    logger.info(f"Using {len(feature_cols)} feature columns")

    # Setup time series splitter
    logger.info(f"Setting up {n_splits}-fold walk-forward validation")
    splitter = TimeSeriesSplitter(
        n_splits=n_splits,
        test_size=0.2,
        gap_bars=0,
        expanding_window=True
    )

    # Setup backtest config
    backtest_config = BacktestConfig(
        initial_capital=10000.0,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_bps=2.0,
        enable_leverage=config.backtest.leverage > 1 if config.backtest else False,
        max_leverage=float(config.backtest.leverage) if config.backtest else 1.0,
    )

    logger.info(f"Backtest config: {backtest_config.initial_capital} capital, "
                f"{backtest_config.taker_fee*100:.2f}% taker fee, "
                f"{backtest_config.slippage_bps} bps slippage")

    # Run walk-forward validation
    logger.info("Starting walk-forward validation...")
    logger.info("=" * 80)

    results = []
    splits = list(splitter.split(df_clean, timestamp_column='timestamp'))

    for i, split_info in enumerate(splits, 1):
        logger.info(f"\nSplit {i}/{len(splits)}")
        logger.info(f"  Train: {split_info.train_start} to {split_info.train_end}")
        logger.info(f"  Val:   {split_info.val_start} to {split_info.val_end}")

        # Get indices
        train_idx, val_idx = splitter.get_train_val_indices(df_clean, split_info)

        # Prepare data
        X_train = df_clean.iloc[train_idx][feature_cols]
        y_train = target_clean.iloc[train_idx]
        X_val = df_clean.iloc[val_idx][feature_cols]
        y_val = target_clean.iloc[val_idx]
        df_val = df_clean.iloc[val_idx]

        logger.info(f"  Train samples: {len(X_train):,}")
        logger.info(f"  Val samples: {len(X_val):,}")

        # Retrain model on this split's training data if retrain mode
        if retrain:
            logger.info(f"  Training fresh {config.model_type_str} model on split {i}...")
            model = create_model(config.model_type_str, config.model_params)

            # Calculate sample weights if class_weights specified
            sample_weights = None
            if hasattr(config.target, 'class_weights') and config.target.class_weights:
                sample_weights = np.array([config.target.class_weights[label] for label in y_train])

            # Fit model
            fit_kwargs = {'X': X_train, 'y': y_train}
            if sample_weights is not None:
                fit_kwargs['sample_weight'] = sample_weights
            if config.model_type_str == 'xgboost':
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
            elif config.model_type_str == 'lightgbm':
                fit_kwargs['eval_set'] = [(X_val, y_val)]

            model.fit(**fit_kwargs)

        # Generate predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        # Get prediction probabilities for confidence filtering
        val_probs = None
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(X_val)

        # Get confidence threshold from config
        confidence_threshold = getattr(config.target, 'confidence_threshold', 0.0)

        # Calculate classification metrics
        train_metrics = calculate_classification_metrics(y_train.values, train_preds)
        val_metrics = calculate_classification_metrics(y_val.values, val_preds)

        logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}, MCC: {train_metrics['matthews_correlation']:.4f}")
        logger.info(f"  Val   Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}, MCC: {val_metrics['matthews_correlation']:.4f}")
        logger.info(f"  Val predictions: {val_metrics['pred_pct_short']:.1%} short, {val_metrics['pred_pct_neutral']:.1%} neutral, {val_metrics['pred_pct_long']:.1%} long")

        # Convert to signals with confidence filtering
        val_signals = convert_predictions_to_signals(val_preds, val_probs, confidence_threshold)
        if confidence_threshold > 0:
            filtered_pct = (val_signals == 0).sum() / len(val_signals)
            logger.info(f"  Confidence threshold: {confidence_threshold:.0%}, filtered to neutral: {filtered_pct:.1%}")

        # Run backtest
        backtest_engine = BacktestEngine(backtest_config)
        backtest_metrics = backtest_engine.run(
            df=df_val,
            signals=pd.Series(val_signals, index=df_val.index),
            price_column='close'
        )

        logger.info(f"  Backtest:")
        logger.info(f"    Total Return: {backtest_metrics.total_return:.2%}")
        logger.info(f"    Sharpe Ratio: {backtest_metrics.sharpe_ratio:.2f}")
        logger.info(f"    Max Drawdown: {backtest_metrics.max_drawdown:.2%}")
        logger.info(f"    Win Rate: {backtest_metrics.win_rate:.2%}")
        logger.info(f"    Total Trades: {backtest_metrics.total_trades}")

        # Store results
        results.append({
            'split': i,
            'train_start': str(split_info.train_start),
            'train_end': str(split_info.train_end),
            'val_start': str(split_info.val_start),
            'val_end': str(split_info.val_end),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'backtest_metrics': {
                'total_return': float(backtest_metrics.total_return),
                'annual_return': float(backtest_metrics.annual_return),
                'sharpe_ratio': float(backtest_metrics.sharpe_ratio),
                'sortino_ratio': float(backtest_metrics.sortino_ratio),
                'calmar_ratio': float(backtest_metrics.calmar_ratio),
                'max_drawdown': float(backtest_metrics.max_drawdown),
                'max_drawdown_duration': int(backtest_metrics.max_drawdown_duration) if backtest_metrics.max_drawdown_duration is not None else 0,
                'volatility': float(backtest_metrics.volatility),
                'win_rate': float(backtest_metrics.win_rate),
                'profit_factor': float(backtest_metrics.profit_factor) if backtest_metrics.profit_factor is not None else 0.0,
                'total_trades': int(backtest_metrics.total_trades),
                'winning_trades': int(backtest_metrics.winning_trades),
                'losing_trades': int(backtest_metrics.losing_trades),
            }
        })

    # Calculate aggregate metrics
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE RESULTS ACROSS ALL SPLITS")
    logger.info("=" * 80)

    # Classification metrics
    avg_val_acc = np.mean([r['val_metrics']['accuracy'] for r in results])
    avg_val_f1 = np.mean([r['val_metrics']['f1_macro'] for r in results])
    avg_val_mcc = np.mean([r['val_metrics']['matthews_correlation'] for r in results])

    logger.info(f"\nClassification Metrics (Validation):")
    logger.info(f"  Average Accuracy: {avg_val_acc:.4f}")
    logger.info(f"  Average F1 (macro): {avg_val_f1:.4f}")
    logger.info(f"  Average MCC: {avg_val_mcc:.4f}")

    # Backtest metrics
    avg_return = np.mean([r['backtest_metrics']['total_return'] for r in results])
    avg_sharpe = np.mean([r['backtest_metrics']['sharpe_ratio'] for r in results])
    avg_drawdown = np.mean([r['backtest_metrics']['max_drawdown'] for r in results])
    avg_win_rate = np.mean([r['backtest_metrics']['win_rate'] for r in results])
    total_trades = sum([r['backtest_metrics']['total_trades'] for r in results])

    logger.info(f"\nBacktest Metrics:")
    logger.info(f"  Average Return: {avg_return:.2%}")
    logger.info(f"  Average Sharpe: {avg_sharpe:.2f}")
    logger.info(f"  Average Max Drawdown: {avg_drawdown:.2%}")
    logger.info(f"  Average Win Rate: {avg_win_rate:.2%}")
    logger.info(f"  Total Trades (all splits): {total_trades}")

    # Consistency check
    positive_splits = sum(1 for r in results if r['backtest_metrics']['total_return'] > 0)
    consistency_pct = positive_splits / len(results)

    logger.info(f"\nConsistency:")
    logger.info(f"  Positive splits: {positive_splits}/{len(results)} ({consistency_pct:.1%})")

    # Save results
    if save_results:
        results_dir = Path("results/walk_forward_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{config.strategy_name}_{config.version}_{timestamp}.json"

        output = {
            'model_id': model_id,
            'config_path': config_path,
            'n_splits': n_splits,
            'results': results,
            'aggregate': {
                'avg_val_accuracy': float(avg_val_acc),
                'avg_val_f1_macro': float(avg_val_f1),
                'avg_val_mcc': float(avg_val_mcc),
                'avg_return': float(avg_return),
                'avg_sharpe': float(avg_sharpe),
                'avg_max_drawdown': float(avg_drawdown),
                'avg_win_rate': float(avg_win_rate),
                'total_trades': int(total_trades),
                'consistency_pct': float(consistency_pct),
            }
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate momentum classifier strategy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/strategies/momentum_classifier_v1.yaml",
        help="Path to strategy configuration file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID to validate (default: latest for strategy)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of walk-forward splits"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain model on each split (true walk-forward validation)"
    )

    args = parser.parse_args()

    results = validate_strategy(
        config_path=args.config,
        model_id=args.model_id,
        n_splits=args.n_splits,
        save_results=True,
        retrain=args.retrain
    )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
