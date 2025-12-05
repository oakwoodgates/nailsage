"""Hyperparameter optimization using Optuna."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config.strategy import StrategyConfig
from data.loader import DataLoader
from features.engine import FeatureEngine
from training.targets import create_3class_target, create_binary_target
from training.validation.time_series_split import TimeSeriesSplitter
from utils.logger import get_logger, setup_logger

setup_logger(level=20)
logger = get_logger("optimization")


def get_search_space(model_type: str, num_classes: int) -> Dict[str, Any]:
    """
    Define hyperparameter search space for each model type.

    Args:
        model_type: Type of model (xgboost, lightgbm, random_forest)
        num_classes: Number of classes (2 or 3)

    Returns:
        Dictionary with parameter ranges and distributions
    """
    if model_type == "xgboost":
        return {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float", 0.001, 0.3, "log"),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "min_child_weight": ("int", 1, 10),
            "gamma": ("float", 0, 5),
        }
    elif model_type == "lightgbm":
        return {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float", 0.001, 0.3, "log"),
            "n_estimators": ("int", 50, 500),
            "num_leaves": ("int", 15, 127),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "min_child_samples": ("int", 5, 50),
        }
    elif model_type == "random_forest":
        return {
            "max_depth": ("int", 3, 20),
            "n_estimators": ("int", 50, 300),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", None]),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def suggest_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest hyperparameters based on search space.

    Args:
        trial: Optuna trial object
        search_space: Dictionary defining parameter ranges

    Returns:
        Dictionary of suggested parameters
    """
    params = {}

    for param_name, spec in search_space.items():
        param_type = spec[0]

        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
        elif param_type == "float":
            log = len(spec) > 3 and spec[3] == "log"
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=log)
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, spec[1])

    return params


def create_model(model_type: str, params: Dict[str, Any], num_classes: int):
    """Create model with given parameters."""
    if model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            objective="multi:softmax" if num_classes == 3 else "binary:logistic",
            num_class=num_classes if num_classes == 3 else None,
            random_state=42,
            **params
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="multiclass" if num_classes == 3 else "binary",
            num_class=num_classes if num_classes == 3 else None,
            random_state=42,
            verbose=-1,
            **params
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params
        )


def objective(
    trial: optuna.Trial,
    config: StrategyConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray = None
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial
        config: Strategy configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        sample_weights: Optional sample weights

    Returns:
        Validation score (higher is better)
    """
    # Get search space and suggest parameters
    num_classes = config.target.classes or 3
    search_space = get_search_space(config.model_type_str, num_classes)
    params = suggest_params(trial, search_space)

    # Create and train model
    model = create_model(config.model_type_str, params, num_classes)

    fit_kwargs = {'X': X_train, 'y': y_train}
    if sample_weights is not None:
        fit_kwargs['sample_weight'] = sample_weights
    if config.model_type_str == 'xgboost':
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['verbose'] = False
    elif config.model_type_str == 'lightgbm':
        fit_kwargs['eval_set'] = [(X_val, y_val)]

    model.fit(**fit_kwargs)

    # Evaluate on validation set
    from sklearn.metrics import f1_score
    val_preds = model.predict(X_val)
    score = f1_score(y_val, val_preds, average='macro')

    return score


def optimize_strategy(
    config_path: str,
    n_trials: int = 50,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a strategy.

    Args:
        config_path: Path to strategy configuration
        n_trials: Number of optimization trials
        timeout: Optional timeout in seconds

    Returns:
        Dictionary with best parameters and results
    """
    logger.info(f"Loading configuration from {config_path}")
    config = StrategyConfig.from_yaml(config_path)

    # Load and prepare data
    logger.info(f"Loading data from {config.data_config.data_dir / config.data_source}")
    loader = DataLoader(config.data_config)
    df = loader.load(filename=config.data_source)

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

    # Generate features
    logger.info("Computing features...")
    feature_engine = FeatureEngine(config.feature_config)
    df_features = feature_engine.compute_features(df)

    # Create target
    num_classes = config.target.classes or 3
    logger.info(f"Creating {num_classes}-class target...")

    if num_classes == 2:
        target = create_binary_target(
            df_features,
            lookahead_bars=config.target_lookahead_bars,
            threshold_pct=config.target_threshold_pct
        )
    else:
        target = create_3class_target(
            df_features,
            lookahead_bars=config.target_lookahead_bars,
            threshold_pct=config.target_threshold_pct
        )

    # Clean data
    valid_idx = target.notna() & df_features.notna().all(axis=1)
    df_clean = df_features[valid_idx].copy()
    target_clean = target[valid_idx].copy()

    # Filter by date range
    train_mask = (df_clean['timestamp'] >= config.train_start) & (df_clean['timestamp'] <= config.train_end)
    val_mask = (df_clean['timestamp'] >= config.validation_start) & (df_clean['timestamp'] <= config.validation_end)

    # Prepare features (exclude OHLCV)
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_clean.columns if col not in ohlcv_cols]

    X_train = df_clean[train_mask][feature_cols].values
    y_train = target_clean[train_mask].values
    X_val = df_clean[val_mask][feature_cols].values
    y_val = target_clean[val_mask].values

    logger.info(f"Training samples: {len(X_train):,}, Validation samples: {len(X_val):,}")

    # Calculate sample weights if specified
    sample_weights = None
    if hasattr(config.target, 'class_weights') and config.target.class_weights:
        sample_weights = np.array([config.target.class_weights[label] for label in y_train])
        logger.info(f"Using class weights: {config.target.class_weights}")

    # Create Optuna study
    logger.info(f"Starting optimization with {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=5)
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, X_train, y_train, X_val, y_val, sample_weights),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"\nOptimization complete!")
    logger.info(f"Best F1 score: {best_score:.4f}")
    logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")

    # Save results
    results = {
        "strategy_name": config.strategy_name,
        "version": config.version,
        "model_type": config.model_type_str,
        "n_trials": n_trials,
        "best_score": float(best_score),
        "best_params": best_params,
        "original_params": config.model_params,
        "optimization_time": datetime.now().isoformat(),
    }

    results_dir = Path("results/optimization")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{config.strategy_name}_{config.version}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize strategy hyperparameters")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy configuration file"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds"
    )

    args = parser.parse_args()

    results = optimize_strategy(
        config_path=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best F1 Score: {results['best_score']:.4f}")
    print(f"\nBest Parameters:")
    print(json.dumps(results['best_params'], indent=2))
