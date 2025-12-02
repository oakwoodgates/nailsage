"""Training script for momentum classifier strategy."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config.strategy import StrategyConfig
from data.loader import DataLoader
from data.validator import DataValidator
from features.engine import FeatureEngine
from models import ModelRegistry, create_model_metadata
from targets.classification import create_3class_target, create_binary_target
from validation.time_series_split import TimeSeriesSplitter
from utils.logger import get_logger, setup_logger

# Setup logging
setup_logger(level=20)  # INFO
logger = get_logger("training")


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


def train_strategy(config_path: str):
    """
    Train momentum classifier strategy.

    Args:
        config_path: Path to strategy configuration YAML
    """
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = StrategyConfig.from_yaml(config_path)

    # Load and validate data
    logger.info(f"Loading data from {config.data_config.data_dir / config.data_source}")
    loader = DataLoader(config.data_config)
    df = loader.load(filename=config.data_source)

    logger.info(f"Loaded {len(df):,} rows")

    # Validate data quality
    validator = DataValidator(config.data_config)
    quality_report = validator.validate(df)
    logger.info(f"Data quality: {quality_report.quality_score:.2%}")

    if not quality_report.is_valid:
        logger.warning("Data quality check found issues (may be false positives for non-1m intervals)")
        errors = quality_report.get_errors()
        # Only show first 5 errors to avoid log spam
        for error in errors[:5]:
            logger.warning(f"  - {error.message}")
        if len(errors) > 5:
            logger.warning(f"  ... and {len(errors) - 5} more similar issues")
        logger.info("Continuing with training despite validation warnings...")

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

    logger.info(f"Generated {len(df_features.columns)} features")

    # Create target variable based on number of classes
    num_classes = config.target.classes or 3
    logger.info(f"Creating {num_classes}-class target variable...")

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

    # Remove rows with NaN (from lookahead and feature computation)
    valid_idx = target.notna() & df_features.notna().all(axis=1)
    df_clean = df_features[valid_idx].copy()
    target_clean = target[valid_idx].copy()

    logger.info(f"Clean dataset: {len(df_clean):,} rows")
    logger.info(f"Target distribution: {target_clean.value_counts().to_dict()}")

    # Filter by date range
    train_mask = (df_clean['timestamp'] >= config.train_start) & (df_clean['timestamp'] <= config.train_end)
    val_mask = (df_clean['timestamp'] >= config.validation_start) & (df_clean['timestamp'] <= config.validation_end)

    # Exclude OHLCV columns to prevent data leakage (predicting close from close)
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_clean.columns if col not in ohlcv_cols]
    logger.info(f"Using {len(feature_cols)} feature columns (excluded OHLCV)")

    X_train = df_clean[train_mask][feature_cols]
    y_train = target_clean[train_mask]
    X_val = df_clean[val_mask][feature_cols]
    y_val = target_clean[val_mask]

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Validation set: {len(X_val):,} samples")

    # Calculate sample weights if class_weights specified in config
    sample_weights = None
    if hasattr(config.target, 'class_weights') and config.target.class_weights:
        class_weights = config.target.class_weights
        sample_weights = np.array([class_weights[label] for label in y_train])
        logger.info(f"Using class weights: {class_weights}")
        logger.info(f"Sample weight distribution: min={sample_weights.min():.1f}, max={sample_weights.max():.1f}, mean={sample_weights.mean():.1f}")

        # Log class distribution with weights
        for class_label in sorted(class_weights.keys()):
            count = (y_train == class_label).sum()
            weight = class_weights[class_label]
            logger.info(f"  Class {class_label}: {count:,} samples, weight={weight:.1f}, effective={count*weight:.1f}")
    else:
        logger.info("No class weights specified - using default equal weighting")

    # Train model
    model_type = config.model_type_str
    logger.info(f"Training {model_type} model...")

    model = create_model(model_type, config.model_params)

    # Fit with optional sample weights
    # XGBoost and LightGBM support eval_set, sklearn models don't
    fit_kwargs = {
        'X': X_train,
        'y': y_train,
    }

    # Add eval_set for models that support it
    if model_type == 'xgboost':
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['verbose'] = False
    elif model_type == 'lightgbm':
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        # LightGBM uses 'verbose' in constructor params, not in fit()

    # Add sample weights if specified
    if sample_weights is not None:
        fit_kwargs['sample_weight'] = sample_weights

    model.fit(**fit_kwargs)

    # Evaluate
    logger.info("Evaluating model...")
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    logger.info("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds, target_names=['Short', 'Neutral', 'Long']))

    # Save model
    model_filename = f"models/trained/{config.strategy_name}_{config.version}_temp.joblib"
    Path(model_filename).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_filename)
    logger.info(f"Saved model to {model_filename}")

    # Create model metadata
    logger.info("Creating model metadata...")

    metadata = create_model_metadata(
        strategy_name=config.strategy_name,
        strategy_timeframe=config.strategy_timeframe,
        version=config.version,
        training_dataset_path=str(config.data_config.data_dir / config.data_source),
        training_date_range=(config.train_start, config.train_end),
        validation_date_range=(config.validation_start, config.validation_end),
        model_type=config.model_type_str,
        feature_config=config.feature_config.model_dump(),
        model_config=config.model_params,
        target_config={
            "type": "classification",
            "classes": 3,
            "lookahead_bars": config.target_lookahead_bars,
            "threshold_pct": config.target_threshold_pct,
            "class_weights": config.target.class_weights if hasattr(config.target, 'class_weights') else None
        },
        validation_metrics={
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "val_long_precision": float(0.0),  # TODO: Extract from classification_report
            "val_short_precision": float(0.0),
        },
        model_artifact_path=model_filename,
        notes=f"First momentum classifier - {config.description}"
    )

    # Register model
    registry = ModelRegistry()
    registered_metadata = registry.register_model(
        model_artifact_path=Path(model_filename),
        metadata=metadata
    )

    logger.info(f"\n{'='*70}")
    logger.info(f"Model registered successfully!")
    logger.info(f"{'='*70}")
    logger.info(f"Model ID: {registered_metadata.model_id}")
    logger.info(f"Config Hash: {registered_metadata.get_config_hash()}")
    logger.info(f"Training Time: {registered_metadata.get_training_timestamp()}")
    logger.info(f"Artifact: {registered_metadata.model_artifact_path}")
    logger.info(f"{'='*70}")

    return registered_metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train momentum classifier strategy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/strategies/momentum_classifier_v1.yaml",
        help="Path to strategy configuration file"
    )

    args = parser.parse_args()

    trained_metadata = train_strategy(args.config)

    if trained_metadata:
        print("\n" + trained_metadata.summary())
