# Strategy Implementation Guide

> **⚠️ DEPRECATED**: This guide uses the old strategy-specific training approach. For the new generic, configuration-driven training architecture, see [MODEL_TRAINING.md](MODEL_TRAINING.md).
>
> **What changed**: We've moved from strategy-specific scripts (`strategies/short_term/train_*.py`) to generic scripts (`scripts/train_model.py`, `scripts/validate_model.py`, `scripts/run_backtest.py`) that work with any strategy configuration.
>
> **Why**: This architecture scales better to 100s of models - 1 bug fix benefits all strategies, no code duplication, and configuration-driven behavior.
>
> **Migration**: See [MODEL_TRAINING.md](MODEL_TRAINING.md) for the new workflow.

---

Complete guide for implementing your first ML trading strategy in NailSage (LEGACY APPROACH).

## 📋 Prerequisites

Before implementing a strategy, ensure you have:

1. ✅ **Data with metadata**: Run `python data/generate_metadata.py --dir data/raw`
2. ✅ **Dependencies installed**: `pip install xgboost lightgbm scikit-learn pandas numpy`
3. ✅ **Tests passing**: `pytest tests/unit/ -v` shows 21/21 passing
4. ✅ **Imports verified**: `python scripts/verify_imports.py` shows all PASS

## 🎯 Strategy Overview: First Implementation

**Goal**: Build a momentum-based classifier for BTC/USDT perpetual futures

**Specifications**:
- **Asset**: BTC/USDT perps (Binance)
- **Data**: 1-minute bars → resampled to 15-minute
- **Model**: XGBoost 3-class classifier (long/neutral/short)
- **Features**: RSI, MACD, EMA crossovers, volume indicators
- **Validation**: Walk-forward with 4 splits
- **Target**: Predict 15-minute directional move (>0.5% threshold)

---

## 🏗️ Implementation Steps

### Step 1: Create Strategy Configuration

Create `strategies/momentum_classifier_v1.yaml`:

```yaml
# Strategy metadata
strategy_name: momentum_classifier
version: v1
strategy_timeframe: short_term
description: "Momentum-based 3-class classifier for BTC perps"

# Data configuration
data:
  source_file: "data/raw/merged_binance_BTC_USDT_perps_1m_20251108_033030.parquet"
  resample_interval: "15min"  # Resample from 1m to 15m
  train_start: "2025-07-11"
  train_end: "2025-10-15"
  validation_start: "2025-10-16"
  validation_end: "2025-11-08"

# Feature configuration
features:
  indicators:
    # Moving averages
    - name: ema_12
      type: EMA
      params: {window: 12}
    - name: ema_26
      type: EMA
      params: {window: 26}
    - name: ema_50
      type: EMA
      params: {window: 50}

    # Momentum
    - name: rsi_14
      type: RSI
      params: {window: 14}
    - name: macd
      type: MACD
      params: {fast: 12, slow: 26, signal: 9}
    - name: roc_10
      type: ROC
      params: {window: 10}

    # Volatility
    - name: bollinger
      type: BollingerBands
      params: {window: 20, num_std: 2}
    - name: atr_14
      type: ATR
      params: {window: 14}

    # Volume
    - name: volume_ma_20
      type: VolumeMA
      params: {window: 20}

# Target variable configuration
target:
  type: classification
  classes: 3  # Short (0), Neutral (1), Long (2)
  lookahead_bars: 1  # Predict next 15m bar
  threshold_pct: 0.5  # 0.5% move threshold for long/short

# Model configuration (XGBoost)
model:
  type: xgboost
  params:
    objective: "multi:softmax"
    num_class: 3
    max_depth: 7
    learning_rate: 0.01
    n_estimators: 200
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    random_state: 42

# Validation configuration
validation:
  method: walk_forward
  n_splits: 4
  expanding_window: true
  gap_bars: 0

# Backtest configuration
backtest:
  transaction_cost_pct: 0.04  # 0.04% taker fee
  slippage_bps: 2  # 2 basis points
  leverage: 3  # 3x leverage
  capital: 10000  # Starting capital (USDT)

# Risk configuration
risk:
  max_position_size_pct: 100  # 100% of capital (with leverage)
  stop_loss_pct: 2.0  # 2% stop loss
  take_profit_pct: 3.0  # 3% take profit
```

### Step 2: Create Training Script

Create `strategies/short_term/train_momentum_classifier.py`:

```python
"""Training script for momentum classifier strategy."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config.strategy import StrategyConfig
from data.loader import DataLoader
from data.validator import DataValidator
from features.engine import FeatureEngine
from models import ModelRegistry, create_model_metadata
from validation.time_series_split import TimeSeriesSplitter
from utils.logger import get_logger, setup_logger

# Setup logging
setup_logger(level=20)  # INFO
logger = get_logger("training")


def create_target_variable(df: pd.DataFrame, lookahead_bars: int, threshold_pct: float) -> pd.Series:
    """
    Create 3-class target variable based on future price movement.

    Args:
        df: DataFrame with OHLCV data
        lookahead_bars: How many bars ahead to look
        threshold_pct: Percentage threshold for classification

    Returns:
        Series with target labels: 1 (long), 0 (neutral), -1 (short)
    """
    # Calculate future returns
    future_returns = df['close'].pct_change(periods=lookahead_bars).shift(-lookahead_bars)

    # Classify based on threshold
    target = pd.Series(0, index=df.index)  # Default: neutral
    target[future_returns > threshold_pct / 100] = 1  # Long
    target[future_returns < -threshold_pct / 100] = -1  # Short

    return target


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
        logger.error("Data quality check failed!")
        for error in quality_report.get_errors():
            logger.error(f"  - {error.message}")
        return

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

    # Create target variable
    logger.info("Creating target variable...")
    target = create_target_variable(
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

    X_train = df_clean[train_mask].drop(columns=['timestamp'])
    y_train = target_clean[train_mask]
    X_val = df_clean[val_mask].drop(columns=['timestamp'])
    y_val = target_clean[val_mask]

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Validation set: {len(X_val):,} samples")

    # Train model
    logger.info("Training XGBoost model...")

    import xgboost as xgb
    model = xgb.XGBClassifier(**config.model_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    # Evaluate
    logger.info("Evaluating model...")
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    logger.info("\\nValidation Classification Report:")
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
        model_type="xgboost",
        feature_config=config.feature_config.model_dump(),
        model_config=config.model_params,
        target_config={
            "type": "classification",
            "classes": 3,
            "lookahead_bars": config.target_lookahead_bars,
            "threshold_pct": config.target_threshold_pct
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

    logger.info(f"\\n{'='*70}")
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
        default="strategies/momentum_classifier_v1.yaml",
        help="Path to strategy configuration file"
    )

    args = parser.parse_args()

    trained_metadata = train_strategy(args.config)

    if trained_metadata:
        print("\\n" + trained_metadata.summary())
```

### Step 3: Run the Training

```bash
# Train the strategy
python strategies/short_term/train_momentum_classifier.py \\
    --config strategies/momentum_classifier_v1.yaml
```

**Expected Output**:
```
[INFO] Loading data from data/raw/merged_binance_BTC_USDT_perps_1m_20251108_033030.parquet
[INFO] Loaded 172,756 rows
[INFO] Data quality: 99.94%
[INFO] Resampling from 1m to 15min
[INFO] After resampling: 11,517 rows
[INFO] Computing features...
[INFO] Generated 15 features
[INFO] Creating target variable...
[INFO] Clean dataset: 11,500 rows
[INFO] Training set: 8,500 samples
[INFO] Validation set: 1,800 samples
[INFO] Training XGBoost model...
[INFO] Training accuracy: 0.4521
[INFO] Validation accuracy: 0.4233
[INFO] Model registered successfully!
[INFO] Model ID: a3f9c2d1e5b84f67_20251108_153045_a3f9c2
```

### Step 4: Query and Compare Models

```python
from models import ModelRegistry

registry = ModelRegistry()

# Find all momentum classifier models
models = registry.find_models(strategy_name="momentum_classifier")
print(f"Found {len(models)} momentum classifier models")

# Get latest
latest = registry.get_latest_model("momentum_classifier")
print(f"Latest model: {latest.model_id}")
print(f"Val accuracy: {latest.validation_metrics['val_accuracy']:.4f}")

# Find all runs of same configuration
config_hash = latest.get_config_hash()
same_config = registry.find_models_by_config(config_hash)
print(f"\\n{len(same_config)} training runs of this configuration:")
for model in same_config:
    print(f"  {model.get_training_timestamp()}: Acc={model.validation_metrics['val_accuracy']:.4f}")
```

---

## 🔍 Next Steps After First Strategy

1. **Backtest the Model**
   - Use `BacktestEngine` to simulate trading
   - Calculate Sharpe ratio, max drawdown, win rate
   - Analyze performance across different market regimes

2. **Implement Walk-Forward Validation**
   - Use `WalkForwardValidator` for rigorous testing
   - Check performance consistency across time periods
   - Detect overfitting

3. **Hyperparameter Tuning**
   - Try different XGBoost parameters
   - Experiment with feature sets
   - Adjust target thresholds

4. **Train Second Strategy**
   - Prove modularity of the system
   - Different timeframe or different approach
   - Compare performance with first strategy

5. **Paper Trading**
   - Deploy to testnet
   - Monitor live performance
   - Validate execution infrastructure

---

## 📊 Common Issues & Solutions

### Issue: Low Validation Accuracy

**Solutions**:
- Check target class balance (might need rebalancing)
- Add more features or engineer new ones
- Try different lookahead windows
- Adjust classification thresholds

### Issue: Data Leakage

**Symptoms**: Training accuracy >> Validation accuracy

**Check**:
```python
# Verify temporal ordering
assert df_train['timestamp'].max() < df_val['timestamp'].min()

# Verify no future data in features
feature_engine.validate_lookback(df, lookback_bars=50)
```

### Issue: Model Overfitting

**Solutions**:
- Increase regularization (max_depth, min_child_weight)
- Use more validation splits
- Add early stopping
- Reduce feature complexity

---

## 📚 Additional Resources

- **[.claude/PROJECT_CONTEXT.md](../.claude/PROJECT_CONTEXT.md)**: Complete system architecture
- **[.claude/DECISIONS.md](../.claude/DECISIONS.md)**: Why we built it this way
- **[validation/](../validation/)**: Validation framework details
- **[features/indicators/](../features/indicators/)**: Available indicators

---

**Ready to train your first model?** Start with Step 1 and work through sequentially. The system will handle all metadata tracking, model registration, and provenance automatically!
