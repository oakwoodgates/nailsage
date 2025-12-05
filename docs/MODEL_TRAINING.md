# Model Training & Validation Guide

Complete guide for training, validating, and backtesting ML trading models in NailSage using the generic, configuration-driven training architecture.

## 🎯 Overview

NailSage provides three generic scripts that work with any strategy configuration:

1. **`training/cli/train_model.py`** - Train models with optional walk-forward validation
2. **`training/cli/validate_model.py`** - Standalone validation of existing models
3. **`training/cli/run_backtest.py`** - Quick single-period backtests

All scripts are **configuration-driven** - strategy differences are captured in YAML config files, not code.

## 📋 Quick Start

```bash
# Train with walk-forward validation (saves results to JSON)
python training/cli/train_model.py --config configs/strategies/dev_scalper_1m_v1.yaml

# Validate existing model
python training/cli/validate_model.py \
    --config configs/strategies/dev_scalper_1m_v1.yaml \
    --model-id 33b9a1937aacaa4d_20251126_152519_f8835d

# Quick backtest
python training/cli/run_backtest.py \
    --config configs/strategies/dev_scalper_1m_v1.yaml \
    --model-id 33b9a1937aacaa4d_20251126_152519_f8835d
```

## 🏗️ Script Details

### 1. Train Model (`scripts/train_model.py`)

**Purpose**: Train models with optional walk-forward validation

**Features**:
- Supports all model types (XGBoost, LightGBM, RandomForest, ExtraTrees)
- Walk-forward validation (default on)
- Class weighting for imbalanced datasets
- Confidence filtering and trade cooldown
- Backtest integration per validation split
- Aggregate metrics reporting
- JSON results output (default on)

**Usage**:

```bash
# Train and validate (default)
python scripts/train_model.py --config configs/strategies/your_strategy.yaml

# Train only, skip validation
python scripts/train_model.py --config configs/strategies/your_strategy.yaml --train-only

# Train without saving JSON results
python scripts/train_model.py --config configs/strategies/your_strategy.yaml --no-json
```

**What It Does**:

1. Loads and prepares data
2. Computes features using FeatureEngine
3. Creates target variable (2-class or 3-class)
4. Trains model on training period
5. **If validation enabled** (default):
   - Runs walk-forward validation with N splits
   - Applies confidence filtering and trade cooldown
   - Runs backtest on each split
   - Computes aggregate metrics
6. Registers model in ModelRegistry
7. Saves results to `results/training/`

**Output**:

```
Model ID: 33b9a1937aacaa4d_20251126_152519_f8835d
Training accuracy: 0.8400

Walk-Forward Validation (4 splits):
  Split 1: Return=5.2%, Sharpe=1.2
  Split 2: Return=-2.1%, Sharpe=0.3
  Split 3: Return=8.5%, Sharpe=1.8
  Split 4: Return=3.2%, Sharpe=0.9

Aggregate Metrics:
  Average Return: 3.7%
  Average Sharpe: 1.05
  Consistency: 75% (3/4 splits profitable)

Results saved to: results/training/dev_scalper_1m_v1_20251126_152519.json
```

---

### 2. Validate Model (`scripts/validate_model.py`)

**Purpose**: Run comprehensive walk-forward validation on existing trained models

**Features**:
- Standalone validation without retraining
- Optional model retraining per split (true walk-forward)
- Confidence filtering and trade cooldown
- Split-level and aggregate metrics
- Classification metrics (accuracy, F1, MCC)
- Backtest metrics per split
- JSON results output (default on)

**Usage**:

```bash
# Validate existing model
python scripts/validate_model.py \
    --config configs/strategies/dev_scalper_1m_v1.yaml \
    --model-id 33b9a1937aacaa4d_20251126_152519_f8835d

# Validate with model retraining per split (true walk-forward)
python scripts/validate_model.py \
    --config configs/strategies/dev_scalper_1m_v1.yaml \
    --model-id MODEL_ID \
    --retrain

# Validate without saving JSON
python scripts/validate_model.py \
    --config CONFIG \
    --model-id MODEL_ID \
    --no-save
```

**What It Does**:

1. Loads existing model from registry
2. Loads and prepares validation data
3. Creates validation splits based on config
4. For each split:
   - Generates predictions (optionally retrains model)
   - Applies confidence filtering
   - Applies trade cooldown
   - Runs backtest
   - Computes classification and backtest metrics
5. Aggregates metrics across all splits
6. Saves results to `results/validation/`

**Output**:

```
Split 1/4:
  Val Accuracy: 0.8803, F1: 0.4162, MCC: 0.2190
  Backtest: Total Return=-15.2%, Sharpe=-1.48, Max DD=-20.1%
  Win Rate: 17.79%, Total Trades: 3784

Aggregate Results:
  Avg Accuracy: 0.8650
  Avg Return: -8.5%
  Avg Sharpe: -0.95
  Consistency: 25% (1/4 splits profitable)

Results saved to: results/validation/dev_scalper_1m_v1_20251126_174211.json
```

---

### 3. Run Backtest (`scripts/run_backtest.py`)

**Purpose**: Quick single-period backtests on validation data

**Features**:
- Fast single-period validation
- Confidence filtering
- Trade cooldown
- Comprehensive metrics display
- JSON results output (default on)

**Usage**:

```bash
# Run backtest
python scripts/run_backtest.py \
    --config configs/strategies/dev_scalper_1m_v1.yaml \
    --model-id 33b9a1937aacaa4d_20251126_152519_f8835d

# Run without saving JSON
python scripts/run_backtest.py \
    --config CONFIG \
    --model-id MODEL_ID \
    --no-json
```

**What It Does**:

1. Loads model from registry
2. Loads validation period data
3. Generates predictions
4. Applies confidence filtering and cooldown
5. Runs backtest on filtered signals
6. Displays comprehensive metrics
7. Saves results to `results/backtest/`

**Output**:

```
======================================================================
BACKTEST RESULTS
======================================================================

Model: 33b9a1937aacaa4d_20251126_152519_f8835d
Strategy: dev_scalper_1m v1
Period: 2025-08-16 to 2025-11-15
Initial Capital: $10,000.00
Leverage: 3x

Transaction Costs:
  Fee: 0.040%
  Slippage: 2 bps

Signal Filtering:
  Confidence Threshold: 0.00
  Trade Cooldown: 0 bars

-----------------------------PERFORMANCE METRICS--------------------------
Total Return: -100.00%
Annual Return: -100.00%
Sharpe Ratio: -1.48
Max Drawdown: -100.00%
Win Rate: 17.79%

-----------------------------TRADING STATISTICS---------------------------
Total Trades: 3784
Winning Trades: 673
Losing Trades: 3111
Average Win: $8.45
Average Loss: $-3.12
Profit Factor: 0.89

------------------------------RISK METRICS--------------------------------
Volatility: 5.23%
Downside Volatility: 4.87%
Sortino Ratio: -1.62
Calmar Ratio: -0.85
======================================================================

Results saved to: results/backtest/dev_scalper_1m_v1_20251126_180145.json
```

---

## 🔧 Strategy Configuration

All scripts use the same YAML configuration format. See `configs/strategies/dev_scalper_1m_v1.yaml` for a complete example.

### Key Configuration Sections

```yaml
# Strategy identification
strategy_name: dev_scalper_1m
version: v1
strategy_timeframe: ultra_short_term
description: "Aggressive 1m scalping model"

# Data configuration
data:
  source_file: "merged_binance_BTC_USDT_perps_1m_20251116_014156.parquet"
  resample_interval: null  # Keep native 1m resolution
  train_start: "2024-11-16"
  train_end: "2025-08-15"
  validation_start: "2025-08-16"
  validation_end: "2025-11-15"

# Feature configuration
features:
  indicators:
    - name: ema_5
      type: EMA
      params: {window: 5}
    # ... more indicators

# Target configuration
target:
  type: classification
  classes: 3  # 2 or 3
  lookahead_bars: 3
  threshold_pct: 0.2
  class_weights: {0: 15.0, 1: 1.0, 2: 15.0}  # Optional
  confidence_threshold: 0.0  # 0.0-1.0, 0 = no filtering

# Model configuration
model:
  type: lightgbm  # xgboost, lightgbm, random_forest, extra_trees
  params:
    objective: "multiclass"
    num_class: 3
    max_depth: 5
    learning_rate: 0.05
    n_estimators: 200
    # ... more hyperparameters

# Validation configuration
validation:
  method: walk_forward  # walk_forward or single_split
  n_splits: 4
  expanding_window: true  # true = expanding, false = sliding
  gap_bars: 0

# Backtest configuration
backtest:
  transaction_cost_pct: 0.04  # Taker fee %
  slippage_bps: 2
  leverage: 3
  capital: 10000
  min_bars_between_trades: 0  # Trade cooldown (0 = no cooldown)

# Risk configuration
risk:
  max_position_size_pct: 100
  stop_loss_pct: 1.0
  take_profit_pct: 1.5
```

---

## 📊 Advanced Features

### Class Weights for Imbalanced Data

When your target classes are imbalanced (e.g., 90% neutral, 5% long, 5% short), use class weights to focus the model on trading signals:

```yaml
target:
  classes: 3
  class_weights: {0: 15.0, 1: 1.0, 2: 15.0}  # 15x weight on short/long
```

This tells the model to prioritize correct long/short predictions over neutral.

### Confidence Filtering

Suppress low-confidence predictions to reduce false signals:

```yaml
target:
  confidence_threshold: 0.6  # Only trade when model is >60% confident
```

This filters out predictions where the model's maximum probability is below the threshold.

### Trade Cooldown

Prevent overtrading by enforcing minimum bars between trades:

```yaml
backtest:
  min_bars_between_trades: 5  # Wait 5 bars after each trade
```

This suppresses signals within N bars of the last trade.

### Walk-Forward Validation

Configure the validation method:

```yaml
validation:
  method: walk_forward
  n_splits: 4  # Number of validation splits
  expanding_window: true  # Expanding (vs sliding) window
  gap_bars: 0  # Gap between train/val splits
```

**Expanding vs Sliding Window**:
- **Expanding** (recommended): Training window grows over time, validation window slides forward
- **Sliding**: Both windows slide forward, keeping training size constant

---

## 📁 Results Output

All scripts save results to JSON files for programmatic access:

```
results/
├── training/
│   └── dev_scalper_1m_v1_20251126_152519.json
├── validation/
│   └── dev_scalper_1m_v1_20251126_174211.json
└── backtest/
    └── dev_scalper_1m_v1_20251126_180145.json
```

### JSON Structure (Training)

```json
{
  "model_id": "33b9a1937aacaa4d_20251126_152519_f8835d",
  "config": {
    "strategy_name": "dev_scalper_1m",
    "version": "v1",
    "training_period": {"start": "2024-11-16", "end": "2025-08-15"},
    "validation_period": {"start": "2025-08-16", "end": "2025-11-15"}
  },
  "training_metrics": {
    "train_accuracy": 0.8400
  },
  "validation_results": {
    "n_splits": 4,
    "splits": [...],
    "aggregate": {
      "avg_val_accuracy": 0.8650,
      "avg_total_return": -0.085,
      "avg_sharpe_ratio": -0.95,
      "consistency_pct": 0.25
    }
  }
}
```

---

## 🔬 Workflow Examples

### 1. Train New Model

```bash
# Create config file: configs/strategies/my_strategy_v1.yaml

# Train with validation
python scripts/train_model.py --config configs/strategies/my_strategy_v1.yaml

# Note the model ID from output
# Model ID: abc123_20251126_120000_xyz789
```

### 2. Validate Existing Model

```bash
# Run standalone validation
python scripts/validate_model.py \
    --config configs/strategies/my_strategy_v1.yaml \
    --model-id abc123_20251126_120000_xyz789

# Check results
cat results/validation/my_strategy_v1_20251126_120530.json
```

### 3. Quick Backtest

```bash
# Fast single-period backtest
python scripts/run_backtest.py \
    --config configs/strategies/my_strategy_v1.yaml \
    --model-id abc123_20251126_120000_xyz789
```

### 4. Hyperparameter Tuning

```bash
# 1. Create multiple config files with different hyperparameters
configs/strategies/my_strategy_v1_hp1.yaml  # max_depth=5
configs/strategies/my_strategy_v1_hp2.yaml  # max_depth=7
configs/strategies/my_strategy_v1_hp3.yaml  # max_depth=10

# 2. Train all variants
python scripts/train_model.py --config configs/strategies/my_strategy_v1_hp1.yaml
python scripts/train_model.py --config configs/strategies/my_strategy_v1_hp2.yaml
python scripts/train_model.py --config configs/strategies/my_strategy_v1_hp3.yaml

# 3. Compare JSON results
python -c "
import json
import glob

for file in glob.glob('results/training/my_strategy_v1_*.json'):
    with open(file) as f:
        data = json.load(f)
        sharpe = data['validation_results']['aggregate']['avg_sharpe_ratio']
        print(f'{file}: Sharpe={sharpe:.2f}')
"
```

---

## 🎓 Best Practices

### 1. Always Validate

Don't skip validation - always run walk-forward validation to check consistency:

```bash
# Good: Train with validation
python scripts/train_model.py --config CONFIG

# Bad: Train only (no validation)
python scripts/train_model.py --config CONFIG --train-only
```

### 2. Use Expanding Windows

Expanding windows are more realistic - you always have all historical data available:

```yaml
validation:
  expanding_window: true  # Recommended
```

### 3. Check Consistency

A model that's profitable on 1/4 splits is not robust:

```
Consistency: 25% (1/4 splits profitable)  # ❌ Not consistent
Consistency: 75% (3/4 splits profitable)  # ✅ Better
```

### 4. Monitor Overfitting

If training accuracy >> validation accuracy, you're likely overfitting:

```
Training accuracy: 0.95  # Too high
Validation accuracy: 0.52  # Too low
→ Increase regularization, reduce features
```

### 5. Test Different Thresholds

Try different confidence thresholds and cooldowns to optimize trade quality:

```yaml
# Test different confidence thresholds
target:
  confidence_threshold: 0.0   # Baseline
  confidence_threshold: 0.5   # Medium filtering
  confidence_threshold: 0.7   # Aggressive filtering
```

---

## 📚 Additional Resources

- **Configuration**: See `config/strategy.py` for all available options
- **Features**: See `features/indicators/` for available technical indicators
- **Validation**: See `validation/` for validation framework details
- **Registry**: See `models/registry.py` for model management
- **Examples**: See `configs/strategies/` for example configurations

---

## 🤝 Migration from Old Scripts

If you have old strategy-specific scripts in `strategies/short_term/`, they've been deprecated in favor of the new generic scripts.

**Old approach** (deprecated):
```bash
python strategies/short_term/train_momentum_classifier.py --config CONFIG
```

**New approach** (recommended):
```bash
python scripts/train_model.py --config CONFIG
```

See `strategies/_deprecated/README.md` for migration details.

---

**Ready to train your first model?** Start with the Quick Start section and work through the examples!
