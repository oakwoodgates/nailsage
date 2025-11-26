# Feature Schema Usage Guide

## Overview

The `FeatureSchema` class ensures that features computed at inference time **exactly match** what the model was trained on, preventing shape mismatches and silent prediction errors.

## Problem Solved

### Before Feature Schema (The Bug):
```python
# Training: Features include OHLCV + indicators
X_train = df[['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi_14']]

# Inference: Features might be different!
feature_df = engine.compute_features(candle_df)  # Returns OHLCV + indicators
X_inference = feature_df[['sma_20', 'rsi_14']]  # Oops! Missing OHLCV!

# Result: Shape mismatch error at runtime ❌
```

### After Feature Schema (The Fix):
```python
# Training: Create schema from training features
schema = FeatureSchema.from_dataframe(X_train, include_ohlcv=False)
metadata.feature_schema = schema.to_dict()

# Inference: Schema validates and extracts correct features
schema.validate(feature_df)
X_inference = schema.extract_features(feature_df)

# Result: Guaranteed correct features ✅
```

## Usage in Model Training

### Step 1: Create Feature Schema

```python
from models.feature_schema import FeatureSchema

# After computing training features
features_df = feature_engine.compute_features(price_data)

# Create schema (exclude OHLCV for Phase 10 best practice)
feature_schema = FeatureSchema.from_dataframe(
    features_df,
    include_ohlcv=False,       # Exclude raw price data (prevents leakage)
    include_volume=False,       # Also exclude volume
    include_num_trades=False,   # Also exclude num_trades
)

print(f"Feature schema: {feature_schema.feature_names}")
# Output: ['sma_20', 'rsi_14', 'macd', 'bollinger_upper', ...]
```

### Step 2: Save Schema in Model Metadata

```python
from models.metadata import ModelMetadata

metadata = ModelMetadata(
    model_id="abc123...",
    strategy_name="my_strategy",
    # ... other fields ...
    feature_schema=feature_schema.to_dict(),  # Add this!
)

# Save metadata
metadata.save(f"models/metadata/{model_id}.json")
```

### Step 3: Inference Uses Schema Automatically

The `ModelPredictor` automatically loads and uses the schema:

```python
from execution.inference.predictor import ModelPredictor

predictor = ModelPredictor(
    model_id="abc123...",
    registry=registry,
    feature_engine=engine,
)

await predictor.load_model()  # Loads schema from metadata

# Prediction validates features automatically
prediction = await predictor.predict(candle_df, timestamp)
# If features don't match schema → ValueError with helpful message ✅
```

## Configuration Options

### Include OHLCV (Legacy Models)

Some older models were trained with OHLCV included:

```python
schema = FeatureSchema.from_dataframe(
    features_df,
    include_ohlcv=True,   # Include open, high, low, close
    include_volume=True,  # Include volume
)
```

### Partial OHLCV (OHLC but no volume)

```python
schema = FeatureSchema.from_dataframe(
    features_df,
    include_ohlcv=True,    # Include OHLC
    include_volume=False,  # But exclude volume
)
```

### Include num_trades

```python
schema = FeatureSchema.from_dataframe(
    features_df,
    include_ohlcv=False,
    include_num_trades=True,  # Include num_trades column
)
```

### Custom Exclusions

```python
schema = FeatureSchema.from_dataframe(
    features_df,
    include_ohlcv=False,
    exclude_columns=['experimental_feature'],  # Exclude specific columns
)
```

## Validation Errors

### Missing Features Error

```python
# At inference time, if features are missing:
ValueError: Missing required features: {'macd', 'rsi_14'}.
Expected: ['sma_20', 'rsi_14', 'macd']
```

**Fix:** Ensure your FeatureEngine configuration matches training.

### NaN Values Error

```python
# If features have NaN values:
ValueError: Insufficient data for feature computation - NaN in: ['sma_20', 'macd']
```

**Fix:** Provide more historical candles for indicator computation.

## Best Practices

### 1. Always Exclude OHLCV (Phase 10 Best Practice)

```python
# ✅ GOOD: Prevents data leakage
schema = FeatureSchema.from_dataframe(df, include_ohlcv=False)

# ❌ BAD: Raw price data can leak future information
schema = FeatureSchema.from_dataframe(df, include_ohlcv=True)
```

**Why?** Raw OHLCV columns can introduce look-ahead bias in some ML models.

### 2. Create Schema from Training Data

```python
# ✅ GOOD: Create from actual training features
X_train = features[train_mask]
schema = FeatureSchema.from_dataframe(X_train, include_ohlcv=False)

# ❌ BAD: Creating manually (error-prone)
schema = FeatureSchema(feature_names=['sma_20', 'rsi_14'])  # Easy to make mistakes
```

### 3. Verify Schema Before Saving Model

```python
# Verify schema extracts features correctly
test_features = schema.extract_features(X_val)
assert len(test_features.columns) == len(schema.feature_names)

# Check for NaN issues
nan_features = schema.check_nan_values(test_features)
if nan_features:
    print(f"Warning: NaN in features: {nan_features}")
```

## Backward Compatibility

Models trained **before** FeatureSchema was implemented will still work:

```python
# Old model (no feature_schema in metadata)
predictor.load_model()
# Warning: "No feature schema in metadata - will use legacy feature extraction"

# Still works, but without validation
prediction = await predictor.predict(candle_df, timestamp)
```

**Recommendation:** Retrain models to include feature_schema for better reliability.

## Migration Guide

### Updating Existing Models

If you have trained models without feature schemas:

**Option 1: Retrain (Recommended)**
```bash
# Retrain model with updated training script
python strategies/train_model.py --config configs/my_strategy.yaml
```

**Option 2: Add Schema to Existing Model**
```python
from models.registry import ModelRegistry
from models.feature_schema import FeatureSchema

# Load existing model
registry = ModelRegistry()
metadata = registry.get_model("existing_model_id")

# Create schema from training data (requires original training data)
training_df = pd.read_parquet(metadata.training_dataset_path)
features_df = feature_engine.compute_features(training_df)
schema = FeatureSchema.from_dataframe(features_df, include_ohlcv=False)

# Update metadata
metadata.feature_schema = schema.to_dict()
metadata.save(f"models/metadata/{metadata.model_id}.json")
```

## Testing

Unit tests are available at `tests/unit/test_feature_schema.py`:

```bash
# Run feature schema tests
pytest tests/unit/test_feature_schema.py -v

# Expected: 17/17 passing ✅
```

## Example: Full Training Workflow

```python
import pandas as pd
from features.engine import FeatureEngine
from models.feature_schema import FeatureSchema
from models.metadata import ModelMetadata
from config.feature import FeatureConfig

# 1. Load data
df = pd.read_parquet("data/raw/BTC_USDT_4h.parquet")

# 2. Compute features
config = FeatureConfig.load("configs/features/momentum_v1.yaml")
engine = FeatureEngine(config)
features_df = engine.compute_features(df)

# 3. Create feature schema
feature_schema = FeatureSchema.from_dataframe(
    features_df,
    include_ohlcv=False,  # Phase 10 best practice
)

# 4. Prepare training data
X = feature_schema.extract_features(features_df)
y = create_binary_target(df, lookahead_bars=4, threshold_pct=1.5)

# 5. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 6. Save with schema
metadata = ModelMetadata(
    model_id=generate_model_id(),
    strategy_name="my_strategy",
    # ... other fields ...
    feature_schema=feature_schema.to_dict(),  # ✅ Include schema!
)

joblib.dump(model, f"models/trained/{metadata.model_id}.joblib")
metadata.save(f"models/metadata/{metadata.model_id}.json")
```

## Troubleshooting

### Q: "Missing required features" error at inference

**A:** Your inference feature configuration doesn't match training. Check:
1. Same indicator configurations (e.g., SMA window=20)
2. Same feature exclusions (OHLCV, volume, etc.)
3. Feature engine config matches training

### Q: Features are in different order

**A:** FeatureSchema handles this automatically! Features are always extracted in the correct order.

### Q: Can I update schema for existing model?

**A:** Yes, but only if you have the original training data. See "Migration Guide" above.

### Q: What if I want to add new features?

**A:** You must retrain the model. Changing features = new model.

## Summary

✅ **Always use FeatureSchema** for new models
✅ **Exclude OHLCV** by default (Phase 10 best practice)
✅ **Create schema from training data**, not manually
✅ **Test schema validation** before deploying
✅ **Retrain old models** to add feature schemas

---

**Related Documentation:**
- [Phase 10 Features](PHASE_10_FEATURES.md)
- [Model Training Guide](../strategies/README.md)
- [Feature Engineering](../features/README.md)
