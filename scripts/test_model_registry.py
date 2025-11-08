"""Test script for Model Registry system."""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    ModelMetadata,
    ModelRegistry,
    compare_models,
    create_model_metadata,
    get_model_lineage,
)

print("=" * 70)
print("Testing Model Registry System")
print("=" * 70)
print()

# Create a temporary model artifact (simulate a trained model)
temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
temp_model.write(b"fake model data")
temp_model.close()
temp_model_path = Path(temp_model.name)

print(f"1. Creating sample model metadata...")
print("-" * 70)

# Create model metadata using helper function
metadata1 = create_model_metadata(
    strategy_name="ema_cross_rsi",
    strategy_timeframe="short_term",
    version="v1",
    training_dataset_path="data/raw/merged_binance_BTC_USDT_perps_1m_20251108_033030.parquet",
    training_date_range=("2025-07-11T03:31:00Z", "2025-10-15T00:00:00Z"),
    validation_date_range=("2025-10-16T00:00:00Z", "2025-11-08T02:46:00Z"),
    model_type="xgboost",
    feature_config={
        "indicators": ["ema_12", "ema_26", "rsi_14"],
        "resampling": "15m",
        "parameters": {"ema_12": 12, "ema_26": 26, "rsi_14": 14},
    },
    model_config={
        "max_depth": 5,
        "learning_rate": 0.01,
        "n_estimators": 100,
        "objective": "multi:softmax",
        "num_class": 3,
    },
    target_config={
        "type": "classification",
        "classes": ["long", "neutral", "short"],
        "lookahead_bars": 5,
        "threshold_pct": 0.5,
    },
    validation_metrics={
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.31,
        "max_drawdown": -0.12,
        "win_rate": 0.58,
        "profit_factor": 1.42,
        "total_return": 0.23,
    },
    model_artifact_path=str(temp_model_path),
    notes="Initial baseline model using EMA crossover + RSI",
    tags=["baseline", "momentum"],
)

print(f"Model ID: {metadata1.model_id}")
print(f"Strategy: {metadata1.strategy_name} ({metadata1.strategy_timeframe})")
print(f"Model Type: {metadata1.model_type}")
print()

# Create a second model for comparison
print(f"2. Creating second model (improved version)...")
print("-" * 70)

temp_model2 = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
temp_model2.write(b"fake model data v2")
temp_model2.close()
temp_model_path2 = Path(temp_model2.name)

metadata2 = create_model_metadata(
    strategy_name="ema_cross_rsi",
    strategy_timeframe="short_term",
    version="v2",
    training_dataset_path="data/raw/merged_binance_BTC_USDT_perps_1m_20251108_033030.parquet",
    training_date_range=("2025-07-11T03:31:00Z", "2025-10-15T00:00:00Z"),
    validation_date_range=("2025-10-16T00:00:00Z", "2025-11-08T02:46:00Z"),
    model_type="xgboost",
    feature_config={
        "indicators": ["ema_12", "ema_26", "rsi_14", "macd"],
        "resampling": "15m",
        "parameters": {"ema_12": 12, "ema_26": 26, "rsi_14": 14},
    },
    model_config={
        "max_depth": 7,  # Increased
        "learning_rate": 0.015,  # Tuned
        "n_estimators": 150,  # More trees
        "objective": "multi:softmax",
        "num_class": 3,
    },
    target_config={
        "type": "classification",
        "classes": ["long", "neutral", "short"],
        "lookahead_bars": 5,
        "threshold_pct": 0.5,
    },
    validation_metrics={
        "sharpe_ratio": 2.12,  # Improved
        "sortino_ratio": 2.68,
        "max_drawdown": -0.09,  # Better
        "win_rate": 0.62,  # Better
        "profit_factor": 1.58,
        "total_return": 0.31,  # Better
    },
    model_artifact_path=str(temp_model_path2),
    notes="Improved model with additional MACD feature and tuned hyperparameters",
    tags=["improved", "momentum"],
)

print(f"Model ID: {metadata2.model_id}")
print(f"Sharpe Ratio: {metadata2.validation_metrics['sharpe_ratio']}")
print()

# Initialize registry
print(f"3. Initializing Model Registry...")
print("-" * 70)

registry = ModelRegistry()
print(f"Registry initialized")
print(f"Models dir: {registry.models_dir}")
print(f"Metadata dir: {registry.metadata_dir}")
print()

# Register models
print(f"4. Registering models...")
print("-" * 70)

registry.register_model(temp_model_path, metadata1)
print(f"Registered model 1: {metadata1.model_id}")

registry.register_model(temp_model_path2, metadata2)
print(f"Registered model 2: {metadata2.model_id}")
print()

# Query models
print(f"5. Querying models...")
print("-" * 70)

# Find all short_term models
short_term_models = registry.find_models(strategy_timeframe="short_term")
print(f"Short-term models: {len(short_term_models)}")

# Find all ema_cross_rsi models
ema_models = registry.find_models(strategy_name="ema_cross_rsi")
print(f"EMA cross RSI models: {len(ema_models)}")

# Get latest model
latest = registry.get_latest_model("ema_cross_rsi")
print(f"Latest ema_cross_rsi model: {latest.model_id} (version {latest.version})")

# Get best model by Sharpe ratio
best = registry.get_best_model("sharpe_ratio", strategy_name="ema_cross_rsi")
print(f"Best Sharpe ratio: {best.validation_metrics['sharpe_ratio']} (version {best.version})")
print()

# Compare models
print(f"6. Comparing models...")
print("-" * 70)

comparison = compare_models(metadata1, metadata2, metrics=["sharpe_ratio", "win_rate", "max_drawdown"])
print(f"Comparison between v1 and v2:")
for metric, values in comparison["metrics"].items():
    print(f"  {metric}:")
    print(f"    v1: {values['model1']:.4f}")
    print(f"    v2: {values['model2']:.4f}")
    print(f"    Change: {values['difference']:+.4f} ({values['percent_change']:+.2f}%)")
print()

# Get model lineage
print(f"7. Getting model lineage...")
print("-" * 70)

lineage = get_model_lineage(metadata2, include_dataset=True)
print(f"Model: {lineage['model_id']}")
print(f"Strategy: {lineage['strategy']['name']} v{lineage['strategy']['version']}")
print(f"Training period: {lineage['training']['date_range'][0]} to {lineage['training']['date_range'][1]}")
print(f"Features: {', '.join(lineage['configuration']['features']['indicators'])}")
print(f"Performance:")
for metric, value in lineage['performance'].items():
    print(f"  {metric}: {value:.4f}")
print()

# Get summary
print(f"8. Registry summary...")
print("-" * 70)

summary = registry.get_strategies_summary()
print(f"Total models: {registry.get_model_count()}")
print(f"Strategies:")
for strategy, info in summary.items():
    print(f"  {strategy}:")
    print(f"    Count: {info['count']}")
    print(f"    Timeframes: {', '.join(info['timeframes'])}")
    print(f"    Versions: {', '.join(info['versions'])}")
print()

# Print full metadata summary
print(f"9. Full metadata summary (v2 model)...")
print("-" * 70)
print(metadata2.summary())

# Cleanup
print(f"10. Cleanup...")
print("-" * 70)
temp_model_path.unlink()
temp_model_path2.unlink()
print("Temporary files cleaned up")
print()

print("=" * 70)
print("Model Registry Test Complete!")
print("=" * 70)
print()
print("The Model Registry system successfully:")
print("  - Created model metadata with full provenance")
print("  - Registered models with centralized storage")
print("  - Queried models by multiple criteria")
print("  - Compared model performance")
print("  - Tracked complete lineage")
print("  - Maintained modular, flexible architecture")
