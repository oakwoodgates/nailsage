"""Verify all module imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("Verifying Module Imports")
print("=" * 70)
print()

modules_to_test = [
    ("config", ["BaseConfig", "DataConfig", "FeatureConfig", "StrategyConfig", "BacktestConfig", "RiskConfig"]),
    ("data", ["DataLoader", "DataValidator", "DatasetMetadata", "load_metadata", "save_metadata"]),
    ("features", ["FeatureEngine", "BaseIndicator"]),
    ("features.indicators.moving_average", ["SMA", "EMA"]),
    ("features.indicators.momentum", ["RSI", "MACD", "ROC"]),
    ("features.indicators.volatility", ["BollingerBands", "ATR"]),
    ("features.indicators.volume", ["VolumeMA"]),
    ("training.validation", ["TimeSeriesSplitter", "WalkForwardValidator", "BacktestEngine", "PerformanceMetrics"]),
    ("models", ["ModelMetadata", "ModelRegistry", "create_model_metadata", "generate_model_id", "compare_models", "get_model_lineage"]),
    ("utils.logger", ["setup_logger", "get_logger"]),
]

failed_imports = []
passed_imports = []

for module_name, exports in modules_to_test:
    try:
        module = __import__(module_name, fromlist=exports)

        # Verify exports exist
        missing = []
        for export in exports:
            if not hasattr(module, export):
                missing.append(export)

        if missing:
            print(f"FAIL {module_name}: Missing exports: {', '.join(missing)}")
            failed_imports.append((module_name, f"Missing: {', '.join(missing)}"))
        else:
            print(f"PASS {module_name}: All {len(exports)} exports verified")
            passed_imports.append(module_name)

    except Exception as e:
        print(f"FAIL {module_name}: Import failed - {e}")
        failed_imports.append((module_name, str(e)))

print()
print("=" * 70)
print(f"Results: {len(passed_imports)} passed, {len(failed_imports)} failed")
print("=" * 70)

if failed_imports:
    print()
    print("Failed imports:")
    for module, error in failed_imports:
        print(f"  - {module}: {error}")
    sys.exit(1)
else:
    print()
    print("SUCCESS: All module imports verified!")
    sys.exit(0)
