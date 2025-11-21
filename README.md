# NailSage

**ML Trading Research Platform for Cryptocurrency Markets**

Named after the Great Nailsage Sly, trainer of the Nailmasters, from Hollow Knight.

## 🎯 Overview

NailSage is a production-ready ML trading research platform designed for building, testing, and deploying machine learning trading strategies with rigorous validation and complete reproducibility.

**Current Status**: Core framework complete (17/25 milestones, 68%) - Ready for first strategy implementation

**Phase 1 Focus**: Classical ML (XGBoost, LightGBM, Random Forest) with walk-forward validation

## ✨ Key Features

- **✅ Complete Metadata Tracking**: Full data and model provenance for reproducibility
- **✅ Data Leakage Prevention**: Strict temporal ordering with walk-forward validation
- **✅ Realistic Backtesting**: Transaction costs, slippage, and leverage simulation
- **✅ Hybrid Model Registry**: Track configuration intent and training history
- **✅ Dynamic Feature Engineering**: 8 technical indicators computed on-the-fly
- **✅ Modular Architecture**: Independent strategies with centralized model management

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository>
cd nailsage

# Install dependencies
pip install -r requirements.txt  # Or use pyproject.toml

# Verify imports
python scripts/verify_imports.py
```

### Generate Data Metadata

```bash
# For a single file
python scripts/generate_data_metadata.py --file data/raw/your_data.parquet

# For entire directory
python scripts/generate_data_metadata.py --dir data/raw
```

### Run Tests

```bash
# Run all tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=. --cov-report=term-missing
```

## 📁 Project Structure

```
nailsage/
├── config/                    # Pydantic configuration models
│   ├── base.py               # BaseConfig with YAML loading
│   ├── data.py               # DataConfig (OHLCV loading)
│   ├── feature.py            # FeatureConfig (indicators)
│   ├── strategy.py           # StrategyConfig
│   ├── backtest.py           # BacktestConfig (fees, slippage)
│   └── risk.py               # RiskConfig (position sizing)
├── configs/                   # YAML configuration files
├── data/                      # Data management
│   ├── loader.py             # Load OHLCV data (Parquet/CSV)
│   ├── validator.py          # Data quality validation
│   ├── metadata.py           # Dataset provenance tracking
│   └── raw/                  # Raw OHLCV data storage
├── features/                  # Feature engineering
│   ├── engine.py             # Dynamic feature computation
│   ├── indicators/           # 8 technical indicators
│   └── cache/                # Feature cache storage
├── validation/                # Validation framework
│   ├── time_series_split.py # Walk-forward splitting
│   ├── backtest.py           # Backtesting engine
│   ├── metrics.py            # Performance metrics
│   └── walk_forward.py       # Complete validation pipeline
├── models/                    # Model registry & metadata
│   ├── metadata.py           # ModelMetadata (hybrid IDs)
│   ├── registry.py           # Centralized model storage
│   ├── utils.py              # Model utilities
│   ├── trained/              # Serialized models
│   └── metadata/             # Model metadata (JSON)
├── strategies/                # Strategy implementations
│   ├── short_term/           # Short-term strategies
│   └── long_term/            # Long-term strategies
├── tests/                     # Test suite (21 passing tests)
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
└── scripts/                   # Helper scripts
    ├── generate_data_metadata.py
    ├── test_model_registry.py
    └── verify_imports.py
```

## 🔑 Core Concepts

### Hybrid Model IDs

Models use hybrid IDs that encode both **what** (configuration) and **when** (training time):

```
Format: {config_hash}_{timestamp}_{random_suffix}
Example: 2e30bea4e8f93845_20251108_153045_a3f9c2
         └─────┬─────┘ └──────┬──────┘ └──┬──┘
           Intent      Implementation   Safety
```

**Benefits**:
- Track multiple training runs of same configuration
- Natural chronological ordering
- Find similar models via config hash
- Full audit trail for compliance

### Complete Reproducibility

Every model links to complete provenance chain:

```
Model → ModelMetadata → DatasetMetadata → Raw Data File
  ↓          ↓                 ↓                ↓
Sharpe   Hyperparams      Data Quality      OHLCV
Metrics  Features         99.94%            172K bars
         Training Range   Asset: BTC        July-Nov 2025
```

### Data Leakage Prevention

Strict temporal ordering prevents future data contamination:

```python
# TimeSeriesSplitter ensures:
assert train_max_timestamp < validation_min_timestamp
assert lookback_window < split_start_timestamp
```

## 📊 Available Components

**Configuration** (6 Pydantic models):
- BaseConfig, DataConfig, FeatureConfig, StrategyConfig, BacktestConfig, RiskConfig

**Data Pipeline**:
- DataLoader (Parquet/CSV), DataValidator, DatasetMetadata

**Feature Engineering** (8 indicators):
- SMA, EMA, RSI, MACD, ROC, Bollinger Bands, ATR, VolumeMA

**Validation**:
- TimeSeriesSplitter, WalkForwardValidator, BacktestEngine, PerformanceMetrics

**Model Registry**:
- ModelMetadata, ModelRegistry, Hybrid ID system

## 🎓 Next Steps

**Ready to implement your first strategy?** See [STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) for a complete walkthrough.

**Key Documentation**:
- [.claude/PROJECT_CONTEXT.md](.claude/PROJECT_CONTEXT.md) - Complete project overview
- [.claude/STATUS.md](.claude/STATUS.md) - Current status and progress
- [.claude/DECISIONS.md](.claude/DECISIONS.md) - Architectural Decision Records
- [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) - Strategy implementation guide

## 📈 Current Status

**Completed** (17 milestones):
- ✅ Project infrastructure & configuration system
- ✅ Logging infrastructure
- ✅ Data pipeline with quality validation
- ✅ Feature engineering (8 indicators)
- ✅ Data leakage prevention
- ✅ Validation framework (walk-forward, backtesting)
- ✅ Dataset metadata tracking
- ✅ Model registry with hybrid IDs
- ✅ Unit tests (21 passing)

**Next Up**:
- First strategy implementation (BTC perps, momentum-based)
- Second strategy (modularity proof)
- Paper trading integration
- Docker deployment

## 🔬 Testing

```bash
# Run all tests
pytest tests/unit/ -v

# Current results: 21/21 passing ✓
# - 7 tests: Dataset metadata
# - 9 tests: Model registry
# - 5 tests: Hybrid ID system
```

## 📝 Philosophy

1. **Research First**: Optimized for rapid iteration and experimentation
2. **Validation Rigorous**: Walk-forward validation, realistic backtesting
3. **Production Ready**: Well-documented, tested, reproducible
4. **Modular**: Independent strategies, centralized infrastructure

## 🤝 Contributing

See [.claude/PROJECT_CONTEXT.md](.claude/PROJECT_CONTEXT.md) for architectural context and design decisions.

## 📄 License

[Your License Here]
