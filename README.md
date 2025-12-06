# NailSage

**ML Trading Research Platform for Cryptocurrency Markets**

Named after the Great Nailsage Sly, trainer of the Nailmasters, from Hollow Knight.

## 🎯 Overview

NailSage is a production-ready ML trading research platform designed for building, testing, and deploying machine learning trading strategies with rigorous validation and complete reproducibility.

**Status**: Production ML Trading Platform ✅ - Live paper trading operational

## ✨ Key Features

- **✅ Complete Metadata Tracking**: Full data and model provenance for reproducibility
- **✅ Data Leakage Prevention**: Strict temporal ordering with walk-forward validation
- **✅ Realistic Backtesting**: Transaction costs, slippage, and leverage simulation
- **✅ Hybrid Model Registry**: Track configuration intent and training history
- **✅ Dynamic Feature Engineering**: 18 technical indicators computed on-the-fly
- **✅ Modular Architecture**: Independent strategies with centralized model management
- **✅ Binary Classification Models**: SHORT/LONG signals with confidence filtering
- **✅ Real-Time Execution**: Live paper trading with realistic market simulation
- **✅ Risk Management**: Per-strategy bankrolls with automatic position sizing
- **✅ Transparent Logging**: Complete audit trail of signal generation and execution
- **✅ Production Deployment**: Docker-based multi-strategy execution
- **✅ Walk-Forward Validation**: Time series cross-validation preventing data leakage

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

### Train and Validate Models

```bash
# Unified train + validate (walk-forward, per-split retrain, saves JSON)
python training/cli/run_train_validate.py --config strategies/dev_scalper_1m_v1.yaml

# Train only
python training/cli/run_train_validate.py --config strategies/dev_scalper_1m_v1.yaml --train-only

# Validate existing model
python training/cli/validate_model.py --config strategies/dev_scalper_1m_v1.yaml --model-id MODEL_ID

# Quick backtest
python training/cli/run_backtest.py --config strategies/dev_scalper_1m_v1.yaml --model-id MODEL_ID
```

### Run Tests

```bash
# Run unit tests (no external env required)
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=. --cov-report=term-missing

# Optional end-to-end training pipeline test (requires RUN_E2E_TRAINING=1)
RUN_E2E_TRAINING=1 pytest tests/integration/training/test_e2e_training_pipeline.py -q

# Integration tests (API + ML pipelines); websocket test is skipped by default
pytest tests/integration -q
```

## Logging

- Training/validation/backtest emit JSON-friendly logs with contextual fields (`strategy`, `version`, `run_id`) and event tags (e.g., `train_start`, `train_timings`, `validation_split`, `validation_aggregate`, `feature_cache_hit`).
- Use `training/cli/run_train_validate.py --summary` for concise metrics output, `--dry-run` for schema/config validation only, and `--force-cache-bust` to bypass feature cache.

### 🐳 Docker Quick Start (Recommended for Paper Trading)

Run strategies in production-ready Docker containers with PostgreSQL:

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your Kirby API credentials

# 2. Start all services
docker compose up -d

# 3. View logs
docker logs nailsage-binance -f

# 4. Check status
docker compose ps

# 5. Stop services
docker compose down
```

**What's running:**
- PostgreSQL database (port 5433)
- FastAPI dashboard API (port 8001)
- Strategy containers (Binance, Hyperliquid)

**Development workflow:**
```bash
# Make code changes, then rebuild
docker compose build nailsage-binance && docker compose up -d nailsage-binance

# View live predictions
docker logs nailsage-binance --tail 100 -f
```

**Full documentation:** See [docs/DOCKER.md](docs/DOCKER.md) for complete setup, deployment, and troubleshooting guides.

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
├── configs/                   # Default configuration files
├── strategies/          # Strategy YAML configs (not versioned)
├── data/                      # Data management
│   ├── loader.py             # Load OHLCV data (Parquet/CSV)
│   ├── validator.py          # Data quality validation
│   ├── metadata.py           # Dataset provenance tracking
│   ├── generate_metadata.py  # Metadata generation utility
│   └── raw/                  # Raw OHLCV data storage
├── features/                  # Feature engineering
│   ├── engine.py             # Dynamic feature computation
│   ├── indicators/           # 8 technical indicators
│   └── cache/                # Feature cache storage (optional)
├── training/                  # ML training & backtesting
│   ├── cli/                  # Training command-line tools
│   │   ├── train_model.py    # Main training entry point
│   │   ├── run_backtest.py   # Backtesting entry point
│   │   ├── validate_model.py # Standalone validation
│   │   └── optimize_hyperparameters.py # Hyperparameter optimization
│   ├── pipeline.py           # TrainingPipeline orchestrator (timing, seeding)
│   ├── data_pipeline.py      # Data loading and preparation (schema checks, cache)
│   ├── signal_pipeline.py    # Signal generation and filtering (guards regression)
│   ├── validator.py          # Walk-forward validation (per-split retrain)
│   ├── backtest_pipeline.py  # Backtesting workflow (risk/exec parity)
│   └── targets.py            # Target variable creation
├── execution/                 # Paper trading & live execution
│   ├── cli/                  # Execution command-line tools
│   │   ├── run_multi_strategy.py # Multi-strategy paper trading
│   │   ├── check_paper_trading_stats.py # Statistics checker
│   │   ├── test_websocket_integration.py # WebSocket testing
│   │   ├── test_signal_save.py # Signal testing
│   │   └── debug_kirby_messages.py # Kirby debugging
│   ├── portfolio/            # Portfolio coordination & signals
│   │   ├── coordinator.py    # PortfolioCoordinator class
│   │   ├── position.py       # Position tracking
│   │   └── signal.py         # StrategySignal class
│   ├── inference/            # Model inference for live trading
│   ├── persistence/          # Database state management
│   ├── risk/                 # Risk management
│   ├── runner/               # Live strategy orchestration
│   ├── simulator/            # Order execution simulation
│   ├── streaming/            # Real-time data processing
│   ├── tracking/             # Position management
│   ├── websocket/            # Live market data connection
│   └── state/                # Database files
├── models/                    # Model registry & metadata
│   ├── metadata.py           # ModelMetadata (hybrid IDs)
│   ├── registry.py           # Centralized model storage
│   ├── utils.py              # Model utilities
│   ├── trained/              # Serialized models
│   └── metadata/             # Model metadata (JSON)
├── api/                       # FastAPI REST/WebSocket API
│   ├── routers/              # Endpoint routers
│   │   ├── strategies.py     # Strategy management
│   │   ├── arenas.py         # Arena metadata (exchange, pair, interval)
│   │   ├── positions.py      # Position tracking
│   │   └── trades.py         # Trade history
│   ├── services/             # Business logic layer
│   ├── schemas/              # Pydantic models
│   └── websocket/            # Real-time updates
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
│       ├── test_kirby_websocket.py # WebSocket integration
│       └── test_model_registry_demo.py # Model registry demo
└── scripts/                   # Development utilities
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

## 📚 Documentation

**Getting Started**:
- [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) - Complete training and validation guide
- [docs/DOCKER.md](docs/DOCKER.md) - Docker deployment and paper trading
- [docs/ACTIVE_FILES.md](docs/ACTIVE_FILES.md) - Codebase structure reference

**API & Integration**:
- [docs/API.md](docs/API.md) - REST API for portfolio management
- [docs/WEBSOCKET.md](docs/WEBSOCKET.md) - Real-time WebSocket connections
- [docs/DATABASE.md](docs/DATABASE.md) - Database schema and operations

**Architecture**:
- [docs/DECISIONS.md](docs/DECISIONS.md) - Key architectural decisions
- [docs/FEATURE_SCHEMA_USAGE.md](docs/FEATURE_SCHEMA_USAGE.md) - Feature engineering details


## 🔬 Testing

```bash
# Run all tests
pytest tests/unit/ -v

# Current results: 145/145 passing ✓
# - 28 tests: SignalGenerator (confidence filtering, cooldown, deduplication)
# - 21 tests: ModelPredictor (async inference, caching, feature computation)
# - 28 tests: OrderExecutor (fees, slippage, order validation)
# - 26 tests: Phase 10 features (binary target, confidence sizing, cooldown)
# - 21 tests: Portfolio coordinator
# - 9 tests: Model registry
# - 7 tests: Dataset metadata
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
