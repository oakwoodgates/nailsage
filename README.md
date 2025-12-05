# NailSage

**ML Trading Research Platform for Cryptocurrency Markets**

Named after the Great Nailsage Sly, trainer of the Nailmasters, from Hollow Knight.

## 🎯 Overview

NailSage is a production-ready ML trading research platform designed for building, testing, and deploying machine learning trading strategies with rigorous validation and complete reproducibility.

**Current Status**: MVP Complete (35/35 milestones, 100%) ✅ - Paper trading operational with model quality improvements

**Phase 1 Focus**: Classical ML (XGBoost, LightGBM, Random Forest) with walk-forward validation

## ✨ Key Features

- **✅ Complete Metadata Tracking**: Full data and model provenance for reproducibility
- **✅ Data Leakage Prevention**: Strict temporal ordering with walk-forward validation
- **✅ Realistic Backtesting**: Transaction costs, slippage, and leverage simulation
- **✅ Hybrid Model Registry**: Track configuration intent and training history
- **✅ Dynamic Feature Engineering**: 18 technical indicators computed on-the-fly
- **✅ Modular Architecture**: Independent strategies with centralized model management
- **✅ Binary Classification Models**: Phase 10 aggressive trading with SHORT/LONG signals
- **✅ Confidence-Based Filtering**: Minimum confidence thresholds for signal generation
- **✅ Signal Cooldown**: Prevents spam with minimum bars between signals
- **✅ Real-Time P&L Updates**: Position profitability updated every candle
- **✅ Transparent Decision Logging**: See why signals are generated or suppressed
- **✅ Smart Feature Caching**: Enabled for training/backtesting, disabled for live trading
- **✅ Per-Strategy Bankroll**: Isolated $10k bankroll per strategy with percentage-based sizing
- **✅ Automatic Position Sizing**: Trades sized at 10% of current strategy bankroll
- **✅ Bankroll Depletion Protection**: Strategies auto-pause when bankroll <= $0
- **✅ Arena Metadata**: Trading arena data (exchange, pair, interval) synced from Kirby API

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
# Train with walk-forward validation (saves results to JSON)
python training/cli/train_model.py --config configs/strategies/dev_scalper_1m_v1.yaml

# Validate existing model
python training/cli/validate_model.py --config configs/strategies/dev_scalper_1m_v1.yaml --model-id MODEL_ID

# Quick backtest
python training/cli/run_backtest.py --config configs/strategies/dev_scalper_1m_v1.yaml --model-id MODEL_ID
```

### Run Tests

```bash
# Run all tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=. --cov-report=term-missing
```

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
├── training/                  # ML training & backtesting
│   ├── cli/                  # Training command-line tools
│   │   ├── train_model.py    # Main training entry point
│   │   ├── run_backtest.py   # Backtesting entry point
│   │   ├── validate_model.py # Standalone validation
│   │   └── optimize_hyperparameters.py # Hyperparameter optimization
│   ├── pipeline.py           # TrainingPipeline orchestrator
│   ├── data_pipeline.py      # Data loading and preparation
│   ├── signal_pipeline.py    # Signal generation and filtering
│   ├── validator.py          # Walk-forward validation
│   ├── backtest_pipeline.py  # Backtesting workflow
│   └── targets.py            # Target variable creation
├── execution/                 # Paper trading & live execution
│   ├── cli/                  # Execution command-line tools
│   │   ├── run_multi_strategy.py # Multi-strategy paper trading
│   │   ├── check_paper_trading_stats.py # Statistics checker
│   │   ├── test_websocket_integration.py # WebSocket testing
│   │   ├── test_signal_save.py # Signal testing
│   │   └── debug_kirby_messages.py # Kirby debugging
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
└── scripts/                   # System administration scripts
    ├── generate_data_metadata.py
    ├── migrate_arenas.py
    ├── test_historical_candles.py
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

**Ready to train your first model?** See [MODEL_TRAINING.md](docs/MODEL_TRAINING.md) for comprehensive training and validation guide.

**Key Documentation**:
- [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) - Training, validation, and backtesting guide
- [docs/API.md](docs/API.md) - REST API reference (strategies, arenas, trades, positions)
- [docs/DOCKER.md](docs/DOCKER.md) - Docker deployment guide
- [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) - Strategy implementation guide (legacy)
- [.claude/PROJECT_CONTEXT.md](.claude/PROJECT_CONTEXT.md) - Complete project overview
- [.claude/STATUS.md](.claude/STATUS.md) - Current status and progress
- [.claude/DECISIONS.md](.claude/DECISIONS.md) - Architectural Decision Records

## 📈 Current Status

**MVP Complete** (35/35 milestones): ✅
- ✅ Core infrastructure & configuration (Phases 1-5)
- ✅ Data pipeline with quality validation
- ✅ Feature engineering (10 indicators)
- ✅ Validation framework (walk-forward, backtesting)
- ✅ Model registry with hybrid IDs
- ✅ Multi-algorithm support (XGBoost, LightGBM, RandomForest, ExtraTrees)
- ✅ Portfolio coordination system
- ✅ Paper trading infrastructure (Phase 8-9)
  - WebSocket client with Kirby API integration
  - Live inference pipeline
  - State persistence (SQLite)
- ✅ Model quality improvements (Phase 10)
  - Binary classification support
  - Confidence-based position sizing
  - Trade cooldown mechanism
  - Hyperparameter optimization
- ✅ Unit tests (145 passing)

**Ready for Production Testing**:
- Paper trading validation with real models
- Extended monitoring and performance tracking

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
