# NailSage - Project Context

## Quick Overview
NailSage is an ML trading research platform for cryptocurrency markets (starting with Bitcoin). The goal is to build, test, and deploy machine learning trading strategies with rigorous validation and production-ready code.

## Phase 1 MVP Goals
- Build modular platform supporting multiple independent trading strategies
- Start with classical ML (XGBoost, LightGBM, Random Forest)
- Support 2+ assets (BTC spot and perps)
- Implement 2+ timeframes/strategies (short-term and long-term)
- Paper trading integration
- Docker deployment to Digital Ocean
- 80%+ test coverage on core modules

## Key Architectural Decisions

### Independent Models Philosophy
- Each strategy/model operates independently
- Models can have opposing signals (short-term long, long-term short) - this is VALID
- PortfolioCoordinator in Phase 1 is a simple MVP placeholder
- Complex portfolio optimization deferred to Phase 2

### Technology Stack (Phase 1)
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Data**: Pandas, NumPy, Polars (performance-critical paths)
- **Config**: Pydantic (type-safe configurations)
- **Testing**: Pytest, Hypothesis
- **Deployment**: Docker, Docker Compose
- **Storage**: Parquet (historical data), SQLite (state persistence)
- **Logging**: Python logging with JSON structure
- **Monitoring**: Custom lightweight metrics (no paid services)

### Data Sources
- Data ingestion platform provides OHLCV + num_trades
- Formats: Parquet (primary), CSV, API, WebSocket
- Starting granularity: 1-minute bars
- Phase 1 scope: Technical + volume indicators only

### What's NOT in Phase 1
- Deep learning models (LSTM, Transformers)
- Alternative data (funding rates, order book, on-chain metrics)
- Real money trading
- Multi-exchange support
- High-frequency (<1min bars)
- Distributed computing
- Real-time dashboards
- Paid monitoring services (Prometheus, Grafana, MLflow, W&B)

## Critical Principles

### Data Leakage Prevention
- Strict timestamp validation in train/validation splits
- Lookback-aware feature computation
- Explicit assertions: train_max_timestamp < validation_min_timestamp
- Test suite specifically for leakage detection

### Validation Rigor
- Walk-forward validation (time-series aware)
- Multiple validation windows across different market regimes
- Realistic backtesting (transaction costs, slippage)
- Overfitting detection (train/val gap, parameter sensitivity)

### Production Readiness
- Well-documented, tested code
- Graceful error handling and recovery
- State persistence and restart capability
- Deployment automation
- Operational runbooks

## Success Criteria (Phase 1 Complete)
- [ ] 2 strategies running independently
- [ ] Walk-forward validation shows consistent behavior
- [ ] Paper trading runs 2+ weeks without crashes
- [ ] System restart recovers state correctly
- [ ] Backtest results match paper trading (within variance)
- [ ] 80%+ test coverage on core modules
- [ ] Digital Ocean deployment completes in <15 minutes
- [ ] Documentation sufficient for 3-month handoff gap

## Timeline
**Target**: 8-12 weeks (flexible, iterate until solid)
**Started**: 2025-11-06

## Project Structure

```
nailsage/                      # Project root
├── config/                    # Configuration models (Pydantic classes)
│   ├── base.py               # BaseConfig with YAML loading
│   ├── data.py               # DataConfig (OHLCV loading)
│   ├── feature.py            # FeatureConfig (indicator parameters)
│   ├── strategy.py           # StrategyConfig (strategy definition)
│   ├── backtest.py           # BacktestConfig (fees, slippage)
│   └── risk.py               # RiskConfig (position sizing, limits)
├── configs/                   # Configuration files (YAML)
│   │   ├── backtest_default.yaml
│   └── risk_default.yaml
├── data/                      # Data loading code + storage
│   ├── loader.py             # DataLoader class
│   ├── validator.py          # DataValidator class
│   ├── schemas.py            # Data schemas
│   ├── metadata.py           # DatasetMetadata class
│   ├── raw/                  # Raw OHLCV data (Parquet/CSV)
│   └── processed/            # Processed datasets
├── features/                  # Feature engineering
│   ├── engine.py             # FeatureEngine (compute-on-the-fly)
│   ├── base.py               # BaseIndicator class
│   ├── indicators/           # Technical indicators
│   │   ├── moving_average.py # SMA, EMA
│   │   ├── momentum.py       # RSI, MACD, ROC
│   │   ├── volatility.py     # Bollinger Bands, ATR
│   │   └── volume.py         # VolumeMA
│   └── cache/                # Feature cache storage
├── validation/                # Validation framework
│   ├── time_series_split.py # TimeSeriesSplitter (leakage prevention)
│   ├── backtest.py           # BacktestEngine (realistic costs)
│   ├── metrics.py            # PerformanceMetrics
│   └── walk_forward.py       # WalkForwardValidator
├── strategies/                # Strategy implementations
│   ├── short_term/           # Short-term strategies
│   └── long_term/            # Long-term strategies
├── models/                    # Model training and registry
│   ├── metadata.py           # ModelMetadata dataclass
│   ├── registry.py           # ModelRegistry class
│   ├── utils.py              # Model utilities
│   ├── trained/              # Serialized models (.joblib, .pkl)
│   └── metadata/             # Model metadata JSON files
├── execution/                 # Execution and portfolio management
├── utils/                     # Utilities
│   ├── logger.py             # Structured logging
│   └── config.py             # Config loading helpers
├── tests/                     # Test suite
└── scripts/                   # Helper scripts
    ├── generate_data_metadata.py  # CLI for metadata generation
    └── test_model_registry.py     # Model registry demonstration
```

## Completed Components (Phase 1)

### ✅ Configuration System
- Type-safe Pydantic models for all configurations
- YAML serialization/deserialization
- Validation with sensible defaults
- Config loader utilities

### ✅ Data Pipeline
- DataLoader: Parquet/CSV loading with schema validation
- DataValidator: Quality checks (gaps, outliers, OHLC validity)
- DataQualityReport: Detailed reporting and scoring

### ✅ Feature Engineering
- FeatureEngine: Dynamic computation with caching
- 8 Technical Indicators: SMA, EMA, RSI, MACD, ROC, Bollinger, ATR, Volume MA
- Lookback-aware calculations
- BaseIndicator: Extensible indicator framework

### ✅ Logging Infrastructure
- Structured JSON logging for production
- Human-readable logging for development
- Category-based loggers (data, features, training, backtest, execution, validation)

### ✅ Data Leakage Prevention
- TimeSeriesSplitter: Walk-forward validation with strict temporal ordering
- Expanding/rolling window support
- Configurable gaps between train/validation
- Lookback validation to prevent feature computation leakage

### ✅ Validation Framework
- WalkForwardValidator: Complete validation pipeline
- BacktestEngine: Realistic backtesting with fees, slippage, leverage
- PerformanceMetrics: Sharpe, Sortino, Calmar, drawdown, win rate, profit factor
- Overfitting detection (train vs validation gap analysis)
- Aggregate metrics and consistency scoring

### ✅ Data Metadata Tracking
- DatasetMetadata: Complete provenance tracking (asset, quote, exchange, market type, interval)
- Metadata generation CLI: Auto-extract from filenames, validate data quality
- Flexible filename parsing: Handles multiple formats and prefixes
- Read-only validation: Never modifies source data files
- Quality metrics integration: Stores data quality scores with metadata

### ✅ Model Registry & Metadata
- ModelMetadata: Complete provenance (strategy, data, features, hyperparameters, performance)
- Hybrid ID system: {config_hash}_{timestamp}_{random_suffix} for tracking training runs
- ModelRegistry: Centralized storage with flexible querying
- Query models by: strategy, config hash, timeframe, version, metrics
- Config-aware queries: find all runs of same configuration
- Model comparison and lineage tracking
- Links ModelMetadata → DatasetMetadata for full reproducibility
- 21 unit tests passing

### ✅ First Strategy Trained
- **momentum_classifier v1**: XGBoost 3-class classifier for BTC/USDT perps
- **Data**: 172k 1m bars resampled to 15m (Jul-Nov 2025)
- **Features**: 9 technical indicators (EMA, RSI, MACD, ROC, BB, ATR, Volume MA)
- **Performance**: 95.65% validation accuracy (but class imbalance issue)
- **Model ID**: 28ddedac3886db54_20251115_015256_89beaa
- **Config**: YAML-based (configs/strategies/momentum_classifier_v1.yaml)
- **Training**: Automated pipeline (strategies/short_term/train_momentum_classifier.py)
- **Registered**: Full metadata tracking with hybrid ID

### ✅ Strategy Configuration System
- Extended StrategyConfig with nested sections (DataSection, TargetSection, ModelSection, etc.)
- YAML-based strategy definitions with 7 configuration sections
- IndicatorConfig for flexible feature specification
- Comprehensive strategy guide (docs/STRATEGY_GUIDE.md)

## Current Status
**Phase**: MVP Complete ✅ (35/35 milestones, 100%)
**Lines of Code**: ~8,000+ (production-ready)
**Test Coverage**: 145 passing unit tests
**Next Milestone**: Production validation with extended paper trading

## Complete ML & Trading Pipeline
The full system is operational end-to-end:
1. Load and validate data ✅ (with metadata tracking)
2. Compute features dynamically ✅ (10 indicators, caching enabled)
3. Split data with no leakage ✅ (TimeSeriesSplitter)
4. Train ML models ✅ (Multi-algorithm support: XGBoost, LightGBM, RandomForest, ExtraTrees)
5. Track complete provenance ✅ (Dataset + Model metadata with hybrid IDs)
6. Query and compare models ✅ (ModelRegistry with config-aware queries)
7. Backtest simulation ✅ (BacktestEngine with realistic fees/slippage)
8. Walk-forward validation ✅ (WalkForwardValidator with retrain support)
9. Paper trading infrastructure ✅ (WebSocket client, live inference, state persistence)
10. Model quality improvements ✅ (Binary classification, confidence sizing, cooldown, hyperopt)
