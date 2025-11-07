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

## Current Status
**Phase**: Foundation & Setup
**Next Milestone**: Complete project scaffolding and core data pipeline
