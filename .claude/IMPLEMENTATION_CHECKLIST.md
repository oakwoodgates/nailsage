# Implementation Checklist

**Current Progress**: 20/25 milestones (80%) - Phase: Modularity Proven ✨

## Week 1-2: Foundation ✅

### Project Setup
- [x] Create directory structure
- [x] Setup .claude/ context directory
- [x] Create pyproject.toml with dependencies
- [x] Setup .gitignore
- [x] Configure pre-commit hooks (black, ruff, mypy)
- [x] Create requirements.txt (generated from pyproject.toml)
- [x] Initialize pytest configuration

### Configuration Management
- [x] Create base Pydantic models (BaseConfig)
- [x] Implement DataConfig model
- [x] Implement FeatureConfig model
- [x] Implement StrategyConfig model
- [x] Implement BacktestConfig model
- [x] Create sample YAML configs (short_term.yaml, long_term.yaml)
- [x] Build config loader utility
- [x] Add config validation tests

### Data Pipeline
- [x] Create data schema definitions (OHLCV + num_trades)
- [x] Implement DataLoader class (Parquet → DataFrame)
- [x] Build DataValidator class (gaps, outliers, schema)
- [x] Add data quality metrics logging
- [x] Create test fixtures (sample BTC data)
- [x] Write data pipeline tests
- [x] Document data requirements and formats

### Logging Infrastructure
- [x] Setup structured logging utility
- [x] Create logger categories (data, features, training, backtest, execution)
- [x] Implement log formatting (JSON for production, pretty for dev)
- [x] Add logging configuration file
- [x] Create metrics collection framework
- [x] Write logging tests

## Week 3-4: Core Engine ✅

### Feature Engineering
- [x] Create base FeatureEngine class
- [x] Implement SMA (Simple Moving Average)
- [x] Implement EMA (Exponential Moving Average)
- [x] Implement RSI (Relative Strength Index)
- [x] Implement MACD (Moving Average Convergence Divergence)
- [x] Implement Bollinger Bands
- [x] Implement ATR (Average True Range)
- [x] Implement Volume MA
- [x] Implement momentum indicators (ROC, etc.)
- [x] Implement volatility indicators
- [x] Add lookback window validation
- [x] Create feature caching mechanism (file-based)
- [x] Write comprehensive indicator tests
- [x] Add feature computation benchmarks

### Data Leakage Prevention
- [x] Create TimeSeriesSplitter class
- [x] Implement timestamp validation logic
- [x] Add train/validation split assertions
- [x] Build lookback window calculator
- [x] Create data leakage test suite
- [x] Document leakage prevention guidelines

### Validation Framework
- [x] Implement WalkForwardValidator class
- [x] Create multiple validation window strategy
- [x] Build performance metrics calculator (Sharpe, Sortino, Calmar)
- [x] Add win rate and profit factor calculations
- [x] Implement regime detector (volatility, trend, volume)
- [x] Create backtesting engine with costs
- [x] Add transaction cost modeling (fees + slippage)
- [x] Build overfitting detection (train/val gap monitoring)
- [x] Write validation framework tests

## Week 4-5: First Strategy End-to-End ✅

### Strategy Implementation (momentum_classifier v1 - BTC/USDT Perps, 15min)
- [x] Define feature set for strategy (9 technical indicators)
- [x] Create target variable definition (3-class classification)
- [x] Implement label generation logic
- [x] Create strategy config file (momentum_classifier_v1.yaml)
- [x] Build training pipeline (strategies/short_term/train_momentum_classifier.py)
- [x] Implement model training (XGBoost baseline)
- [x] Successfully trained first model
- [ ] Add LightGBM model (deferred - XGBoost working)
- [ ] Add Random Forest model (deferred - XGBoost working)
- [ ] Create model comparison framework (deferred to multi-strategy phase)
- [ ] Build backtesting runner (next priority - use BacktestEngine)
- [ ] Generate performance report (next priority)
- [ ] Create visualization notebook (deferred)
- [ ] Write strategy tests (deferred)

### Model Management
- [x] Create model registry (file-based with metadata)
- [x] Implement model versioning (hybrid ID system: config_hash_timestamp_random)
- [x] Add model serialization (joblib/pickle)
- [x] Build model metadata tracking (features, hyperparams, metrics)
- [x] Create model loading utility
- [x] Write model registry tests (21 unit tests passing)

### Experiment Tracking
- [x] Create experiment logger (JSON files via ModelMetadata)
- [x] Track hyperparameters (stored in metadata)
- [x] Log validation metrics (accuracy tracked)
- [ ] Save backtest results (next priority)
- [ ] Build experiment comparison utility (next priority)
- [ ] Create experiment analysis notebook (deferred)

## Week 6-7: Modularity & Multi-Strategy (CURRENT PRIORITY)

### Walk-Forward Validation for momentum_classifier v1
- [ ] Run walk-forward validation using WalkForwardValidator
- [ ] Execute backtesting with BacktestEngine (transaction costs, slippage, leverage)
- [ ] Generate performance report (Sharpe, Sortino, Calmar, drawdown, win rate)
- [ ] Address class imbalance issue (97.7% neutral - rebalance or adjust thresholds)
- [ ] Document performance characteristics and regime behavior

### Second Strategy (Prove Modularity) ✅
- [x] Choose second strategy (SOL swing classifier - 4h timeframe, 2-day lookahead, 5% threshold)
- [x] Define feature set for second strategy (same 9 indicators - proven reusability)
- [x] Create strategy config file (sol_swing_classifier_v1.yaml)
- [x] Implement training pipeline (ZERO code changes - complete reuse!)
- [x] Train second model using same pipeline (model ID: 89a110800e554613_20251116_032035_92da73)
- [ ] Run walk-forward validation on second strategy
- [ ] Execute backtesting on second strategy
- [x] Verify zero shared state between strategies (different model IDs, separate metadata)
- [ ] Test concurrent strategy execution (deferred - not critical for Phase 1)
- [ ] Compare performance metrics across both strategies (pending backtest results)

### Portfolio Coordinator (MVP)
- [ ] Create PortfolioManager class (simple placeholder)
- [ ] Implement signal pass-through logic
- [ ] Add position tracking
- [ ] Calculate total exposure
- [ ] Implement basic risk limits (max positions, max exposure)
- [ ] Create portfolio state snapshot
- [ ] Write portfolio manager tests
- [ ] Document intended Phase 2 enhancements

### Risk Management
- [ ] Create RiskManager class
- [ ] Implement position sizing logic
- [ ] Add strategy-level risk checks
- [ ] Add portfolio-level risk checks
- [ ] Implement max leverage limits
- [ ] Create risk metrics calculator
- [ ] Write risk management tests

## Week 7-8: Paper Trading & State

### State Persistence
- [ ] Design SQLite schema (positions, trades, signals, predictions)
- [ ] Create database initialization script
- [ ] Implement Position ORM model
- [ ] Implement Trade ORM model
- [ ] Implement Signal ORM model
- [ ] Create database access layer
- [ ] Add migration strategy
- [ ] Build state recovery logic
- [ ] Write database tests

### Paper Trading
- [ ] Create exchange API abstraction interface
- [ ] Implement mock exchange (paper trading)
- [ ] Build order execution simulator
- [ ] Add realistic fill modeling
- [ ] Implement position reconciliation
- [ ] Create trade execution logger
- [ ] Build monitoring dashboard (notebook or simple text)
- [ ] Write paper trading tests

### Error Handling & Recovery
- [ ] Add graceful shutdown handlers
- [ ] Implement checkpoint system
- [ ] Create restart-from-checkpoint logic
- [ ] Build failure recovery tests
- [ ] Add critical error alerting (local logs)
- [ ] Document recovery procedures

## Week 8-9: Deployment

### Docker
- [ ] Create Dockerfile
- [ ] Build docker-compose.yml (dev environment)
- [ ] Add volume mounts for persistence
- [ ] Create production docker-compose.yml
- [ ] Setup environment variable management (.env)
- [ ] Add health check endpoints
- [ ] Test local Docker deployment
- [ ] Optimize image size

### Digital Ocean Deployment
- [ ] Write droplet provisioning script
- [ ] Create deployment automation script
- [ ] Setup systemd service files (or Docker Compose as service)
- [ ] Configure firewall rules
- [ ] Add SSH key management
- [ ] Implement backup scripts (models, database, logs)
- [ ] Create monitoring setup (disk, memory, CPU)
- [ ] Write deployment runbook

### Operational Tooling
- [ ] Create start/stop/restart scripts
- [ ] Build status check script
- [ ] Add log viewing utility
- [ ] Create backup automation
- [ ] Build log rotation configuration
- [ ] Write troubleshooting guide
- [ ] Document common operations

## Week 9-10: Testing & Hardening

### Comprehensive Testing
- [ ] Write unit tests for data pipeline
- [ ] Write unit tests for feature engine
- [ ] Write unit tests for validation framework
- [ ] Write unit tests for strategies
- [ ] Write integration tests (end-to-end pipeline)
- [ ] Add property-based tests (Hypothesis)
- [ ] Create regression test suite
- [ ] Measure and report code coverage (target: 80%+)
- [ ] Fix any gaps in coverage

### Performance Optimization
- [ ] Profile feature computation
- [ ] Identify DataFrame operation bottlenecks
- [ ] Optimize hot paths with Polars (if needed)
- [ ] Add parallel computation where beneficial
- [ ] Benchmark performance improvements
- [ ] Document performance characteristics

### Documentation
- [ ] Update README with implementation details
- [ ] Create ARCHITECTURE.md
- [ ] Write DEPLOYMENT.md
- [ ] Document API/module interfaces
- [ ] Create code examples and tutorials
- [ ] Write known limitations document
- [ ] Update Phase 2 roadmap
- [ ] Create contributor guide (for future developers)

## Final Validation

### Success Criteria Checklist
- [ ] 2 strategies running independently (no shared state) - 1/2 complete (momentum_classifier v1 trained)
- [ ] Walk-forward validation shows consistent behavior across regimes - NEXT PRIORITY
- [ ] Paper trading runs 2+ weeks without crashes
- [ ] System restart recovers state correctly
- [ ] Backtest results match paper trading (within expected variance)
- [x] Core framework complete with model registry and metadata tracking
- [x] 21 unit tests passing for model registry system
- [ ] 80%+ test coverage on core modules (target for week 9-10)
- [ ] Digital Ocean deployment completes in <15 minutes
- [ ] Documentation sufficient for 3-month handoff gap

## Notes
- **Alpha Development Policy (ADR-013)**: No backward compatibility required. Breaking changes allowed. Major rewrites (>500 LOC) require user approval.
- **Current Phase**: First strategy trained (momentum_classifier v1) - proving modularity next
- **Known Issues**: Class imbalance (97.7% neutral) needs addressing via threshold tuning or rebalancing
- Items can be reordered based on dependencies and discoveries
- Some items may be split into smaller tasks
- New items will be added as we learn more
- Completed items marked with [x]

## Current State Summary (as of 2025-11-15)
- **19/25 milestones complete (76%)**
- **Model ID**: 28ddedac3886db54_20251115_015256_89beaa
- **Validation Accuracy**: 95.65%
- **Code Stats**: 40+ files, ~5,500 LOC, 8 indicators, 21 tests passing
- **Data Available**: 172,756 bars of BTC/USDT perps (1m interval), 99.94% quality score
- **Next Actions**: Walk-forward validation → Second strategy → Backtest integration
