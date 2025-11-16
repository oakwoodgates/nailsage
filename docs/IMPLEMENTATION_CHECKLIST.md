# NailSage Implementation Checklist

Track progress toward MVP completion.

**Current Status**: 21/25 milestones complete (84%)

## Phase 1: Foundation ✅

- [x] **Project Structure** - Core directory layout, config system
- [x] **Data Pipeline** - OHLCV loading, validation, metadata tracking
- [x] **Feature Engine** - Technical indicators, caching, modular design
- [x] **Target Creation** - Classification target module with 3-class labeling
- [x] **Model Training** - XGBoost integration, hybrid ID system
- [x] **Model Registry** - SQLite-based tracking with metadata

## Phase 2: Validation & Testing ✅

- [x] **Time Series Splitter** - Walk-forward validation framework
- [x] **Backtest Engine** - Position simulation with fees and slippage
- [x] **Walk-Forward Validator** - Multi-split validation pipeline
- [x] **Unit Tests** - Core components tested (data, models, portfolio)
- [x] **Integration Tests** - End-to-end workflow verification

## Phase 3: First Strategy ✅

- [x] **Strategy Config System** - YAML-based strategy definitions
- [x] **Momentum Classifier v1** - 3-class short-term BTC strategy
- [x] **Training Script** - Generic script for 3-class classification
- [x] **Validation Script** - Walk-forward validation with backtesting
- [x] **Initial Results** - First trained model with validation metrics

## Phase 4: Modularity Proof ✅

- [x] **Second Strategy (Prove Modularity)** - SOL swing classifier on 4h timeframe
  - Different asset (SOL vs BTC)
  - Different timeframe (4h vs 15m)
  - Different prediction target (12 bars / 2 days vs 4 bars / 1 hour)
  - **Zero code changes required** - Modularity proven!

## Phase 5: Code Quality & Refactoring ✅

- [x] **Code Review** - Identify duplication and refactoring opportunities
- [x] **Target Module Extraction** - Moved `create_3class_target()` to shared module
- [x] **Git Strategy** - Updated `.gitignore` for results/ and data/

## Phase 6: Portfolio Coordination ✅

- [x] **Portfolio Coordinator MVP** - Phase 1 pass-through coordinator
  - [x] Position tracking (Position dataclass)
  - [x] Signal handling (StrategySignal dataclass)
  - [x] Coordinator logic (PortfolioCoordinator class)
  - [x] Safety checks (max positions, max exposure limits)
  - [x] Unit tests (21 tests, 100% passing)

## Phase 7: Live Trading Infrastructure 🔄

- [ ] **Exchange Connector** - Binance Futures API integration
  - [ ] Authentication and rate limiting
  - [ ] Order execution (market, limit orders)
  - [ ] Position management
  - [ ] WebSocket price feeds

- [ ] **Live Strategy Runner** - Real-time signal generation
  - [ ] Feature computation on live data
  - [ ] Model inference pipeline
  - [ ] Signal emission to coordinator

- [ ] **Risk Manager** - Additional safety layer
  - [ ] Daily loss limits
  - [ ] Position size validation
  - [ ] Emergency stop mechanisms

- [ ] **Monitoring & Alerts** - Observability layer
  - [ ] Position tracking dashboard
  - [ ] Performance metrics logging
  - [ ] Error alerting (email/SMS)

---

## Next Steps

1. **Exchange Connector** - Begin live trading infrastructure
2. **Live Strategy Runner** - Connect strategies to real-time data
3. **SOL Validation** (Optional) - Run walk-forward validation on SOL strategy
4. **Paper Trading** - Test with Binance testnet before going live

---

## Notes

- **Backward Compatibility**: Not a concern during alpha phase (ADR-013)
- **Training Script Reusability**: Current scripts work for all 3-class classification strategies
- **Future Enhancements**: Will need factory patterns for different target types (2-class, regression) and model types (LightGBM, RandomForest)
- **Git Strategy**:
  - Keep in git: `configs/strategies/`, `strategies/`, `models/metadata/`
  - Ignore: `results/`, `data/raw/`, `models/trained/`
  - Backup separately: Raw data and trained models (cloud storage)

---

**Last Updated**: 2025-11-15
