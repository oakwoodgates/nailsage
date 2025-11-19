# NailSage Implementation Checklist

Track progress toward MVP completion.

**Current Status**: 25/35 milestones complete (71%)

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

## Phase 7: Multi-Algorithm Support ✅

- [x] **Model Factory Pattern** - Support multiple ML algorithms
  - [x] XGBoost (original)
  - [x] LightGBM (comparable performance)
  - [x] Random Forest (baseline comparison)
  - [x] Extra Trees (ensemble alternative)
- [x] **Training Script Updates** - Generic training with algorithm selection
- [x] **Model Comparison** - Trained multiple algorithms on same data
  - XGBoost: 81% accuracy, -92% return
  - LightGBM: 79% accuracy, -75% return
  - Random Forest: 30% accuracy, -92% return (not suitable for crypto)
  - **First profitable model**: SOL swing LightGBM (59% acc, +20% return)

## Phase 8: Paper Trading Infrastructure ✅

- [x] **Configuration System** - Dev/prod environment support
  - [x] Pydantic-settings with environment variables
  - [x] MODE switching (dev/prod)
  - [x] WebSocket configuration with reconnection settings
  - [x] Starlisting ID mappings

- [x] **WebSocket Client** - Kirby API integration
  - [x] Async connection with authentication
  - [x] Subscribe/unsubscribe to starlistings
  - [x] Industry-standard exponential backoff reconnection
  - [x] Heartbeat monitoring (30s interval, 45s timeout)
  - [x] Message parsing with Pydantic models

- [x] **Data Models** - Kirby message format support
  - [x] Candle (time-based with 18-decimal precision)
  - [x] FundingRate, OpenInterest (partial - needs format fixes)
  - [x] SubscribeRequest, SuccessMessage, ErrorMessage
  - [x] All message types validated with Pydantic

- [x] **Candle Buffering** - Efficient ring buffer system
  - [x] CandleBuffer (fixed-size with O(1) operations)
  - [x] MultiCandleBuffer (multi-starlisting support)
  - [x] DataFrame conversion for feature engineering
  - [x] Thread-safe operations

- [x] **State Persistence** - SQLite database
  - [x] Schema for strategies, positions, trades, signals
  - [x] StateManager with ACID guarantees
  - [x] WAL mode for concurrency
  - [x] Crash recovery support

- [x] **Integration Testing** - End-to-end verification
  - [x] Connected to local Kirby Docker instance
  - [x] Received 95 candles in 60 seconds
  - [x] Buffer and database persistence verified

## Phase 9: Live Inference Pipeline 🔄

- [ ] **Feature Streaming** - Real-time feature computation
  - [ ] Stream candles from buffer to feature engine
  - [ ] Compute indicators on sliding window
  - [ ] Handle missing/incomplete data

- [ ] **Model Predictor** - Live inference
  - [ ] Load trained models from registry
  - [ ] Async prediction (use asyncio.to_thread for CPU-bound work)
  - [ ] Prediction caching

- [ ] **Signal Generator** - Convert predictions to signals
  - [ ] Threshold-based signal generation
  - [ ] Confidence filtering
  - [ ] Signal deduplication

- [ ] **Order Simulator** - Paper trading execution
  - [ ] Simulated order placement
  - [ ] Realistic fees and slippage
  - [ ] Position tracking
  - [ ] P&L calculation

---

## Next Steps

1. **Feature Streaming** - Connect candle buffer to feature engine for real-time computation
2. **Model Predictor** - Load trained models and generate predictions on live data
3. **Signal Generator** - Convert model predictions to trading signals
4. **Order Simulator** - Simulate order execution with realistic fees/slippage
5. **End-to-End Testing** - Test full pipeline from WebSocket to simulated trades
6. **Frontend Dashboard** (Separate project) - Build React/Vue dashboard for monitoring

---

## Notes

- **Architecture Scope**: NailSage = Model training + paper trading engine + API server (NOT a full trading platform)
- **Real Trading**: Handled by separate system (outside NailSage scope)
- **Frontend**: Separate project that consumes NailSage's REST + WebSocket APIs
- **Backward Compatibility**: Not a concern during alpha phase (ADR-013)
- **Algorithm Support**: Factory pattern supports XGBoost, LightGBM, RandomForest, ExtraTrees
- **Kirby Integration**: WebSocket client tested with local Docker instance (ws://localhost:8000/ws)
- **Git Strategy**:
  - Keep in git: `configs/`, `strategies/`, `execution/` (code only)
  - Ignore: `results/`, `data/`, `models/trained/`, `execution/state/*.db`, `.env`
  - Backup separately: Raw data, trained models, production databases

---

**Last Updated**: 2025-11-19
