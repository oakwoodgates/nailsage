# NailSage Implementation Checklist

Track progress toward MVP completion.

**Current Status**: 35/35 milestones complete (100%) ✅ MVP COMPLETE!

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

## Phase 9: Live Inference Pipeline ✅

- [x] **Feature Streaming** - Real-time feature computation
  - [x] Stream candles from buffer to feature engine
  - [x] Compute indicators on sliding window
  - [x] Handle missing/incomplete data

- [x] **Model Predictor** - Live inference
  - [x] Load trained models from registry (`execution/inference/predictor.py`)
  - [x] Async prediction (use asyncio.to_thread for CPU-bound work)
  - [x] Prediction caching and validation

- [x] **Signal Generator** - Convert predictions to signals
  - [x] Threshold-based signal generation (`execution/inference/signal_generator.py`)
  - [x] Confidence filtering
  - [x] Signal deduplication (cooldown mechanism)

- [x] **Order Simulator** - Paper trading execution
  - [x] Simulated order placement (`execution/simulator/order_executor.py`)
  - [x] Realistic fees and slippage (0.1% fee, 5bps slippage)
  - [x] Position tracking (`execution/tracking/position_tracker.py`)
  - [x] P&L calculation (unrealized and realized)

- [x] **Live Strategy Orchestrator** - Complete pipeline
  - [x] `execution/runner/live_strategy.py` - Orchestrates all components
  - [x] Candle-close detection (only trade when candles close)
  - [x] Historical candle support (batch loading on startup)
  - [x] WebSocket integration with Kirby API
  - [x] Database state persistence

---

## Phase 10: Model Quality Improvements

### P0 - Critical (Data Integrity & Valid Results) ✅

- [x] **Exclude OHLCV from features** - Prevent data leakage (predicting close from close)
  - [x] Update `train_momentum_classifier.py` to drop OHLCV columns
  - [x] Update `validate_momentum_classifier.py` to match
  - [x] Verified: 18 features (was 24 with OHLCV)

- [x] **Probability-based filtering** - Only trade on high-confidence predictions
  - [x] Add `predict_proba()` support to validation script
  - [x] Add `confidence_threshold` config option in TargetSection
  - [x] Filter signals in validation based on threshold
  - [x] Verified: 17.6% filtered at 50% threshold

- [x] **True walk-forward validation** - Retrain model each split
  - [x] Add `--retrain` flag to validation script
  - [x] Train fresh model on each split's training data
  - [x] Verified: "Training fresh random_forest model on split 1..."

### P1 - High (Trading Improvements)

- [x] **Binary classification target** - Long/Short only (no neutral class)
  - [x] Create `create_binary_target()` in targets module
  - [x] Add `target.classes: 2` config support
  - [x] Update training/validation to handle binary case

- [x] **Trade cooldown/debounce** - Minimum bars between trades
  - [x] Add `min_bars_between_trades` to BacktestSection config
  - [x] Implement `apply_trade_cooldown()` in validation script
  - [x] Suppress signals within cooldown window

- [x] **Position sizing by confidence** - Scale position with prediction probability
  - [x] Update `BacktestEngine.run()` to accept confidences series
  - [x] Scale position size: base_size * (confidence - 0.5) * 2
  - [x] Pass max probabilities from validation to backtest

- [x] **Hyperparameter optimization** - Automated tuning
  - [x] Add Optuna integration (`scripts/optimize_hyperparameters.py`)
  - [x] Define search space per model type (XGBoost, LightGBM, RandomForest)
  - [x] Save best params to JSON (results/optimization/)
  - [x] Tested: 10 trials improved F1 from baseline to 0.3527

### P2 - Medium (Advanced Features)

- [ ] **Multi-timeframe features** - Higher TF context for short TF models
- [ ] **Regime detection** - Trending vs ranging market classification
- [ ] **Feature selection** - SHAP/permutation importance filtering
- [ ] **Ensemble models** - Voting/stacking multiple models

### P3 - Future (Research)

- [ ] **Reinforcement learning** - Train on PnL not classification
- [ ] **Online learning** - Incremental model updates
- [ ] **Alternative targets** - Regression, ranking, custom losses
- [ ] **Order book features** - Depth, imbalance (requires new data)

---

## 🎉 MVP Complete! What's Next?

The core NailSage paper trading MVP is **feature complete**! Here are potential next steps:

### Production Readiness
1. **Request 500 historical candles** - Change from test (100) to production (500) in [run_paper_trading.py](../scripts/run_paper_trading.py:107)
2. **Run extended paper trading** - Let engine run for 24-48 hours to verify stability
3. **Monitor performance** - Track trades, P&L, and system health
4. **Add alerting** - Email/SMS alerts for errors, large losses, etc.

### System Improvements
5. **Monitoring Dashboard** (Separate project) - React/Vue dashboard for real-time monitoring
6. **Multi-strategy testing** - Run multiple strategies simultaneously
7. **Performance optimization** - Profile and optimize hot paths
8. **Additional safety checks** - Max drawdown limits, circuit breakers

### Strategy Development
9. **Feature engineering** - Experiment with new indicators
10. **Hyperparameter tuning** - Optimize model parameters
11. **Ensemble methods** - Combine multiple models
12. **New strategies** - Test different assets, timeframes, approaches

---

## Notes

- **Architecture Scope**: NailSage = Model training + paper trading engine + API server (NOT a full trading platform)
- **Real Trading**: Handled by separate system (outside NailSage scope)
- **Frontend**: Separate project that consumes NailSage's REST + WebSocket APIs
- **Backward Compatibility**: Not a concern during alpha phase (ADR-013)
- **Algorithm Support**: Factory pattern supports XGBoost, LightGBM, RandomForest, ExtraTrees
- **Kirby Integration**: WebSocket client with historical candle support (batch loading + live streaming)
  - **Historical Messages**: Batch format with `count` field and `data` array
  - **Live Messages**: Single candle format with `data` object
  - **Parameter**: Use `history` (not `historical_candles`) in subscription request
- **Git Strategy**:
  - Keep in git: `configs/`, `strategies/`, `execution/` (code only)
  - Ignore: `results/`, `data/`, `models/trained/`, `execution/state/*.db`, `.env`
  - Backup separately: Raw data, trained models, production databases

---

**Last Updated**: 2025-11-20

## Recent Additions (2025-11-20)

### Historical Candle Support
- ✅ Fixed WebSocket subscription parameter (`history` not `historical_candles`)
- ✅ Updated Pydantic models to handle batch historical messages
- ✅ Implemented candle-close detection (only trade when candles close, not on every update)
- ✅ Verified 100 historical candles load correctly from Kirby
- ✅ Complete end-to-end pipeline tested and working

### Paper Trading Engine
- ✅ Full paper trading engine operational
- ✅ Connects to Kirby WebSocket API
- ✅ Loads trained models from registry
- ✅ Runs live inference on real-time data
- ✅ Generates trading signals with confidence filtering
- ✅ Simulates order execution with realistic fees/slippage
- ✅ Tracks positions and P&L in database
- ✅ Graceful shutdown with signal handlers
