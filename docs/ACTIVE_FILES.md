# Active Files Reference

> **Last Updated**: 2025-11-27
> **Purpose**: Single source of truth for active vs deprecated files in the codebase

This document tracks which files are currently in use, which have been deprecated, and why. Use this to avoid confusion about which implementations to modify.

---

## Current System Architecture

### Entry Points (Active)

| File | Purpose | Used By | Status |
|------|---------|---------|--------|
| [execution/cli/run_multi_strategy.py](../execution/cli/run_multi_strategy.py) | **PRIMARY ENTRY POINT** - Multi-strategy paper trading for Docker | Docker containers | ✅ **ACTIVE** |
| [training/cli/train_model.py](../training/cli/train_model.py) | Generic model training with walk-forward validation | Manual training | ✅ **ACTIVE** |
| [training/cli/validate_model.py](../training/cli/validate_model.py) | Standalone model validation | Manual validation | ✅ **ACTIVE** |
| [training/cli/run_backtest.py](../training/cli/run_backtest.py) | Quick backtesting | Manual backtesting | ✅ **ACTIVE** |

### Core Execution Modules (Active)

| File | Purpose | Used By | Status |
|------|---------|---------|--------|
| [execution/runner/live_strategy.py](../execution/runner/live_strategy.py) | Live strategy orchestrator with separated concerns | run_multi_strategy.py | ✅ **ACTIVE** |
| [execution/runner/trade_execution_pipeline.py](../execution/runner/trade_execution_pipeline.py) | Orchestrates prediction → signal → execution workflow | LiveStrategy | ✅ **ACTIVE** |
| [execution/runner/candle_close_detector.py](../execution/runner/candle_close_detector.py) | Detects candle closes for triggering predictions | LiveStrategy | ✅ **ACTIVE** |
| [execution/inference/predictor.py](../execution/inference/predictor.py) | Model inference and prediction | TradeExecutionPipeline | ✅ **ACTIVE** |
| [execution/inference/signal_generator.py](../execution/inference/signal_generator.py) | Converts predictions to trading signals | TradeExecutionPipeline | ✅ **ACTIVE** |
| [execution/simulator/order_executor.py](../execution/simulator/order_executor.py) | Simulates order execution with fees/slippage | TradeExecutionPipeline | ✅ **ACTIVE** |
| [execution/tracking/position_tracker.py](../execution/tracking/position_tracker.py) | Tracks open positions and P&L | LiveStrategy | ✅ **ACTIVE** |
| [execution/persistence/state_manager.py](../execution/persistence/state_manager.py) | Database operations for strategies, positions, trades | All execution modules | ✅ **ACTIVE** |

### Data & Streaming (Active)

| File | Purpose | Used By | Status |
|------|---------|---------|--------|
| [execution/websocket/client.py](../execution/websocket/client.py) | Kirby WebSocket client for real-time market data | run_multi_strategy.py | ✅ **ACTIVE** |
| [execution/streaming/candle_buffer.py](../execution/streaming/candle_buffer.py) | Multi-strategy candle buffer with chronological sorting | run_multi_strategy.py | ✅ **ACTIVE** |

---

## Deprecated Files (Removed)

These files have been **DELETED** and should not be referenced in new code or documentation.

| File | Removed Date | Reason | Replacement |
|------|--------------|--------|-------------|
| `execution/runner/live_strategy_refactored.py` | 2025-11-26 | Renamed to remove "refactored" suffix | [execution/runner/live_strategy.py](../execution/runner/live_strategy.py) |
| `scripts/run_paper_trading.py` | 2025-11-26 | Single-strategy runner deprecated in favor of multi-strategy | [execution/cli/run_multi_strategy.py](../execution/cli/run_multi_strategy.py) |
| `scripts/run_sol_paper_trading.py` | 2025-11-26 | Duplicate of run_paper_trading.py for SOL | [execution/cli/run_multi_strategy.py](../execution/cli/run_multi_strategy.py) |

### Deprecated Documentation

| File | Status | Replacement |
|------|--------|-------------|
| [docs/deprecated/SOL_PAPER_TRADING_GUIDE.md](deprecated/SOL_PAPER_TRADING_GUIDE.md) | ⚠️ **ARCHIVED** 2025-11-27 | [docs/DOCKER.md](DOCKER.md) |
| [docs/deprecated/STRATEGY_GUIDE.md](deprecated/STRATEGY_GUIDE.md) | ⚠️ **ARCHIVED** 2025-11-27 | [docs/MODEL_TRAINING.md](MODEL_TRAINING.md) |
| [docs/deprecated/PHASE_10_FEATURES.md](deprecated/PHASE_10_FEATURES.md) | ⚠️ **ARCHIVED** 2025-11-27 | [docs/DECISIONS.md](DECISIONS.md) (ADR-014, ADR-015) |

---

## Architecture Evolution

### Phase 1: Monolithic LiveStrategy (Deprecated)
- **File**: `execution/runner/live_strategy.py` (original, ~582 lines)
- **Structure**: All logic in one class
- **Status**: ❌ Deleted 2025-11-26

### Phase 2: Refactored LiveStrategy (Current)
- **File**: `execution/runner/live_strategy.py` (current, ~272 lines)
- **Structure**:
  - `LiveStrategy` - Orchestrator
  - `CandleCloseDetector` - Candle close detection
  - `TradeExecutionPipeline` - Trading workflow
- **Status**: ✅ **ACTIVE**
- **Benefits**:
  - Separated concerns (testability)
  - Easier to mock for testing
  - Clearer control flow
  - Easier to extend (e.g., risk management)

### Phase 3: Multi-Strategy Execution (Current)
- **File**: `execution/cli/run_multi_strategy.py`
- **Structure**:
  - Single WebSocket connection per exchange
  - Shared candle buffer across strategies
  - Exchange-level fault isolation
- **Status**: ✅ **ACTIVE**
- **Benefits**:
  - Scale to 100s of strategies without 100s of WebSocket connections
  - Binance outage doesn't affect Hyperliquid
  - Docker-based deployment

---

## Training Architecture

### Current (Generic, Configuration-Driven)

All models use the **same training infrastructure** with strategy-specific YAML configs:

| Script | Purpose |
|--------|---------|
| [training/cli/train_model.py](../training/cli/train_model.py) | Generic training + walk-forward validation |
| [training/cli/validate_model.py](../training/cli/validate_model.py) | Standalone validation |
| [training/cli/run_backtest.py](../training/cli/run_backtest.py) | Quick backtesting |

**Configuration**: `strategies/*.yaml`

**Benefits**:
- One bug fix benefits all strategies
- No code duplication
- Easy to add new strategies (just create YAML)

### Legacy (Strategy-Specific Scripts)

**Status**: ⚠️ **DEPRECATED** (See [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) deprecation notice)

Old approach used per-strategy training scripts:
- `strategies/short_term/train_btc_momentum.py`
- `strategies/long_term/train_sol_swing.py`

These are kept for reference but **should not be used for new strategies**.

---

## Key Directories

```
nailsage/
├── strategies/         # Strategy YAML configs (✅ ACTIVE)
├── execution/
│   ├── runner/
│   │   ├── live_strategy.py   # Main orchestrator (✅ ACTIVE)
│   │   ├── trade_execution_pipeline.py (✅ ACTIVE)
│   │   └── candle_close_detector.py (✅ ACTIVE)
│   ├── inference/             # Predictor, SignalGenerator (✅ ACTIVE)
│   ├── simulator/             # OrderExecutor (✅ ACTIVE)
│   ├── tracking/              # PositionTracker (✅ ACTIVE)
│   ├── persistence/           # StateManager (✅ ACTIVE)
│   ├── streaming/             # CandleBuffer (✅ ACTIVE)
│   └── websocket/             # KirbyWebSocketClient (✅ ACTIVE)
├── scripts/
│   ├── run_multi_strategy.py  # PRIMARY ENTRY POINT (✅ ACTIVE)
│   ├── train_model.py         # Generic training (✅ ACTIVE)
│   ├── validate_model.py      # Validation (✅ ACTIVE)
│   └── run_backtest.py        # Backtesting (✅ ACTIVE)
├── strategies/                # Legacy strategy-specific code (⚠️ DEPRECATED)
└── docs/
    ├── ACTIVE_FILES.md        # This file (✅ ACTIVE)
    ├── DATABASE.md            # Database schema and usage (✅ ACTIVE)
    ├── DECISIONS.md           # Architectural Decision Records (✅ ACTIVE)
    ├── DOCKER.md              # Docker guide (✅ ACTIVE)
    ├── MODEL_TRAINING.md      # Training guide (✅ ACTIVE)
    ├── PROJECT_CONTEXT.md     # Project overview (✅ ACTIVE)
    ├── README.md              # Documentation index (✅ ACTIVE)
    └── deprecated/            # Archived documentation
        ├── PHASE_10_FEATURES.md
        ├── SOL_PAPER_TRADING_GUIDE.md
        └── STRATEGY_GUIDE.md
```

---

## Quick Reference

### "I need to add a new strategy" → What do I do?

1. Create YAML config in `strategies/my_strategy_v1.yaml`
2. Train: `python training/cli/train_model.py --config configs/strategies/my_strategy_v1.yaml`
3. Deploy: Update `docker-compose.yml` environment variables
4. See [MODEL_TRAINING.md](MODEL_TRAINING.md) for details

### "I need to modify trading logic" → What file do I edit?

| Task | File |
|------|------|
| Change how signals are generated | [execution/inference/signal_generator.py](../execution/inference/signal_generator.py) |
| Change prediction logic | [execution/inference/predictor.py](../execution/inference/predictor.py) |
| Change order execution (fees, slippage) | [execution/simulator/order_executor.py](../execution/simulator/order_executor.py) |
| Change position tracking | [execution/tracking/position_tracker.py](../execution/tracking/position_tracker.py) |
| Change candle close detection | [execution/runner/candle_close_detector.py](../execution/runner/candle_close_detector.py) |
| Change overall orchestration | [execution/runner/live_strategy.py](../execution/runner/live_strategy.py) |

### "I need to run paper trading" → What command?

**Docker (Production)**:
```bash
docker compose up -d
docker logs -f nailsage-binance
```

**Local (Testing)**:
```bash
python execution/cli/run_multi_strategy.py
```

---

## Maintenance Notes

**When adding new files**:
1. Add entry to appropriate section above
2. Mark status as ✅ **ACTIVE**
3. Document purpose and relationships

**When deprecating files**:
1. Move from "Active" to "Deprecated" section
2. Add removal date and reason
3. Update all documentation references
4. Add deprecation notice to file itself (if keeping for reference)

**When renaming files**:
1. Update all imports across codebase
2. Update this document
3. Update docker-compose.yml if applicable
4. Update GitHub workflows if applicable

---

**Questions about active files?** Check this document first. If unclear, check git history or ask maintainers.
