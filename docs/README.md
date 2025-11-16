# NailSage Documentation

**ML Trading Research Platform for Cryptocurrency Markets**

**Last Updated**: 2025-11-16
**Current Phase**: Portfolio Coordination ✅ (21/25 milestones, 84%)

---

## Quick Start

### Project Overview
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - High-level overview, goals, tech stack, and architecture
- [STATUS.md](STATUS.md) - Current project status, recent accomplishments, and next steps
- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - Detailed progress tracker (21/25 milestones)

### Strategy Development
- [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) - How to create and configure new trading strategies

### Architecture & Decisions
- [DECISIONS.md](DECISIONS.md) - Architectural Decision Records (ADRs) documenting key design choices

---

## Document Guide

### 📋 [STATUS.md](STATUS.md)
**What**: Current project status snapshot
**When to read**: Want to know what's working, what's in progress, what's next
**Key sections**:
- Recent accomplishments (Portfolio Coordinator MVP, SOL strategy, refactoring)
- Active strategies (BTC momentum classifier, SOL swing classifier)
- System capabilities and limitations
- Next milestones

### 📊 [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)
**What**: Granular progress tracker with 25 milestones
**When to read**: Want to see detailed completion status
**Key sections**:
- Phase 1-6: Foundation, Validation, First Strategy, Modularity, Refactoring, Portfolio
- Phase 7: Live Trading Infrastructure (next up)
- Notes on backward compatibility, script reusability, git strategy

### 🎯 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
**What**: High-level project vision and architecture
**When to read**: New to the project or need to understand design philosophy
**Key sections**:
- Phase 1 MVP goals
- Independent models philosophy
- Technology stack
- Success criteria
- Project structure
- Completed components

### 📖 [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md)
**What**: Practical guide for creating trading strategies
**When to read**: Want to add a new strategy
**Key sections**:
- Configuration sections (data, features, target, model, validation, backtest, risk)
- Example strategies (BTC momentum, SOL swing)
- Training and validation workflows
- Best practices

### 🏛️ [DECISIONS.md](DECISIONS.md)
**What**: Architectural Decision Records (ADRs)
**When to read**: Want to understand why specific design choices were made
**Key decisions**:
- ADR-001: Independent Models vs Portfolio Optimization
- ADR-002: Classical ML for Phase 1
- ADR-010: 3-Class Classification
- ADR-012: Hybrid Model ID System
- ADR-013: No Backward Compatibility in Alpha

---

## Project Status Summary

### ✅ What's Working
1. **Config-Driven Strategy Development** - Define strategies in YAML, zero code changes needed
2. **Modular Training Pipeline** - Generic scripts work across assets, timeframes, prediction targets
3. **Walk-Forward Validation** - Time series cross-validation with realistic backtesting
4. **Portfolio Coordination** - Multi-strategy position tracking with safety limits
5. **Model Registry** - Centralized tracking with hybrid ID system for reproducibility

### 🎯 Current Phase: Portfolio Coordination ✅
- ✅ Position and signal tracking (dataclasses with validation)
- ✅ Pass-through coordinator with safety checks
- ✅ 21 unit tests (100% passing)
- ✅ Documentation updated

### 📈 Next Phase: Live Trading Infrastructure
- [ ] Exchange Connector (Binance Futures API)
- [ ] Live Strategy Runner (real-time signal generation)
- [ ] Risk Manager (additional safety layer)
- [ ] Monitoring & Alerts

### 📊 By The Numbers
- **Progress**: 21/25 milestones (84%)
- **Code**: 40+ files, ~6,000 LOC
- **Strategies**: 2 trained (BTC short-term, SOL swing)
- **Tests**: 40+ unit tests passing
- **Assets**: BTC/USDT, SOL/USDT (perps)
- **Indicators**: 9 technical indicators

---

## Getting Started

### For New Developers
1. Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - Understand the vision and architecture
2. Read [STATUS.md](STATUS.md) - See current state and capabilities
3. Review [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) - Learn how strategies work
4. Check [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - See what's done and what's next

### For Strategy Development
1. Read [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) - Comprehensive guide
2. Look at existing configs in `configs/strategies/`
3. Review training scripts in `strategies/short_term/`
4. Check validation results in `results/walk_forward_results/`

### For Architecture Understanding
1. Read [DECISIONS.md](DECISIONS.md) - Understand key design choices
2. Review [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - See project structure
3. Check code in relevant modules (`features/`, `validation/`, `portfolio/`)

---

## Key Concepts

### Independent Models Philosophy (ADR-001)
- Each strategy operates autonomously
- Strategies can have opposing signals (this is valid!)
- Portfolio Coordinator enforces global risk limits only
- No portfolio optimization in Phase 1 (deferred to Phase 2)

### Hybrid Model IDs (ADR-012)
Format: `{config_hash}_{timestamp}_{random_suffix}`
- **Config hash**: What you're training (deterministic from hyperparameters)
- **Timestamp**: When you trained it (YYYYMMDD_HHMMSS)
- **Random suffix**: Prevents collisions

Enables:
- Finding all training runs of the same config
- Chronological ordering
- Audit trail
- No coordination needed

### 3-Class Classification (ADR-010)
- **Short (0)**: Price expected to drop more than threshold
- **Neutral (1)**: Price expected to move within ±threshold
- **Long (2)**: Price expected to rise more than threshold

Benefits:
- Handles sideways markets (neutral class)
- Clear interpretation (direction + confidence)
- Confidence scores can drive position sizing

### Walk-Forward Validation
- Time series cross-validation preventing look-ahead bias
- Multiple validation windows across different market regimes
- Realistic backtesting (transaction costs, slippage, leverage)
- Aggregate metrics and consistency scoring

---

## Development Principles

1. **Data Leakage Prevention** - Strict timestamp validation, lookback-aware features
2. **Validation Rigor** - Walk-forward validation, multiple windows, realistic costs
3. **Production Readiness** - Well-documented, tested code with error handling
4. **Modularity First** - Config-driven, reusable components
5. **No Backward Compatibility (Alpha)** - ADR-013: Breaking changes allowed, iterate fast

---

## File Organization

### Documentation (`docs/`)
- This file - Index to all documentation
- Strategy guides, status reports, checklists
- Architectural decision records

### Configuration (`config/` and `configs/`)
- `config/` - Python Pydantic models (type-safe config classes)
- `configs/` - YAML files (actual strategy configurations)

### Core Modules
- `data/` - Loading, validation, metadata
- `features/` - Feature engineering, indicators
- `validation/` - Walk-forward, backtesting, metrics
- `models/` - Model registry, metadata, trained artifacts
- `portfolio/` - Coordinator, positions, signals
- `strategies/` - Training and validation scripts
- `utils/` - Logging, helpers

### Tests
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - End-to-end workflow tests

---

## Contact & Support

For questions about:
- **Strategy development** → See [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md)
- **Current status** → See [STATUS.md](STATUS.md)
- **Design decisions** → See [DECISIONS.md](DECISIONS.md)
- **Implementation details** → Check module-specific READMEs (if available)

---

**NailSage** - Building robust ML trading strategies with rigorous validation
