# NailSage Documentation

**ML Trading Research Platform for Cryptocurrency Markets**

**Last Updated**: 2025-11-27
**Current Phase**: Phase 10 Complete ✅ - Live Trading Operational with Binary Classification

---

## Documentation Overview

### Getting Started
- [MODEL_TRAINING.md](MODEL_TRAINING.md) - Complete guide to training and validating ML models
- [DOCKER.md](DOCKER.md) - Docker deployment and multi-strategy execution
- [ACTIVE_FILES.md](ACTIVE_FILES.md) - Current codebase structure and file inventory

### Architecture & API
- [API.md](API.md) - REST API endpoints and usage
- [WEBSOCKET.md](WEBSOCKET.md) - Real-time WebSocket connections
- [DATABASE.md](DATABASE.md) - Database schema and operations
- [FEATURE_SCHEMA_USAGE.md](FEATURE_SCHEMA_USAGE.md) - Feature engineering and validation
- [DECISIONS.md](DECISIONS.md) - Key architectural decisions and rationale

---

## Document Guide

### 🚀 [MODEL_TRAINING.md](MODEL_TRAINING.md)
**What**: Complete guide to training, validating, and deploying ML trading models
**When to read**: New to ML training workflow or need detailed procedures
**Key sections**:
- Data preparation and feature engineering
- Model training with walk-forward validation
- Backtesting and performance evaluation
- Strategy deployment and monitoring

### 🐳 [DOCKER.md](DOCKER.md)
**What**: Production deployment guide using Docker containers
**When to read**: Setting up paper trading or production deployment
**Key sections**:
- Single and multi-strategy execution
- Environment configuration
- Database setup and monitoring
- Troubleshooting deployment issues

### 📋 [ACTIVE_FILES.md](ACTIVE_FILES.md)
**What**: Current codebase structure and file inventory
**When to read**: Understanding project organization or finding specific files
**Key sections**:
- Directory structure overview
- File status and descriptions
- Development guidelines

### 🔌 [API.md](API.md)
**What**: REST API documentation for portfolio management and strategy monitoring
**When to read**: Building integrations or monitoring trading performance
**Key sections**:
- Authentication and endpoints
- Portfolio data retrieval
- Real-time updates and WebSocket connections

### 🏛️ [DECISIONS.md](DECISIONS.md)
**What**: Key architectural decisions and design rationale
**When to read**: Understanding why certain design choices were made
**Key decisions**:
- Independent Models vs Portfolio Optimization
- Classical ML selection
- Configuration management approach
- Model identification system

---

## Platform Overview

### 🎯 Core Capabilities
1. **ML Model Training** - Automated training pipeline with walk-forward validation
2. **Paper Trading Execution** - Real-time strategy execution with realistic simulation
3. **Multi-Strategy Support** - Run multiple strategies simultaneously per exchange
4. **Real-Time Monitoring** - Live P&L tracking and signal transparency
5. **Production Deployment** - Docker-based deployment with PostgreSQL persistence
6. **Config-Driven Architecture** - Define strategies in YAML, no code changes needed
7. **Model Registry** - Centralized model tracking with reproducibility features

### 📊 Technical Specifications
- **Models**: Binary classification LightGBM models
- **Features**: 18 technical indicators (OHLCV excluded for leak prevention)
- **Validation**: Walk-forward time series cross-validation
- **Execution**: Real-time paper trading with fees/slippage simulation
- **Data**: Time series cryptocurrency market data
- **Deployment**: Docker Compose with PostgreSQL and Redis

---

## Getting Started

### For New Developers
1. Read [MODEL_TRAINING.md](MODEL_TRAINING.md) - Learn the ML training workflow
2. Read [DOCKER.md](DOCKER.md) - Understand deployment and execution
3. Review [ACTIVE_FILES.md](ACTIVE_FILES.md) - Understand codebase structure
4. Check [DECISIONS.md](DECISIONS.md) - Learn key architectural choices

### For Strategy Development
1. Read [MODEL_TRAINING.md](MODEL_TRAINING.md) - Complete training guide
2. Examine existing strategy configs in `strategies/`
3. Review training scripts in `training/cli/`
4. Check validation results in `results/`

### For System Integration
1. Read [API.md](API.md) - REST API documentation
2. Read [WEBSOCKET.md](WEBSOCKET.md) - Real-time data connections
3. Review [DATABASE.md](DATABASE.md) - Data persistence layer

---

## Key Concepts

### Independent Models Philosophy
- Each strategy operates autonomously with its own risk management
- Strategies can have opposing signals simultaneously (this is valid)
- Portfolio Coordinator enforces global risk limits only
- Focus on individual model quality rather than portfolio optimization

### Hybrid Model IDs
Format: `{config_hash}_{timestamp}_{random_suffix}`
- **Config hash**: What you're training (deterministic from hyperparameters)
- **Timestamp**: When you trained it (YYYYMMDD_HHMMSS)
- **Random suffix**: Prevents collisions

Benefits:
- Track all training runs of the same configuration
- Chronological ordering and audit trails
- Reproducibility and experiment management

### Binary Classification Models
- **Short (0)**: Price expected to drop more than threshold
- **Long (1)**: Price expected to rise more than threshold

Design choices:
- No neutral class (forces directional decisions)
- Confidence-based filtering prevents low-quality signals
- Signal cooldown prevents trading frequency issues

### Walk-Forward Validation
- Time series cross-validation preventing data leakage
- Multiple validation windows across different market regimes
- Realistic backtesting with transaction costs, slippage, and leverage
- Comprehensive performance metrics and consistency scoring

---

## Development Principles

1. **Data Leakage Prevention** - Strict temporal ordering, lookback-aware features
2. **Rigorous Validation** - Walk-forward cross-validation with realistic market simulation
3. **Production Ready** - Well-tested, documented code with comprehensive error handling
4. **Config-Driven Architecture** - YAML-based strategy definitions, minimal code changes
5. **Modular Design** - Independent, reusable components with clear interfaces

---

## Architecture Overview

### Core Modules
- `config/` - Pydantic configuration models (type-safe configs)
- `data/` - Data loading, validation, and metadata management
- `features/` - Technical indicator computation and feature engineering
- `models/` - Model registry, metadata, and trained artifacts
- `training/` - ML training pipeline and validation
- `execution/` - Paper trading execution and portfolio management
- `utils/` - Shared utilities and logging

### Configuration & Strategies
- `strategies/` - YAML strategy configurations (gitignored)
- `results/` - Training and backtesting results (gitignored)

### Testing
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - End-to-end workflow and integration tests

### Documentation
- `docs/` - Complete documentation set
- `README.md` - Main project documentation

---

## Support & Resources

For questions about:
- **Training ML models** → See [MODEL_TRAINING.md](MODEL_TRAINING.md)
- **Deploying strategies** → See [DOCKER.md](DOCKER.md)
- **API integration** → See [API.md](API.md)
- **Design decisions** → See [DECISIONS.md](DECISIONS.md)
- **Codebase structure** → See [ACTIVE_FILES.md](ACTIVE_FILES.md)

---

**NailSage** - Building robust ML trading strategies with rigorous validation
