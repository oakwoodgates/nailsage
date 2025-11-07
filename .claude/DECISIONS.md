# Architectural Decision Records (ADRs)

## ADR-001: Independent Models vs Portfolio Optimization
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Need to decide whether strategies work together as a coordinated portfolio or operate independently.

**Decision**: Phase 1 implements independent models. Each strategy operates autonomously and can produce opposing signals.

**Rationale**:
- Primary goal is training quality individual models
- Simplifies Phase 1 complexity
- Portfolio optimization is complex and not well understood yet
- Can add sophisticated coordination in Phase 2 after models prove themselves

**Consequences**:
- PortfolioCoordinator in Phase 1 is a simple pass-through placeholder
- Each model has its own risk management and position sizing
- May have opposing positions simultaneously (e.g., spot long, perps short)
- Easier to attribute performance to specific strategies

---

## ADR-002: Classical ML for Phase 1
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Choose between classical ML (XGBoost, LightGBM, RF) vs deep learning (LSTM, Transformers).

**Decision**: Phase 1 uses only classical ML models.

**Rationale**:
- Faster training and iteration cycles
- Better interpretability (feature importance)
- Lower computational requirements
- Proven effectiveness on tabular financial data
- Simpler debugging and validation
- User has limited experience with this type of platform

**Consequences**:
- Deep learning deferred to Phase 2
- Focus on feature engineering quality
- May miss sequential patterns (addressed in Phase 2)

---

## ADR-003: Pydantic for Configuration Management
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Need type-safe, validated configuration system for strategies, features, and backtesting.

**Decision**: Use Pydantic models for all configurations, backed by YAML files.

**Rationale**:
- Type safety at runtime
- Automatic validation
- Great IDE support and autocomplete
- Easy to serialize/deserialize
- Better than raw YAML/JSON parsing

**Consequences**:
- Small learning curve for Pydantic
- Slightly more verbose than raw dicts
- Clear contracts between components

---

## ADR-004: Parquet for Historical Data
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Choose data format for historical OHLCV storage.

**Decision**: Use Parquet as primary format for historical data.

**Rationale**:
- Columnar storage efficient for time-series
- Fast read performance
- Compression reduces storage costs
- Supports metadata
- Industry standard for data science

**Consequences**:
- Data ingestion platform needs Parquet export capability (confirmed available)
- WebSocket integration deferred to paper trading phase
- Need conversion scripts if data arrives in other formats

---

## ADR-005: SQLite for State Persistence
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Need to persist positions, trades, signals for recovery and analysis.

**Decision**: Use SQLite for Phase 1 state management.

**Rationale**:
- Zero-configuration, embedded database
- ACID compliance for data integrity
- Sufficient for single-machine deployment
- Easy backups (single file)
- Can migrate to Postgres in Phase 2 if needed
- No additional infrastructure cost

**Consequences**:
- Limited to single-machine deployment (acceptable for Phase 1)
- Need migration strategy if scaling to distributed system
- Good enough for MVP, may need replacement for production scale

---

## ADR-006: File-Based Caching for Features (Phase 1)
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Feature computation can be expensive; need caching strategy.

**Decision**: Simple file-based caching (pickle/joblib) for Phase 1, Redis in future phases.

**Rationale**:
- Avoid Redis infrastructure complexity in MVP
- File system caching sufficient for development and backtesting
- Easy to implement and debug
- Zero additional dependencies

**Consequences**:
- Not suitable for real-time production (acceptable for Phase 1)
- Cache invalidation must be manual/explicit
- Redis upgrade path clear for Phase 2

---

## ADR-007: No Paid Third-Party Services in Phase 1
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Need monitoring, experiment tracking, logging infrastructure.

**Decision**: Build lightweight custom solutions; avoid paid services (MLflow, W&B, Prometheus, Grafana) in Phase 1.

**Rationale**:
- Keep costs minimal for MVP
- User constraint: avoid paid services except Digital Ocean
- Custom solutions give more control and learning
- Can integrate paid services in Phase 2 when value is proven

**Consequences**:
- Build custom experiment logging (JSON files)
- Use Python logging with structured output
- Simple custom metrics collection
- More initial development work, but modular for future upgrades

---

## ADR-008: Docker Compose for Deployment
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Need deployment strategy for Digital Ocean.

**Decision**: Docker + Docker Compose for Phase 1 MVP.

**Rationale**:
- Reproducible environment
- Easy local development matching production
- User needs help with Docker/DevOps
- Sufficient for single-machine deployment
- Clear upgrade path to Kubernetes if needed

**Consequences**:
- Need comprehensive deployment documentation
- Create helper scripts for common operations
- Single-machine limitation (acceptable for Phase 1)

---

## ADR-009: Start with Both Spot and Perps
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Whether to start with single asset or multiple assets.

**Decision**: Implement BTC spot AND BTC perps from day 1.

**Rationale**:
- Proves modularity early
- Forces good architecture (no hardcoded asset assumptions)
- Tests independent model philosophy
- Prevents rework later

**Consequences**:
- Slightly more complex initial setup
- Need to handle different data schemas (spot vs perps)
- May need asset-specific features (funding rates later)
- Higher quality, more flexible codebase

---

## ADR-010: 3-Class Classification for Directional Prediction
**Date**: 2025-11-06
**Status**: Accepted

**Context**: Choose ML task type for trading signals.

**Decision**: Start with 3-class classification (long/short/neutral) with confidence scores.

**Rationale**:
- Clear interpretation: direction + confidence
- Can use confidence for position sizing
- Handles sideways/choppy markets (neutral class)
- Standard evaluation metrics (precision, recall, F1)

**Consequences**:
- Need to define threshold for "neutral" zone
- Class imbalance likely (more neutral than directional moves)
- May experiment with regression in future strategies
- Each strategy can define its own target variable

---

## Template for New Decisions

```markdown
## ADR-XXX: [Title]
**Date**: YYYY-MM-DD
**Status**: [Proposed | Accepted | Deprecated | Superseded]

**Context**: [What is the issue we're trying to solve?]

**Decision**: [What we decided to do]

**Rationale**:
- Reason 1
- Reason 2
- Reason 3

**Consequences**:
- Consequence 1
- Consequence 2
```
