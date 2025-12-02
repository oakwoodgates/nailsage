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

## ADR-011: Dataset Metadata Tracking for Reproducibility
**Date**: 2025-11-08
**Status**: Accepted

**Context**: Need to track full provenance of training data to ensure model reproducibility. When a model is trained on specific data, we must be able to identify exactly which dataset, time range, exchange, and quality metrics were used.

**Decision**: Implement comprehensive dataset metadata tracking with `DatasetMetadata` dataclass and CLI generation tool.

**Rationale**:
- Full reproducibility requires knowing exact data used for training
- Quality metrics at dataset level prevent using low-quality data unknowingly
- Filename-based auto-extraction reduces manual metadata entry
- JSON format allows easy programmatic access and human readability
- Linking models to dataset metadata creates complete audit trail
- Read-only validation ensures source data integrity

**Consequences**:
- Every dataset in `data/raw/` should have accompanying `.metadata.json` file
- Model registry must reference dataset metadata for full provenance
- Filename conventions should follow pattern: `{exchange}_{asset}_{quote}_{market_type}_{interval}.ext`
- CLI tool makes metadata generation easy and consistent
- Metadata files tracked in git for versioning

**Implementation**:
- `data/metadata.py`: DatasetMetadata dataclass
- `scripts/generate_data_metadata.py`: CLI tool for metadata generation
- `data/loader.py`: Enhanced with column name normalization

---

## ADR-012: Centralized Model Registry with Hybrid ID System
**Date**: 2025-11-08
**Status**: Accepted
**Updated**: 2025-11-08 (Implemented hybrid ID system)

**Context**: Need to manage multiple trained models across different strategies, versions, and datasets. Must support querying models by various criteria (strategy, performance, dataset characteristics) and maintain complete provenance for reproducibility. Additionally, need to track multiple training runs of the same configuration over time (audit trail).

**Decision**: Implement centralized `ModelRegistry` with **hybrid model IDs** that encode both configuration intent and training implementation.

**Hybrid ID Format**: `{config_hash}_{timestamp}_{random_suffix}`
- Example: `2e30bea4e8f93845_20251108_153045_a3f9c2`
- Config hash (16 chars): WHAT you're training (deterministic from hyperparameters + data)
- Timestamp (15 chars): WHEN you trained it (YYYYMMDD_HHMMSS)
- Random suffix (6 chars): Collision prevention for same-second training runs

**Rationale**:
- **Separates Intent from Implementation**: Config hash represents configuration intent, timestamp represents training execution
- **Audit Trail**: Can track multiple training runs of same configuration over time
- **Deduplication**: Config hash enables finding "similar" models without preventing retraining
- **Chronological Ordering**: Timestamps embedded in ID for natural sorting
- **Human-Readable**: Can see when model was trained from ID alone
- **No Coordination Needed**: Timestamp + random suffix prevents race conditions
- **File System Friendly**: Models naturally group by config, then sort by time
- **Modular**: Independent of strategy directory structure

**Consequences**:
- All trained models stored centrally in `models/trained/` and `models/metadata/`
- Model IDs are unique per training run (no overwriting)
- Same configuration can be trained multiple times (tracked separately)
- Config hash enables queries like "find all runs of this config"
- Timestamp enables queries like "latest trained model for this config"
- File listings show chronological order automatically
- Slightly longer IDs (38 chars) vs pure hash (16 chars) or UUID (36 chars)
- Helper methods on ModelMetadata parse ID components:
  - `get_config_hash()` - extract configuration hash
  - `get_training_timestamp()` - extract when it was trained
  - `is_hybrid_id()` - check ID format
- ModelRegistry provides config-aware queries:
  - `find_models_by_config(config_hash)` - all runs of a configuration
  - `get_latest_by_config(config_hash)` - most recent run
  - `list_config_families()` - group models by configuration

**Implementation**:
- `models/metadata.py`: ModelMetadata dataclass with ID parsing helpers
- `models/registry.py`: ModelRegistry with config-aware queries
- `models/utils.py`: `generate_model_id()` creates hybrid IDs, `generate_config_hash()` for intent fingerprinting
- `tests/unit/test_hybrid_ids.py`: Comprehensive tests for hybrid ID functionality

---

## Template for New Decisions

```markdown
## ADR-013: No Backward Compatibility in Alpha Phase
**Date**: 2025-11-15
**Status**: Accepted

**Context**: During alpha development, we need to decide whether to maintain backward compatibility when making architectural or configuration changes.

**Decision**: **No backward compatibility is required during alpha phase** (pre-production). Breaking changes are acceptable without migration paths. However, **major rewrites must be explicitly flagged and approved** by the user.

**Rationale**:
- Faster iteration: Can clean up technical debt immediately (e.g., removing "old format" support)
- Simpler codebase: No need to maintain dual code paths or compatibility shims
- No production users: Nothing deployed yet, so no impact on existing systems
- Faster learning: Can experiment with different approaches without legacy burden
- Pre-deployment flexibility: Architecture can evolve based on what we learn

**Consequences**:
- **Positive**:
  - Cleaner, more maintainable code
  - Faster development velocity
  - Can make bold architectural decisions
  - Easier to refactor and improve
- **Negative**:
  - Need to update tests when schemas change
  - Documentation must be kept current
  - Saved configs/models may become incompatible
- **Mitigation**:
  - Always ask user before major rewrites (> ~500 LOC changes)
  - Update documentation immediately when breaking changes occur
  - Keep .claude/ files current with latest architecture
  - Run all tests after breaking changes

**When This Changes**:
Once we deploy to production or have external users:
1. Introduce semantic versioning
2. Maintain backward compatibility within major versions
3. Provide migration guides for major version upgrades
4. Add deprecation warnings before removing features

---

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

---

## ADR-014: Binary Classification for Aggressive Trading
**Date**: 2025-11-27
**Status**: Accepted (Supersedes ADR-010)

**Context**: ADR-010 implemented 3-class classification (SHORT/NEUTRAL/LONG). After extensive backtesting, the NEUTRAL class created ambiguity and poor signal quality, leading to excessive trading and reduced profitability.

**Decision**: Switch to **binary classification** (SHORT/LONG only, no neutral class). Models must commit to a directional prediction.

**Rationale**:
- **Clearer signals**: Model forced to choose direction, no ambiguous neutral zone
- **Better model performance**: Binary decision boundary easier to learn than 3-class
- **Reduced false signals**: Elimination of weak neutral predictions
- **Higher win rates**: 58% vs 45% with 3-class in backtests
- **Better returns**: 65.54% vs -12.3% with 3-class on SOL swing strategy

**Configuration**:
```yaml
target:
  type: classification
  classes: 2  # Binary: SHORT=0, LONG=1 (no neutral)
  lookahead_bars: 3
  threshold_pct: 0.2  # Minimum price movement to classify as signal
  confidence_threshold: 0.5  # Minimum confidence to generate trade signal
```

**Consequences**:
- **Positive**:
  - Simplified model architecture (2 outputs vs 3)
  - Clearer signal interpretation
  - Higher profitability in backtests
  - Better model confidence calibration
- **Negative**:
  - No explicit "do nothing" class
  - Model must always pick a direction
- **Mitigation**:
  - Confidence thresholding prevents weak signals
  - Signal cooldown prevents overtrading
  - Threshold percentage ensures minimum movement required

**Related Features**:
- Confidence-based filtering (minimum 50% confidence to trade)
- Signal cooldown (4 bars minimum between signals)
- Real-time P&L tracking
- Signal suppression logging

---

## ADR-015: Feature Caching Disabled for Live Trading
**Date**: 2025-11-27
**Status**: Accepted

**Context**: Feature caching was originally enabled globally to speed up training/backtesting by caching computed indicators. However, in live trading with streaming data, caching provides no benefit and was causing pickle file corruption issues in Docker environments.

**Decision**: **Disable feature caching for live trading**, keep enabled for training/backtesting.

**Implementation**:
```python
# In scripts/run_multi_strategy.py
feature_config = FeatureConfig.model_validate(model_metadata.feature_config)
feature_config.enable_cache = False  # Disable for live trading
feature_engine = FeatureEngine(feature_config)
```

**Rationale**:
- Live trading processes unique real-time data each second (never reused)
- Caching provides zero performance benefit for streaming data
- Docker volume sync issues were corrupting pickle cache files
- Eliminates "pickle data truncated" errors
- Simpler, more reliable live trading execution

**Consequences**:
- **Positive**:
  - Eliminates cache corruption errors in Docker
  - Cleaner logs (no cache warnings)
  - More reliable live trading
  - Simpler mental model (cache only for batch processing)
- **Negative**:
  - None (caching was useless for streaming data anyway)

**Related ADRs**:
- ADR-006: File-Based Caching for Features (still applies to training)
