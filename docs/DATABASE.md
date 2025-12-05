# Database Structure

**NailSage Paper Trading State Persistence**

**Last Updated**: 2025-12-03

---

## Overview

NailSage uses a relational database to track all paper trading activity with ACID guarantees for state consistency, full audit trail, crash recovery, and performance analysis.

### **Supported Database Engines**

| Engine | Use Case | Connection String |
|--------|----------|-------------------|
| **SQLite** | Local development, testing | `sqlite:///execution/state/paper_trading.db` |
| **PostgreSQL** | Production, Docker deployment | `postgresql://nailsage:password@postgres:5432/nailsage` |

### **Key Features**

- ACID-compliant storage for all trading state
- Crash recovery (persist state across restarts)
- Full audit trail of trading activity
- Performance metrics and analytics
- Thread-safe operations
- Connection pooling (PostgreSQL)
- Foreign key constraints for data integrity

---

## Configuration

### **Environment Variables**

Set `DATABASE_URL` to specify the database connection:

```bash
# SQLite (local development)
DATABASE_URL=sqlite:///execution/state/paper_trading.db

# PostgreSQL (production)
DATABASE_URL=postgresql://nailsage:password@postgres:5432/nailsage
```

### **Docker Deployment**

PostgreSQL is automatically configured in `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: nailsage
      POSTGRES_USER: nailsage
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./execution/persistence/schema.postgres.sql:/docker-entrypoint-initdb.d/01-schema.sql

  nailsage-binance:
    environment:
      DATABASE_URL: postgresql://nailsage:${DB_PASSWORD}@postgres:5432/nailsage
```

---

## Schema Structure

### **Entity Relationship Diagram**

```
┌─────────────┐
│ strategies  │
└──────┬──────┘
       │
       ├─────┬────────┬──────────┐
       │     │        │          │
       ▼     ▼        ▼          ▼
  ┌────────┐ ┌──────┐ ┌────────┐ ┌──────────────────┐
  │signals │ │positions│trades │ │performance_metrics│
  └────────┘ └───┬────┘ └───┬──┘ └──────────────────┘
                 │          │
                 └──────────┘
                  (FK: position_id)

┌──────────────────┐   ┌────────────┐
│state_snapshots   │   │ event_log  │
└──────────────────┘   └────────────┘
(Global system state)  (Audit trail)
```

---

## Tables

### **1. strategies**

Tracks active and historical trading strategies with isolated bankroll management.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `strategy_name` | VARCHAR/TEXT | Strategy identifier (e.g., "dev_scalper_1m") |
| `version` | VARCHAR/TEXT | Strategy version (e.g., "v1") |
| `starlisting_id` | INTEGER | Market/symbol ID from Kirby API |
| `interval` | VARCHAR/TEXT | Candle interval (e.g., "1m", "15m") |
| `model_id` | VARCHAR/TEXT | NailSage model ID from model registry |
| `config_path` | TEXT | Path to strategy YAML config |
| `is_active` | BOOLEAN | Whether strategy is currently running |
| `initial_bankroll` | REAL | Starting capital for this strategy (USDT), default: 10000.0 |
| `current_bankroll` | REAL | Current capital after P&L (USDT), default: 10000.0 |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |
| `updated_at` | BIGINT | Unix timestamp (milliseconds) |

**Bankroll Behavior:**
- Each strategy gets an isolated $10,000 bankroll by default
- Position sizes are calculated as a percentage (default 10%) of `current_bankroll`
- After each trade closes, `current_bankroll` is updated with the realized P&L
- If `current_bankroll <= 0`, the strategy cannot open new trades (but can close existing positions)
- Use the API `PATCH /api/v1/strategies/{id}/bankroll` endpoint to replenish a depleted bankroll

**Indexes:**
- `idx_strategies_active` on `is_active`
- `idx_strategies_starlisting` on `starlisting_id`

**Constraints:**
- `UNIQUE(strategy_name, version, starlisting_id)`

---

### **2. positions**

Tracks open and closed trading positions.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `strategy_id` | INTEGER | Foreign key → `strategies.id` |
| `starlisting_id` | INTEGER | Market/symbol ID |
| `side` | VARCHAR/TEXT | Position direction: `'long'` or `'short'` |
| `size` | REAL | Position size in USDT |
| `entry_price` | REAL | Entry price |
| `entry_timestamp` | BIGINT | Entry time (Unix ms) |
| `exit_price` | REAL | Exit price (NULL if open) |
| `exit_timestamp` | BIGINT | Exit time (NULL if open) |
| `realized_pnl` | REAL | Realized P&L after exit (USDT) |
| `unrealized_pnl` | REAL | Current unrealized P&L (updated every candle) |
| `fees_paid` | REAL | Total fees paid |
| `status` | VARCHAR/TEXT | `'open'`, `'closed'`, or `'liquidated'` |
| `stop_loss_price` | REAL | Stop loss price (NULL if not set) |
| `take_profit_price` | REAL | Take profit price (NULL if not set) |
| `exit_reason` | VARCHAR/TEXT | `'signal'`, `'stop_loss'`, `'take_profit'`, `'manual'` |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |
| `updated_at` | BIGINT | Unix timestamp (milliseconds) |

**Indexes:**
- `idx_positions_strategy` on `strategy_id`
- `idx_positions_status` on `status`
- `idx_positions_starlisting` on `starlisting_id`
- `idx_positions_entry_time` on `entry_timestamp`

**Foreign Keys:**
- `strategy_id` → `strategies(id)`

---

### **3. signals**

Tracks all trading signals generated by strategies (executed and suppressed).

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `strategy_id` | INTEGER | Foreign key → `strategies.id` |
| `starlisting_id` | INTEGER | Market/symbol ID |
| `signal_type` | VARCHAR/TEXT | `'long'`, `'short'`, `'neutral'`, `'close'` |
| `confidence` | REAL | Model confidence (0-1) |
| `price_at_signal` | REAL | Price when signal was generated |
| `timestamp` | BIGINT | Unix timestamp (milliseconds) |
| `was_executed` | BOOLEAN | Whether signal was acted upon |
| `rejection_reason` | VARCHAR/TEXT | If not executed, why? (e.g., "cooldown", "duplicate") |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |

**Indexes:**
- `idx_signals_strategy` on `strategy_id`
- `idx_signals_timestamp` on `timestamp`
- `idx_signals_executed` on `was_executed`

**Foreign Keys:**
- `strategy_id` → `strategies(id)`

**Usage:**
- Tracks all signals (executed and suppressed) for analysis
- `was_executed=TRUE`: Signal resulted in a trade
- `was_executed=FALSE`: Signal was suppressed (see `rejection_reason`)

---

### **4. trades**

Tracks individual trade executions (entry and exit orders).

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `position_id` | INTEGER | Foreign key → `positions.id` |
| `strategy_id` | INTEGER | Foreign key → `strategies.id` |
| `starlisting_id` | INTEGER | Market/symbol ID |
| `trade_type` | VARCHAR/TEXT | `'open_long'`, `'open_short'`, `'close_long'`, `'close_short'` |
| `size` | REAL | Trade size in USDT |
| `price` | REAL | Execution price |
| `fees` | REAL | Transaction fees (USDT) |
| `slippage` | REAL | Slippage (USDT) |
| `timestamp` | BIGINT | Unix timestamp (milliseconds) |
| `signal_id` | INTEGER | Foreign key → `signals.id` (NULL if manual) |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |

**Indexes:**
- `idx_trades_position` on `position_id`
- `idx_trades_strategy` on `strategy_id`
- `idx_trades_timestamp` on `timestamp`
- `idx_trades_signal` on `signal_id`

**Foreign Keys:**
- `position_id` → `positions(id)`
- `strategy_id` → `strategies(id)`
- `signal_id` → `signals(id)`

**Usage:**
- Each position typically has 2 trades: one entry (open_long/open_short) and one exit (close_long/close_short)
- Tracks fees and slippage per trade
- Links to originating signal via `signal_id`

---

### **5. state_snapshots**

Periodic snapshots of global system state.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `total_equity` | REAL | Current total equity (USDT) |
| `available_capital` | REAL | Available capital (USDT) |
| `allocated_capital` | REAL | Capital in open positions (USDT) |
| `total_unrealized_pnl` | REAL | Sum of unrealized P&L across all positions |
| `total_realized_pnl` | REAL | Sum of realized P&L (all-time) |
| `num_open_positions` | INTEGER | Number of open positions |
| `num_strategies_active` | INTEGER | Number of active strategies |
| `timestamp` | BIGINT | Unix timestamp (milliseconds) |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |

**Indexes:**
- `idx_state_snapshots_timestamp` on `timestamp`

**Usage:**
- Saved periodically (e.g., every minute) to track equity curve
- Used for drawdown calculation and performance analysis
- Global view across all strategies

---

### **6. performance_metrics**

Aggregated performance metrics by strategy and timeframe.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `strategy_id` | INTEGER | Foreign key → `strategies.id` (NULL = global) |
| `timeframe` | VARCHAR/TEXT | `'daily'`, `'weekly'`, `'monthly'`, `'all_time'` |
| `period_start` | BIGINT | Period start (Unix ms) |
| `period_end` | BIGINT | Period end (Unix ms) |
| **Returns** | | |
| `total_return_pct` | REAL | Total return percentage |
| `total_return_usdt` | REAL | Total return in USDT |
| **Win Rate** | | |
| `num_trades` | INTEGER | Number of trades |
| `num_wins` | INTEGER | Number of winning trades |
| `num_losses` | INTEGER | Number of losing trades |
| `win_rate_pct` | REAL | Percentage of winning trades |
| **Risk Metrics** | | |
| `avg_win_usdt` | REAL | Average win size |
| `avg_loss_usdt` | REAL | Average loss size |
| `largest_win_usdt` | REAL | Largest single win |
| `largest_loss_usdt` | REAL | Largest single loss |
| `profit_factor` | REAL | Gross profit / gross loss |
| **Position Metrics** | | |
| `avg_position_duration_minutes` | REAL | Average position holding time |
| `max_drawdown_pct` | REAL | Maximum drawdown percentage |
| `max_drawdown_usdt` | REAL | Maximum drawdown in USDT |
| `sharpe_ratio` | REAL | Risk-adjusted return |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |
| `updated_at` | BIGINT | Unix timestamp (milliseconds) |

**Indexes:**
- `idx_performance_strategy` on `strategy_id`
- `idx_performance_timeframe` on `timeframe`
- `idx_performance_period` on `(period_start, period_end)`

**Foreign Keys:**
- `strategy_id` → `strategies(id)` (NULL for global metrics)

**Usage:**
- Pre-aggregated metrics for faster dashboard queries
- Updated periodically or on-demand
- `strategy_id=NULL`: Global metrics across all strategies
- `strategy_id=N`: Metrics for specific strategy

---

### **7. event_log**

Audit trail for system events.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL/INTEGER | Primary key |
| `event_type` | VARCHAR/TEXT | Event type (e.g., `'websocket_connect'`, `'strategy_start'`) |
| `event_data` | TEXT | JSON-encoded event data |
| `severity` | VARCHAR/TEXT | `'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`, `'CRITICAL'` |
| `timestamp` | BIGINT | Unix timestamp (milliseconds) |
| `created_at` | BIGINT | Unix timestamp (milliseconds) |

**Indexes:**
- `idx_event_log_type` on `event_type`
- `idx_event_log_severity` on `severity`
- `idx_event_log_timestamp` on `timestamp`

**Usage:**
- Audit trail for debugging and compliance
- System events (WebSocket connections, strategy lifecycle)
- Searchable by type, severity, and time range

---

## Views

Convenience views for common queries.

### **v_open_positions**

Open positions with strategy details and human-readable timestamps.

```sql
SELECT * FROM v_open_positions;
```

**Columns:**
- All columns from `positions` table
- `strategy_name`, `version`, `interval` (from `strategies`)
- `entry_datetime` (converted from Unix ms)
- `updated_datetime` (converted from Unix ms)

---

### **v_closed_positions**

Closed positions with P&L and duration.

```sql
SELECT * FROM v_closed_positions;
```

**Columns:**
- All columns from `positions` table
- `strategy_name`, `version`, `interval` (from `strategies`)
- `entry_datetime` (converted from Unix ms)
- `exit_datetime` (converted from Unix ms)
- `duration_minutes` (position holding time)

---

### **v_strategy_performance**

Real-time performance summary per strategy.

```sql
SELECT * FROM v_strategy_performance;
```

**Columns:**
- `strategy_id`, `strategy_name`, `version`, `interval`
- `num_open_positions`, `num_closed_positions`
- `total_unrealized_pnl`, `total_realized_pnl`
- `total_fees_paid`, `avg_pnl_per_trade`
- `num_wins`, `num_losses`

**Usage:**
- Live dashboard showing current strategy performance
- Includes both open positions (unrealized P&L) and closed (realized P&L)

---

### **v_recent_signals**

Last 100 signals with execution status.

```sql
SELECT * FROM v_recent_signals;
```

**Columns:**
- All columns from `signals` table
- `strategy_name`, `version`, `interval` (from `strategies`)
- `signal_datetime` (converted from Unix ms)

**Usage:**
- See recent signal activity
- Analyze signal suppression reasons
- Debug model predictions

---

## Usage Examples

### **Python (via StateManager)**

```python
from execution.persistence.state_manager import StateManager, Strategy, Position, Trade

# Initialize (auto-detects SQLite or PostgreSQL from DATABASE_URL)
state = StateManager()

# Save a strategy (with default $10,000 bankroll)
strategy = Strategy(
    id=None,
    strategy_name="dev_scalper_1m",
    version="v1",
    starlisting_id=12345,
    interval="1m",
    model_id="nailsage-momentum-20251127-123456",
    config_path="strategy-configs/dev_scalper_1m_v1.yaml",
    is_active=True,
    initial_bankroll=10000.0,  # Starting capital (USDT)
    current_bankroll=10000.0,  # Current capital after P&L (USDT)
)
strategy_id = state.save_strategy(strategy)

# Update bankroll after a trade closes
state.update_strategy_bankroll(strategy_id, new_bankroll=9850.0)  # Lost $150

# Get open positions
open_positions = state.get_open_positions(strategy_id=strategy_id)

# Save a position
position = Position(
    id=None,
    strategy_id=strategy_id,
    starlisting_id=12345,
    side="long",
    size=10000.0,
    entry_price=95000.0,
    entry_timestamp=int(datetime.now().timestamp() * 1000),
    status="open",
)
position_id = state.save_position(position)

# Update unrealized P&L
position.unrealized_pnl = 150.0
state.save_position(position)

# Close the position
position.exit_price = 95150.0
position.exit_timestamp = int(datetime.now().timestamp() * 1000)
position.realized_pnl = 150.0 - fees_paid
position.status = "closed"
position.exit_reason = "signal"
state.save_position(position)

# Log an event
state.log_event(
    event_type="websocket_connect",
    event_data={"exchange": "binance", "reconnect": False},
    severity="INFO",
)
```

### **Direct SQL Queries**

#### **Get strategy performance**
```sql
SELECT * FROM v_strategy_performance
WHERE strategy_name = 'dev_scalper_1m_v1';
```

#### **Find all suppressed signals**
```sql
SELECT signal_type, rejection_reason, COUNT(*) as count
FROM signals
WHERE was_executed = FALSE
GROUP BY signal_type, rejection_reason
ORDER BY count DESC;
```

#### **Calculate win rate**
```sql
SELECT
    strategy_name,
    COUNT(*) FILTER (WHERE realized_pnl > 0) AS wins,
    COUNT(*) FILTER (WHERE realized_pnl <= 0) AS losses,
    ROUND(100.0 * COUNT(*) FILTER (WHERE realized_pnl > 0) / COUNT(*), 2) AS win_rate_pct
FROM v_closed_positions
GROUP BY strategy_name;
```

#### **Get equity curve (from snapshots)**
```sql
SELECT
    timestamp,
    total_equity,
    total_unrealized_pnl,
    total_realized_pnl,
    num_open_positions
FROM state_snapshots
ORDER BY timestamp ASC;
```

---

## Database Initialization

### **Automatic Initialization**

The `StateManager` class automatically initializes the schema if tables don't exist:

```python
state = StateManager()  # Creates schema if needed
```

### **Manual Initialization**

#### **SQLite**
```bash
sqlite3 execution/state/paper_trading.db < execution/persistence/schema.sql
```

#### **PostgreSQL**
```bash
psql -U nailsage -d nailsage < execution/persistence/schema.postgres.sql
```

#### **Docker (automatic)**
The PostgreSQL schema is automatically initialized via `docker-entrypoint-initdb.d`:

```yaml
postgres:
  volumes:
    - ./execution/persistence/schema.postgres.sql:/docker-entrypoint-initdb.d/01-schema.sql
```

---

## Connection Pooling

### **SQLite**
- Uses `NullPool` (no pooling)
- `check_same_thread=False` for multi-threaded access
- WAL mode enabled for better concurrency

### **PostgreSQL**
- Uses `QueuePool` with connection pooling
- `pool_size=5` (5 persistent connections)
- `max_overflow=10` (up to 15 connections under load)
- `pool_pre_ping=True` (verify connections before use)

---

## Timestamps

**All timestamps are stored as Unix milliseconds** for consistency with the Kirby API.

**Conversion examples:**

```python
# Python: datetime → Unix ms
unix_ms = int(datetime.now().timestamp() * 1000)

# Python: Unix ms → datetime
dt = datetime.fromtimestamp(unix_ms / 1000.0)

# SQL (PostgreSQL): Unix ms → timestamp
SELECT to_timestamp(timestamp / 1000.0) FROM positions;

# SQL (SQLite): Unix ms → datetime
SELECT datetime(timestamp / 1000, 'unixepoch') FROM positions;
```

---

## Performance Considerations

### **Indexes**
All critical foreign keys and query columns are indexed for fast lookups:
- Strategy lookups by `is_active`, `starlisting_id`
- Position lookups by `strategy_id`, `status`, `entry_timestamp`
- Signal lookups by `strategy_id`, `timestamp`, `was_executed`
- Trade lookups by `position_id`, `strategy_id`, `timestamp`

### **Views**
Pre-defined views reduce query complexity and improve maintainability:
- Use `v_open_positions` instead of joining `positions` + `strategies` manually
- Use `v_strategy_performance` for aggregated metrics

### **Transaction Safety**
Use the `StateManager.transaction()` context manager for atomic operations:

```python
with state.transaction():
    state.save_position(position)
    state.save_trade(trade)
# Automatically commits on success, rolls back on exception
```

---

## File Locations

| File | Purpose |
|------|---------|
| [execution/persistence/state_manager.py](../execution/persistence/state_manager.py) | Python ORM and database manager |
| [execution/persistence/schema.sql](../execution/persistence/schema.sql) | SQLite schema definition |
| [execution/persistence/schema.postgres.sql](../execution/persistence/schema.postgres.sql) | PostgreSQL schema definition |
| [docker-compose.yml](../docker-compose.yml) | PostgreSQL configuration for Docker |

---

## Common Tasks

### **Reset Database (Development)**

**SQLite:**
```bash
rm execution/state/paper_trading.db
python -c "from execution.persistence.state_manager import StateManager; StateManager()"
```

**PostgreSQL (Docker):**
```bash
docker compose down -v  # Delete volumes
docker compose up -d postgres  # Recreate with fresh schema
```

### **Backup Database**

**SQLite:**
```bash
cp execution/state/paper_trading.db execution/state/paper_trading.db.backup
```

**PostgreSQL:**
```bash
docker exec -t nailsage-postgres pg_dump -U nailsage nailsage > backup.sql
```

### **Restore Database**

**PostgreSQL:**
```bash
docker exec -i nailsage-postgres psql -U nailsage -d nailsage < backup.sql
```

---

## Related Documentation

- [DOCKER.md](DOCKER.md) - Docker deployment and multi-strategy execution
- [ACTIVE_FILES.md](ACTIVE_FILES.md) - Active vs deprecated files
- [MODEL_TRAINING.md](MODEL_TRAINING.md) - Model training and validation
