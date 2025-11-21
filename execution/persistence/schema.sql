-- NailSage Paper Trading State Persistence Schema
-- SQLite database for tracking positions, trades, signals, and system state
--
-- This schema provides:
-- - ACID guarantees for state consistency
-- - Full audit trail of all trading activity
-- - Crash recovery capabilities
-- - Performance tracking and analysis
--
-- Usage:
--   sqlite3 execution/state/paper_trading.db < execution/persistence/schema.sql

-- ============================================================================
-- Strategies Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    version TEXT NOT NULL,
    starlisting_id INTEGER NOT NULL,
    interval TEXT NOT NULL,
    model_id TEXT,  -- NailSage model ID (from model registry)
    config_path TEXT,  -- Path to strategy YAML config
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    updated_at INTEGER NOT NULL,  -- Unix timestamp (milliseconds)

    UNIQUE(strategy_name, version, starlisting_id)
);

CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active);
CREATE INDEX IF NOT EXISTS idx_strategies_starlisting ON strategies(starlisting_id);

-- ============================================================================
-- Positions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    starlisting_id INTEGER NOT NULL,
    side TEXT NOT NULL,  -- 'long' or 'short'
    size REAL NOT NULL,  -- Position size (USDT)
    entry_price REAL NOT NULL,
    entry_timestamp INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    exit_price REAL,  -- NULL if position is open
    exit_timestamp INTEGER,  -- Unix timestamp (milliseconds)
    realized_pnl REAL,  -- Profit/loss after exit (USDT)
    unrealized_pnl REAL,  -- Current unrealized P&L (updated on each candle)
    fees_paid REAL NOT NULL DEFAULT 0,  -- Total fees paid
    status TEXT NOT NULL DEFAULT 'open',  -- 'open', 'closed', 'liquidated'
    stop_loss_price REAL,
    take_profit_price REAL,
    exit_reason TEXT,  -- 'signal', 'stop_loss', 'take_profit', 'manual', etc.
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,

    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_starlisting ON positions(starlisting_id);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_timestamp);

-- ============================================================================
-- Trades Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    strategy_id INTEGER NOT NULL,
    starlisting_id INTEGER NOT NULL,
    trade_type TEXT NOT NULL,  -- 'open_long', 'open_short', 'close_long', 'close_short'
    size REAL NOT NULL,  -- Trade size (USDT)
    price REAL NOT NULL,  -- Execution price
    fees REAL NOT NULL,  -- Transaction fees (USDT)
    slippage REAL NOT NULL,  -- Slippage (USDT)
    timestamp INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    signal_id INTEGER,  -- Reference to signal that triggered this trade
    created_at INTEGER NOT NULL,

    FOREIGN KEY (position_id) REFERENCES positions(id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_signal ON trades(signal_id);

-- ============================================================================
-- Signals Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    starlisting_id INTEGER NOT NULL,
    signal_type TEXT NOT NULL,  -- 'long', 'short', 'neutral', 'close'
    confidence REAL,  -- Model confidence (0-1)
    price_at_signal REAL NOT NULL,  -- Price when signal was generated
    timestamp INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    was_executed BOOLEAN NOT NULL DEFAULT 0,  -- Whether signal was acted upon
    rejection_reason TEXT,  -- If not executed, why? (e.g., "max_positions_reached")
    created_at INTEGER NOT NULL,

    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(was_executed);

-- ============================================================================
-- System State Snapshots Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS state_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_equity REAL NOT NULL,  -- Current total equity (USDT)
    available_capital REAL NOT NULL,  -- Available capital (USDT)
    allocated_capital REAL NOT NULL,  -- Capital in open positions (USDT)
    total_unrealized_pnl REAL NOT NULL,  -- Sum of unrealized P&L across all positions
    total_realized_pnl REAL NOT NULL,  -- Sum of realized P&L (all-time)
    num_open_positions INTEGER NOT NULL,
    num_strategies_active INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_timestamp ON state_snapshots(timestamp);

-- ============================================================================
-- Performance Metrics Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER,  -- NULL = global metrics across all strategies
    timeframe TEXT NOT NULL,  -- 'daily', 'weekly', 'monthly', 'all_time'
    period_start INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    period_end INTEGER NOT NULL,  -- Unix timestamp (milliseconds)

    -- Returns
    total_return_pct REAL,  -- Total return percentage
    total_return_usdt REAL,  -- Total return in USDT

    -- Win rate
    num_trades INTEGER NOT NULL DEFAULT 0,
    num_wins INTEGER NOT NULL DEFAULT 0,
    num_losses INTEGER NOT NULL DEFAULT 0,
    win_rate_pct REAL,  -- Percentage of winning trades

    -- Risk metrics
    avg_win_usdt REAL,
    avg_loss_usdt REAL,
    largest_win_usdt REAL,
    largest_loss_usdt REAL,
    profit_factor REAL,  -- Gross profit / gross loss

    -- Position metrics
    avg_position_duration_minutes REAL,
    max_drawdown_pct REAL,
    max_drawdown_usdt REAL,
    sharpe_ratio REAL,  -- Risk-adjusted return

    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,

    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_performance_strategy ON performance_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_performance_timeframe ON performance_metrics(timeframe);
CREATE INDEX IF NOT EXISTS idx_performance_period ON performance_metrics(period_start, period_end);

-- ============================================================================
-- Event Log Table (Audit Trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,  -- 'websocket_connect', 'websocket_disconnect', 'strategy_start', etc.
    event_data TEXT,  -- JSON data for the event
    severity TEXT NOT NULL DEFAULT 'INFO',  -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    timestamp INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_log_type ON event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_event_log_severity ON event_log(severity);
CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View: Open positions with current unrealized P&L
CREATE VIEW IF NOT EXISTS v_open_positions AS
SELECT
    p.*,
    s.strategy_name,
    s.version,
    s.interval,
    datetime(p.entry_timestamp / 1000, 'unixepoch') as entry_datetime,
    datetime(p.updated_at / 1000, 'unixepoch') as updated_datetime
FROM positions p
JOIN strategies s ON p.strategy_id = s.id
WHERE p.status = 'open';

-- View: Closed positions with P&L
CREATE VIEW IF NOT EXISTS v_closed_positions AS
SELECT
    p.*,
    s.strategy_name,
    s.version,
    s.interval,
    datetime(p.entry_timestamp / 1000, 'unixepoch') as entry_datetime,
    datetime(p.exit_timestamp / 1000, 'unixepoch') as exit_datetime,
    (p.exit_timestamp - p.entry_timestamp) / 1000.0 / 60.0 as duration_minutes
FROM positions p
JOIN strategies s ON p.strategy_id = s.id
WHERE p.status = 'closed';

-- View: Strategy performance summary
CREATE VIEW IF NOT EXISTS v_strategy_performance AS
SELECT
    s.id as strategy_id,
    s.strategy_name,
    s.version,
    s.interval,
    COUNT(DISTINCT CASE WHEN p.status = 'open' THEN p.id END) as num_open_positions,
    COUNT(DISTINCT CASE WHEN p.status = 'closed' THEN p.id END) as num_closed_positions,
    SUM(CASE WHEN p.status = 'open' THEN p.unrealized_pnl ELSE 0 END) as total_unrealized_pnl,
    SUM(CASE WHEN p.status = 'closed' THEN p.realized_pnl ELSE 0 END) as total_realized_pnl,
    SUM(CASE WHEN p.status = 'closed' THEN p.fees_paid ELSE 0 END) as total_fees_paid,
    AVG(CASE WHEN p.status = 'closed' THEN p.realized_pnl ELSE NULL END) as avg_pnl_per_trade,
    COUNT(DISTINCT CASE WHEN p.status = 'closed' AND p.realized_pnl > 0 THEN p.id END) as num_wins,
    COUNT(DISTINCT CASE WHEN p.status = 'closed' AND p.realized_pnl <= 0 THEN p.id END) as num_losses
FROM strategies s
LEFT JOIN positions p ON s.id = p.strategy_id
WHERE s.is_active = 1
GROUP BY s.id, s.strategy_name, s.version, s.interval;

-- View: Recent signals with execution status
CREATE VIEW IF NOT EXISTS v_recent_signals AS
SELECT
    sig.*,
    s.strategy_name,
    s.version,
    s.interval,
    datetime(sig.timestamp / 1000, 'unixepoch') as signal_datetime
FROM signals sig
JOIN strategies s ON sig.strategy_id = s.id
ORDER BY sig.timestamp DESC
LIMIT 100;

-- ============================================================================
-- Initial Data / Defaults
-- ============================================================================

-- Log schema initialization
INSERT INTO event_log (event_type, event_data, severity, timestamp, created_at)
VALUES (
    'schema_initialized',
    '{"version": "1.0.0", "description": "Paper trading database schema initialized"}',
    'INFO',
    strftime('%s000', 'now'),
    strftime('%s000', 'now')
);

-- ============================================================================
-- Database Settings
-- ============================================================================

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Set cache size to 10MB (10000 pages * 1KB)
PRAGMA cache_size = -10000;

-- Set synchronous mode to NORMAL for better performance with WAL
PRAGMA synchronous = NORMAL;
