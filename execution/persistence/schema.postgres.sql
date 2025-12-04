-- NailSage Paper Trading State Persistence Schema
-- PostgreSQL database for tracking positions, trades, signals, and system state
--
-- This schema provides:
-- - ACID guarantees for state consistency
-- - Full audit trail of all trading activity
-- - Crash recovery capabilities
-- - Performance tracking and analysis
--
-- Usage:
--   psql -U nailsage -d nailsage < execution/persistence/schema.postgres.sql

-- ============================================================================
-- Exchanges Lookup Table
-- ============================================================================
-- Normalized exchange data (e.g., "binance", "hyperliquid")

CREATE TABLE IF NOT EXISTS exchanges (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(50) NOT NULL UNIQUE,         -- e.g., "hyperliquid", "binance"
    display_name VARCHAR(100) NOT NULL,       -- e.g., "Hyperliquid", "Binance"
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- ============================================================================
-- Coins Lookup Table
-- ============================================================================
-- Normalized coin/asset data (includes both base and quote assets)

CREATE TABLE IF NOT EXISTS coins (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,       -- e.g., "BTC", "USD", "USDT"
    name VARCHAR(100) NOT NULL,               -- e.g., "Bitcoin", "US Dollar"
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- ============================================================================
-- Market Types Lookup Table
-- ============================================================================
-- Normalized market type data (e.g., "perps", "spot", "futures")

CREATE TABLE IF NOT EXISTS market_types (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20) NOT NULL UNIQUE,         -- e.g., "perps", "spot", "futures"
    display VARCHAR(50) NOT NULL,             -- e.g., "Perpetuals", "Spot"
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- ============================================================================
-- Arenas Table
-- ============================================================================
-- Trading arenas representing unique (exchange, pair, interval) combinations
-- Data synced from Kirby API starlistings

CREATE TABLE IF NOT EXISTS arenas (
    id SERIAL PRIMARY KEY,
    starlisting_id INTEGER NOT NULL UNIQUE,   -- Kirby external ID

    -- Trading pair info
    trading_pair VARCHAR(50) NOT NULL,        -- e.g., "BTC/USD"
    trading_pair_id INTEGER,                  -- Kirby trading_pair_id

    -- Foreign keys to lookup tables
    coin_id INTEGER NOT NULL REFERENCES coins(id),
    quote_id INTEGER NOT NULL REFERENCES coins(id),
    exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
    market_type_id INTEGER NOT NULL REFERENCES market_types(id),

    -- Interval (not normalized - unique per arena)
    interval VARCHAR(10) NOT NULL,            -- e.g., "15m"
    interval_seconds INTEGER NOT NULL,        -- e.g., 900

    -- Status & timestamps
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_synced_at BIGINT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_arenas_starlisting ON arenas(starlisting_id);
CREATE INDEX IF NOT EXISTS idx_arenas_trading_pair ON arenas(trading_pair);
CREATE INDEX IF NOT EXISTS idx_arenas_exchange ON arenas(exchange_id);
CREATE INDEX IF NOT EXISTS idx_arenas_coin ON arenas(coin_id);
CREATE INDEX IF NOT EXISTS idx_arenas_market_type ON arenas(market_type_id);

-- ============================================================================
-- Strategies Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    starlisting_id INTEGER NOT NULL,
    arena_id INTEGER REFERENCES arenas(id),  -- Reference to arena (nullable for migration)
    interval VARCHAR(10) NOT NULL,
    model_id VARCHAR(255),  -- NailSage model ID (from model registry)
    config_path TEXT,  -- Path to strategy YAML config
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    initial_bankroll REAL NOT NULL DEFAULT 10000.0,  -- Starting capital for this strategy (USDT)
    current_bankroll REAL NOT NULL DEFAULT 10000.0,  -- Current capital after P&L (USDT)
    created_at BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    updated_at BIGINT NOT NULL,  -- Unix timestamp (milliseconds)

    UNIQUE(strategy_name, version, starlisting_id)
);

CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active);
CREATE INDEX IF NOT EXISTS idx_strategies_starlisting ON strategies(starlisting_id);
CREATE INDEX IF NOT EXISTS idx_strategies_arena ON strategies(arena_id);

-- ============================================================================
-- Positions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL,
    starlisting_id INTEGER NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'long' or 'short'
    size REAL NOT NULL,  -- Position size (USDT)
    entry_price REAL NOT NULL,
    entry_timestamp BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    exit_price REAL,  -- NULL if position is open
    exit_timestamp BIGINT,  -- Unix timestamp (milliseconds)
    realized_pnl REAL,  -- Profit/loss after exit (USDT)
    unrealized_pnl REAL,  -- Current unrealized P&L (updated on each candle)
    fees_paid REAL NOT NULL DEFAULT 0,  -- Total fees paid
    status VARCHAR(20) NOT NULL DEFAULT 'open',  -- 'open', 'closed', 'liquidated'
    stop_loss_price REAL,
    take_profit_price REAL,
    exit_reason VARCHAR(50),  -- 'signal', 'stop_loss', 'take_profit', 'manual', etc.
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,

    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_starlisting ON positions(starlisting_id);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_timestamp);

-- ============================================================================
-- Signals Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL,
    starlisting_id INTEGER NOT NULL,
    signal_type VARCHAR(20) NOT NULL,  -- 'long', 'short', 'neutral', 'close'
    confidence REAL,  -- Model confidence (0-1)
    price_at_signal REAL NOT NULL,  -- Price when signal was generated
    timestamp BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    was_executed BOOLEAN NOT NULL DEFAULT FALSE,  -- Whether signal was acted upon
    rejection_reason VARCHAR(255),  -- If not executed, why? (e.g., "max_positions_reached")
    created_at BIGINT NOT NULL,

    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(was_executed);

-- ============================================================================
-- Trades Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    position_id INTEGER NOT NULL,
    strategy_id INTEGER NOT NULL,
    starlisting_id INTEGER NOT NULL,
    trade_type VARCHAR(20) NOT NULL,  -- 'open_long', 'open_short', 'close_long', 'close_short'
    size REAL NOT NULL,  -- Trade size (USDT)
    price REAL NOT NULL,  -- Execution price
    fees REAL NOT NULL,  -- Transaction fees (USDT)
    slippage REAL NOT NULL,  -- Slippage (USDT)
    timestamp BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    signal_id INTEGER,  -- Reference to signal that triggered this trade
    created_at BIGINT NOT NULL,

    FOREIGN KEY (position_id) REFERENCES positions(id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_signal ON trades(signal_id);

-- ============================================================================
-- System State Snapshots Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS state_snapshots (
    id SERIAL PRIMARY KEY,
    total_equity REAL NOT NULL,  -- Current total equity (USDT)
    available_capital REAL NOT NULL,  -- Available capital (USDT)
    allocated_capital REAL NOT NULL,  -- Capital in open positions (USDT)
    total_unrealized_pnl REAL NOT NULL,  -- Sum of unrealized P&L across all positions
    total_realized_pnl REAL NOT NULL,  -- Sum of realized P&L (all-time)
    num_open_positions INTEGER NOT NULL,
    num_strategies_active INTEGER NOT NULL,
    timestamp BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    created_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_timestamp ON state_snapshots(timestamp);

-- ============================================================================
-- Performance Metrics Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER,  -- NULL = global metrics across all strategies
    timeframe VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly', 'all_time'
    period_start BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    period_end BIGINT NOT NULL,  -- Unix timestamp (milliseconds)

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

    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,

    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_performance_strategy ON performance_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_performance_timeframe ON performance_metrics(timeframe);
CREATE INDEX IF NOT EXISTS idx_performance_period ON performance_metrics(period_start, period_end);

-- ============================================================================
-- Event Log Table (Audit Trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS event_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,  -- 'websocket_connect', 'websocket_disconnect', 'strategy_start', etc.
    event_data TEXT,  -- JSON data for the event
    severity VARCHAR(20) NOT NULL DEFAULT 'INFO',  -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    timestamp BIGINT NOT NULL,  -- Unix timestamp (milliseconds)
    created_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_log_type ON event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_event_log_severity ON event_log(severity);
CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View: Open positions with current unrealized P&L
CREATE OR REPLACE VIEW v_open_positions AS
SELECT
    p.*,
    s.strategy_name,
    s.version,
    s.interval,
    to_timestamp(p.entry_timestamp / 1000.0) as entry_datetime,
    to_timestamp(p.updated_at / 1000.0) as updated_datetime
FROM positions p
JOIN strategies s ON p.strategy_id = s.id
WHERE p.status = 'open';

-- View: Closed positions with P&L
CREATE OR REPLACE VIEW v_closed_positions AS
SELECT
    p.*,
    s.strategy_name,
    s.version,
    s.interval,
    to_timestamp(p.entry_timestamp / 1000.0) as entry_datetime,
    to_timestamp(p.exit_timestamp / 1000.0) as exit_datetime,
    (p.exit_timestamp - p.entry_timestamp) / 1000.0 / 60.0 as duration_minutes
FROM positions p
JOIN strategies s ON p.strategy_id = s.id
WHERE p.status = 'closed';

-- View: Strategy performance summary
CREATE OR REPLACE VIEW v_strategy_performance AS
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
WHERE s.is_active = TRUE
GROUP BY s.id, s.strategy_name, s.version, s.interval;

-- View: Recent signals with execution status
CREATE OR REPLACE VIEW v_recent_signals AS
SELECT
    sig.*,
    s.strategy_name,
    s.version,
    s.interval,
    to_timestamp(sig.timestamp / 1000.0) as signal_datetime
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
    EXTRACT(EPOCH FROM NOW()) * 1000,
    EXTRACT(EPOCH FROM NOW()) * 1000
)
ON CONFLICT DO NOTHING;
