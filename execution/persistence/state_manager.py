"""State persistence manager for paper trading.

This module provides database access for tracking:
- Strategies
- Positions (open and closed)
- Trades
- Signals
- System state snapshots
- Performance metrics
- Event logs

All timestamps are stored as Unix milliseconds for consistency with Kirby API.

Supports both SQLite (local development) and PostgreSQL (production) via SQLAlchemy.
"""

import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Strategy record."""

    id: Optional[int]
    strategy_name: str
    version: str
    starlisting_id: int
    interval: str
    model_id: Optional[str] = None
    config_path: Optional[str] = None
    is_active: bool = True
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


@dataclass
class Position:
    """Position record."""

    id: Optional[int]
    strategy_id: int
    starlisting_id: int
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_timestamp: int
    exit_price: Optional[float] = None
    exit_timestamp: Optional[int] = None
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    fees_paid: float = 0.0
    status: str = "open"  # 'open', 'closed', 'liquidated'
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


@dataclass
class Trade:
    """Trade record."""

    id: Optional[int]
    position_id: int
    strategy_id: int
    starlisting_id: int
    trade_type: str  # 'open_long', 'open_short', 'close_long', 'close_short'
    size: float
    price: float
    fees: float
    slippage: float
    timestamp: int
    signal_id: Optional[int] = None
    created_at: Optional[int] = None


@dataclass
class Signal:
    """Signal record."""

    id: Optional[int]
    strategy_id: int
    starlisting_id: int
    signal_type: str  # 'long', 'short', 'neutral', 'close'
    confidence: Optional[float]
    price_at_signal: float
    timestamp: int
    was_executed: bool = False
    rejection_reason: Optional[str] = None
    created_at: Optional[int] = None


@dataclass
class StateSnapshot:
    """System state snapshot."""

    id: Optional[int]
    total_equity: float
    available_capital: float
    allocated_capital: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    num_open_positions: int
    num_strategies_active: int
    timestamp: int
    created_at: Optional[int] = None


class StateManager:
    """
    Database manager for paper trading state persistence.

    This class provides ACID-compliant storage for all paper trading state,
    enabling crash recovery and performance tracking.

    Supports both SQLite (local development) and PostgreSQL (production).

    Attributes:
        database_url: Database connection string (SQLite or PostgreSQL)
        _engine: SQLAlchemy engine (lazily created)
    """

    def __init__(self, database_url: Optional[str] = None, db_path: Optional[Path | str] = None):
        """
        Initialize state manager.

        Args:
            database_url: Database URL (e.g., postgresql://user:pass@host/db or sqlite:///path/to/db.db)
                         If not provided, uses DATABASE_URL env var or falls back to db_path
            db_path: Legacy parameter for SQLite path (for backwards compatibility)
        """
        # Determine database URL
        if database_url:
            self.database_url = database_url
        elif os.getenv('DATABASE_URL'):
            self.database_url = os.getenv('DATABASE_URL')
        elif db_path:
            # Convert legacy db_path to SQLite URL
            db_path = Path(db_path)
            # Ensure directory exists for SQLite
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{db_path}"
        else:
            raise ValueError("Either database_url, DATABASE_URL env var, or db_path must be provided")

        self._engine: Optional[Engine] = None
        self._lock = threading.Lock()  # Thread-safe access

        logger.info(f"Initializing StateManager with database: {self._get_safe_url()}")

        # Get engine and validate connection
        engine = self._get_engine()

        # Check if schema exists, initialize if needed
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if 'strategies' not in tables:
            logger.info("Database tables not found, initializing schema")
            self._initialize_database()
        else:
            logger.info(f"Database schema verified ({len(tables)} tables found)")

    def _get_safe_url(self) -> str:
        """Get database URL with password masked for logging."""
        url = self.database_url
        if '://' in url and '@' in url:
            # Mask password in postgresql://user:password@host/db
            protocol, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host = rest.split('@', 1)
                if ':' in credentials:
                    user, _ = credentials.split(':', 1)
                    return f"{protocol}://{user}:***@{host}"
        return url

    def _get_engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine.

        Returns:
            SQLAlchemy engine
        """
        if self._engine is None:
            # Configure connection pooling based on database type
            if self.database_url.startswith('sqlite'):
                # SQLite: Use NullPool (no pooling) to avoid connection issues
                self._engine = create_engine(
                    self.database_url,
                    poolclass=NullPool,
                    connect_args={"check_same_thread": False},
                )
            else:
                # PostgreSQL: Use connection pooling
                self._engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,  # Verify connections before using
                )

        return self._engine

    def _initialize_database(self) -> None:
        """Initialize database schema from schema.sql file."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        engine = self._get_engine()

        # Execute schema SQL
        with engine.begin() as conn:
            # Split by semicolon for PostgreSQL compatibility
            statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
            for statement in statements:
                conn.execute(text(statement))

        logger.info("Database schema initialized")

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Example:
            with state_manager.transaction():
                state_manager.save_position(position)
                state_manager.save_trade(trade)
            # Automatically commits on success, rolls back on exception
        """
        engine = self._get_engine()
        with engine.begin() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Transaction failed, rolling back: {e}")
                raise

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connections closed")

    # ========================================================================
    # Strategy Methods
    # ========================================================================

    def save_strategy(self, strategy: Strategy) -> int:
        """
        Save or update strategy.

        Args:
            strategy: Strategy to save

        Returns:
            Strategy ID
        """
        now = int(datetime.now().timestamp() * 1000)

        engine = self._get_engine()

        with engine.begin() as conn:
            if strategy.id is None:
                # Insert new strategy
                result = conn.execute(
                    text("""
                        INSERT INTO strategies
                        (strategy_name, version, starlisting_id, interval, model_id, config_path,
                         is_active, created_at, updated_at)
                        VALUES (:strategy_name, :version, :starlisting_id, :interval, :model_id,
                                :config_path, :is_active, :created_at, :updated_at)
                    """),
                    {
                        "strategy_name": strategy.strategy_name,
                        "version": strategy.version,
                        "starlisting_id": strategy.starlisting_id,
                        "interval": strategy.interval,
                        "model_id": strategy.model_id,
                        "config_path": strategy.config_path,
                        "is_active": strategy.is_active,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                strategy_id = result.lastrowid
                logger.info(f"Saved new strategy: {strategy.strategy_name} v{strategy.version} (ID: {strategy_id})")
            else:
                # Update existing strategy
                conn.execute(
                    text("""
                        UPDATE strategies
                        SET strategy_name = :strategy_name, version = :version, starlisting_id = :starlisting_id,
                            interval = :interval, model_id = :model_id, config_path = :config_path,
                            is_active = :is_active, updated_at = :updated_at
                        WHERE id = :id
                    """),
                    {
                        "strategy_name": strategy.strategy_name,
                        "version": strategy.version,
                        "starlisting_id": strategy.starlisting_id,
                        "interval": strategy.interval,
                        "model_id": strategy.model_id,
                        "config_path": strategy.config_path,
                        "is_active": strategy.is_active,
                        "updated_at": now,
                        "id": strategy.id,
                    },
                )
                strategy_id = strategy.id
                logger.debug(f"Updated strategy ID {strategy_id}")

        return strategy_id

    def get_active_strategies(self) -> List[Strategy]:
        """
        Get all active strategies.

        Returns:
            List of active strategies
        """
        engine = self._get_engine()

        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM strategies WHERE is_active = TRUE"))
            rows = result.fetchall()

        return [self._row_to_strategy(row) for row in rows]

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID."""
        engine = self._get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM strategies WHERE id = :id"),
                {"id": strategy_id}
            )
            row = result.fetchone()

        return self._row_to_strategy(row) if row else None

    def _row_to_strategy(self, row) -> Strategy:
        """Convert database row to Strategy object."""
        return Strategy(
            id=row.id if hasattr(row, 'id') else row[0],
            strategy_name=row.strategy_name if hasattr(row, 'strategy_name') else row[1],
            version=row.version if hasattr(row, 'version') else row[2],
            starlisting_id=row.starlisting_id if hasattr(row, 'starlisting_id') else row[3],
            interval=row.interval if hasattr(row, 'interval') else row[4],
            model_id=row.model_id if hasattr(row, 'model_id') else row[5],
            config_path=row.config_path if hasattr(row, 'config_path') else row[6],
            is_active=bool(row.is_active if hasattr(row, 'is_active') else row[7]),
            created_at=row.created_at if hasattr(row, 'created_at') else row[8],
            updated_at=row.updated_at if hasattr(row, 'updated_at') else row[9],
        )

    # ========================================================================
    # Position Methods
    # ========================================================================

    def save_position(self, position: Position) -> int:
        """
        Save or update position.

        Args:
            position: Position to save

        Returns:
            Position ID
        """
        now = int(datetime.now().timestamp() * 1000)
        engine = self._get_engine()

        with engine.begin() as conn:
            if position.id is None:
                # Insert new position
                result = conn.execute(
                    text("""
                        INSERT INTO positions
                        (strategy_id, starlisting_id, side, size, entry_price, entry_timestamp,
                         exit_price, exit_timestamp, realized_pnl, unrealized_pnl, fees_paid,
                         status, stop_loss_price, take_profit_price, exit_reason, created_at, updated_at)
                        VALUES (:strategy_id, :starlisting_id, :side, :size, :entry_price, :entry_timestamp,
                                :exit_price, :exit_timestamp, :realized_pnl, :unrealized_pnl, :fees_paid,
                                :status, :stop_loss_price, :take_profit_price, :exit_reason, :created_at, :updated_at)
                    """),
                    {
                        "strategy_id": position.strategy_id,
                        "starlisting_id": position.starlisting_id,
                        "side": position.side,
                        "size": position.size,
                        "entry_price": position.entry_price,
                        "entry_timestamp": position.entry_timestamp,
                        "exit_price": position.exit_price,
                        "exit_timestamp": position.exit_timestamp,
                        "realized_pnl": position.realized_pnl,
                        "unrealized_pnl": position.unrealized_pnl,
                        "fees_paid": position.fees_paid,
                        "status": position.status,
                        "stop_loss_price": position.stop_loss_price,
                        "take_profit_price": position.take_profit_price,
                        "exit_reason": position.exit_reason,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                position_id = result.lastrowid
                logger.info(
                    f"Saved new position: {position.side} {position.size:.2f} @ {position.entry_price:.2f} "
                    f"(ID: {position_id})"
                )
            else:
                # Update existing position
                conn.execute(
                    text("""
                        UPDATE positions
                        SET strategy_id = :strategy_id, starlisting_id = :starlisting_id, side = :side, size = :size,
                            entry_price = :entry_price, entry_timestamp = :entry_timestamp, exit_price = :exit_price,
                            exit_timestamp = :exit_timestamp, realized_pnl = :realized_pnl, unrealized_pnl = :unrealized_pnl,
                            fees_paid = :fees_paid, status = :status, stop_loss_price = :stop_loss_price,
                            take_profit_price = :take_profit_price, exit_reason = :exit_reason, updated_at = :updated_at
                        WHERE id = :id
                    """),
                    {
                        "strategy_id": position.strategy_id,
                        "starlisting_id": position.starlisting_id,
                        "side": position.side,
                        "size": position.size,
                        "entry_price": position.entry_price,
                        "entry_timestamp": position.entry_timestamp,
                        "exit_price": position.exit_price,
                        "exit_timestamp": position.exit_timestamp,
                        "realized_pnl": position.realized_pnl,
                        "unrealized_pnl": position.unrealized_pnl,
                        "fees_paid": position.fees_paid,
                        "status": position.status,
                        "stop_loss_price": position.stop_loss_price,
                        "take_profit_price": position.take_profit_price,
                        "exit_reason": position.exit_reason,
                        "updated_at": now,
                        "id": position.id,
                    },
                )
                position_id = position.id
                logger.debug(f"Updated position ID {position_id}")

        return position_id

    def get_open_positions(self, strategy_id: Optional[int] = None) -> List[Position]:
        """
        Get all open positions.

        Args:
            strategy_id: Filter by strategy ID (None = all strategies)

        Returns:
            List of open positions
        """
        engine = self._get_engine()

        with engine.connect() as conn:
            if strategy_id is not None:
                result = conn.execute(
                    text("SELECT * FROM positions WHERE status = 'open' AND strategy_id = :strategy_id"),
                    {"strategy_id": strategy_id},
                )
            else:
                result = conn.execute(text("SELECT * FROM positions WHERE status = 'open'"))

            rows = result.fetchall()

        return [self._row_to_position(row) for row in rows]

    def get_position_by_id(self, position_id: int) -> Optional[Position]:
        """Get position by ID."""
        engine = self._get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM positions WHERE id = :position_id"),
                {"position_id": position_id}
            )
            row = result.fetchone()

        return self._row_to_position(row) if row else None

    def _row_to_position(self, row: Any) -> Position:
        """Convert database row to Position object."""
        return Position(
            id=row["id"],
            strategy_id=row["strategy_id"],
            starlisting_id=row["starlisting_id"],
            side=row["side"],
            size=row["size"],
            entry_price=row["entry_price"],
            entry_timestamp=row["entry_timestamp"],
            exit_price=row["exit_price"],
            exit_timestamp=row["exit_timestamp"],
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
            fees_paid=row["fees_paid"],
            status=row["status"],
            stop_loss_price=row["stop_loss_price"],
            take_profit_price=row["take_profit_price"],
            exit_reason=row["exit_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ========================================================================
    # Trade Methods
    # ========================================================================

    def save_trade(self, trade: Trade) -> int:
        """
        Save trade execution.

        Args:
            trade: Trade to save

        Returns:
            Trade ID
        """
        now = int(datetime.now().timestamp() * 1000)
        engine = self._get_engine()

        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO trades
                    (position_id, strategy_id, starlisting_id, trade_type, size, price,
                     fees, slippage, timestamp, signal_id, created_at)
                    VALUES (:position_id, :strategy_id, :starlisting_id, :trade_type, :size, :price,
                            :fees, :slippage, :timestamp, :signal_id, :created_at)
                """),
                {
                    "position_id": trade.position_id,
                    "strategy_id": trade.strategy_id,
                    "starlisting_id": trade.starlisting_id,
                    "trade_type": trade.trade_type,
                    "size": trade.size,
                    "price": trade.price,
                    "fees": trade.fees,
                    "slippage": trade.slippage,
                    "timestamp": trade.timestamp,
                    "signal_id": trade.signal_id,
                    "created_at": now,
                },
            )
            trade_id = result.lastrowid

        logger.info(
            f"Saved trade: {trade.trade_type} {trade.size:.2f} @ {trade.price:.2f} "
            f"(ID: {trade_id})"
        )

        return trade_id

    # ========================================================================
    # Signal Methods
    # ========================================================================

    def save_signal(self, signal: Signal) -> int:
        """
        Save signal.

        Args:
            signal: Signal to save

        Returns:
            Signal ID
        """
        with self._lock:  # Thread-safe access
            now = int(datetime.now().timestamp() * 1000)
            engine = self._get_engine()

            with engine.begin() as conn:
                result = conn.execute(
                    text("""
                        INSERT INTO signals
                        (strategy_id, starlisting_id, signal_type, confidence, price_at_signal,
                         timestamp, was_executed, rejection_reason, created_at)
                        VALUES (:strategy_id, :starlisting_id, :signal_type, :confidence, :price_at_signal,
                                :timestamp, :was_executed, :rejection_reason, :created_at)
                    """),
                    {
                        "strategy_id": signal.strategy_id,
                        "starlisting_id": signal.starlisting_id,
                        "signal_type": signal.signal_type,
                        "confidence": signal.confidence,
                        "price_at_signal": signal.price_at_signal,
                        "timestamp": signal.timestamp,
                        "was_executed": signal.was_executed,
                        "rejection_reason": signal.rejection_reason,
                        "created_at": now,
                    },
                )
                signal_id = result.lastrowid

            logger.debug(
                f"Saved signal: {signal.signal_type} @ {signal.price_at_signal:.2f} "
                f"(ID: {signal_id})"
            )

            return signal_id

    # ========================================================================
    # State Snapshot Methods
    # ========================================================================

    def save_snapshot(self, snapshot: StateSnapshot) -> int:
        """
        Save system state snapshot.

        Args:
            snapshot: Snapshot to save

        Returns:
            Snapshot ID
        """
        now = int(datetime.now().timestamp() * 1000)
        engine = self._get_engine()

        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO state_snapshots
                    (total_equity, available_capital, allocated_capital, total_unrealized_pnl,
                     total_realized_pnl, num_open_positions, num_strategies_active, timestamp, created_at)
                    VALUES (:total_equity, :available_capital, :allocated_capital, :total_unrealized_pnl,
                            :total_realized_pnl, :num_open_positions, :num_strategies_active, :timestamp, :created_at)
                """),
                {
                    "total_equity": snapshot.total_equity,
                    "available_capital": snapshot.available_capital,
                    "allocated_capital": snapshot.allocated_capital,
                    "total_unrealized_pnl": snapshot.total_unrealized_pnl,
                    "total_realized_pnl": snapshot.total_realized_pnl,
                    "num_open_positions": snapshot.num_open_positions,
                    "num_strategies_active": snapshot.num_strategies_active,
                    "timestamp": snapshot.timestamp,
                    "created_at": now,
                },
            )
            snapshot_id = result.lastrowid

        logger.debug(
            f"Saved snapshot: equity={snapshot.total_equity:.2f}, "
            f"open_positions={snapshot.num_open_positions}"
        )

        return snapshot_id

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent state snapshot."""
        engine = self._get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM state_snapshots ORDER BY timestamp DESC LIMIT 1")
            )
            row = result.fetchone()

        if not row:
            return None

        return StateSnapshot(
            id=row.id,
            total_equity=row.total_equity,
            available_capital=row.available_capital,
            allocated_capital=row.allocated_capital,
            total_unrealized_pnl=row.total_unrealized_pnl,
            total_realized_pnl=row.total_realized_pnl,
            num_open_positions=row.num_open_positions,
            num_strategies_active=row.num_strategies_active,
            timestamp=row.timestamp,
            created_at=row.created_at,
        )

    # ========================================================================
    # Event Log Methods
    # ========================================================================

    def log_event(
        self,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> None:
        """
        Log an event to the audit trail.

        Args:
            event_type: Type of event (e.g., 'websocket_connect')
            event_data: Optional event data (will be JSON-encoded)
            severity: Event severity ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        now = int(datetime.now().timestamp() * 1000)
        engine = self._get_engine()

        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO event_log (event_type, event_data, severity, timestamp, created_at)
                    VALUES (:event_type, :event_data, :severity, :timestamp, :created_at)
                """),
                {
                    "event_type": event_type,
                    "event_data": json.dumps(event_data) if event_data else None,
                    "severity": severity,
                    "timestamp": now,
                    "created_at": now,
                },
            )
