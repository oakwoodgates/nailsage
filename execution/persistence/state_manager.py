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
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    Attributes:
        db_path: Path to SQLite database file
        _conn: Database connection (lazily created)
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database if it doesn't exist
        if not self.db_path.exists():
            logger.info(f"Creating new database at {self.db_path}")
            self._initialize_database()
        else:
            logger.info(f"Using existing database at {self.db_path}")

        # Validate connection
        self._get_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection.

        Returns:
            SQLite connection
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # Allow use from multiple threads
            )
            self._conn.row_factory = sqlite3.Row  # Access columns by name
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")

        return self._conn

    def _initialize_database(self) -> None:
        """Initialize database schema from schema.sql file."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        conn = self._get_connection()
        conn.executescript(schema_sql)
        conn.commit()

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
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed, rolling back: {e}")
            raise

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

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

        conn = self._get_connection()
        cursor = conn.cursor()

        if strategy.id is None:
            # Insert new strategy
            cursor.execute(
                """
                INSERT INTO strategies
                (strategy_name, version, starlisting_id, interval, model_id, config_path,
                 is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy.strategy_name,
                    strategy.version,
                    strategy.starlisting_id,
                    strategy.interval,
                    strategy.model_id,
                    strategy.config_path,
                    strategy.is_active,
                    now,
                    now,
                ),
            )
            strategy_id = cursor.lastrowid
            logger.info(f"Saved new strategy: {strategy.strategy_name} v{strategy.version} (ID: {strategy_id})")
        else:
            # Update existing strategy
            cursor.execute(
                """
                UPDATE strategies
                SET strategy_name = ?, version = ?, starlisting_id = ?, interval = ?,
                    model_id = ?, config_path = ?, is_active = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    strategy.strategy_name,
                    strategy.version,
                    strategy.starlisting_id,
                    strategy.interval,
                    strategy.model_id,
                    strategy.config_path,
                    strategy.is_active,
                    now,
                    strategy.id,
                ),
            )
            strategy_id = strategy.id
            logger.debug(f"Updated strategy ID {strategy_id}")

        conn.commit()
        return strategy_id

    def get_active_strategies(self) -> List[Strategy]:
        """
        Get all active strategies.

        Returns:
            List of active strategies
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM strategies WHERE is_active = 1")
        rows = cursor.fetchall()

        return [self._row_to_strategy(row) for row in rows]

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        row = cursor.fetchone()

        return self._row_to_strategy(row) if row else None

    def _row_to_strategy(self, row: sqlite3.Row) -> Strategy:
        """Convert database row to Strategy object."""
        return Strategy(
            id=row["id"],
            strategy_name=row["strategy_name"],
            version=row["version"],
            starlisting_id=row["starlisting_id"],
            interval=row["interval"],
            model_id=row["model_id"],
            config_path=row["config_path"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
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

        conn = self._get_connection()
        cursor = conn.cursor()

        if position.id is None:
            # Insert new position
            cursor.execute(
                """
                INSERT INTO positions
                (strategy_id, starlisting_id, side, size, entry_price, entry_timestamp,
                 exit_price, exit_timestamp, realized_pnl, unrealized_pnl, fees_paid,
                 status, stop_loss_price, take_profit_price, exit_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.strategy_id,
                    position.starlisting_id,
                    position.side,
                    position.size,
                    position.entry_price,
                    position.entry_timestamp,
                    position.exit_price,
                    position.exit_timestamp,
                    position.realized_pnl,
                    position.unrealized_pnl,
                    position.fees_paid,
                    position.status,
                    position.stop_loss_price,
                    position.take_profit_price,
                    position.exit_reason,
                    now,
                    now,
                ),
            )
            position_id = cursor.lastrowid
            logger.info(
                f"Saved new position: {position.side} {position.size:.2f} @ {position.entry_price:.2f} "
                f"(ID: {position_id})"
            )
        else:
            # Update existing position
            cursor.execute(
                """
                UPDATE positions
                SET strategy_id = ?, starlisting_id = ?, side = ?, size = ?,
                    entry_price = ?, entry_timestamp = ?, exit_price = ?, exit_timestamp = ?,
                    realized_pnl = ?, unrealized_pnl = ?, fees_paid = ?, status = ?,
                    stop_loss_price = ?, take_profit_price = ?, exit_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    position.strategy_id,
                    position.starlisting_id,
                    position.side,
                    position.size,
                    position.entry_price,
                    position.entry_timestamp,
                    position.exit_price,
                    position.exit_timestamp,
                    position.realized_pnl,
                    position.unrealized_pnl,
                    position.fees_paid,
                    position.status,
                    position.stop_loss_price,
                    position.take_profit_price,
                    position.exit_reason,
                    now,
                    position.id,
                ),
            )
            position_id = position.id
            logger.debug(f"Updated position ID {position_id}")

        conn.commit()
        return position_id

    def get_open_positions(self, strategy_id: Optional[int] = None) -> List[Position]:
        """
        Get all open positions.

        Args:
            strategy_id: Filter by strategy ID (None = all strategies)

        Returns:
            List of open positions
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if strategy_id is not None:
            cursor.execute(
                "SELECT * FROM positions WHERE status = 'open' AND strategy_id = ?",
                (strategy_id,),
            )
        else:
            cursor.execute("SELECT * FROM positions WHERE status = 'open'")

        rows = cursor.fetchall()
        return [self._row_to_position(row) for row in rows]

    def get_position_by_id(self, position_id: int) -> Optional[Position]:
        """Get position by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()

        return self._row_to_position(row) if row else None

    def _row_to_position(self, row: sqlite3.Row) -> Position:
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

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trades
            (position_id, strategy_id, starlisting_id, trade_type, size, price,
             fees, slippage, timestamp, signal_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.position_id,
                trade.strategy_id,
                trade.starlisting_id,
                trade.trade_type,
                trade.size,
                trade.price,
                trade.fees,
                trade.slippage,
                trade.timestamp,
                trade.signal_id,
                now,
            ),
        )

        trade_id = cursor.lastrowid
        conn.commit()

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
        now = int(datetime.now().timestamp() * 1000)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO signals
            (strategy_id, starlisting_id, signal_type, confidence, price_at_signal,
             timestamp, was_executed, rejection_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.strategy_id,
                signal.starlisting_id,
                signal.signal_type,
                signal.confidence,
                signal.price_at_signal,
                signal.timestamp,
                signal.was_executed,
                signal.rejection_reason,
                now,
            ),
        )

        signal_id = cursor.lastrowid
        conn.commit()

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

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO state_snapshots
            (total_equity, available_capital, allocated_capital, total_unrealized_pnl,
             total_realized_pnl, num_open_positions, num_strategies_active, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.total_equity,
                snapshot.available_capital,
                snapshot.allocated_capital,
                snapshot.total_unrealized_pnl,
                snapshot.total_realized_pnl,
                snapshot.num_open_positions,
                snapshot.num_strategies_active,
                snapshot.timestamp,
                now,
            ),
        )

        snapshot_id = cursor.lastrowid
        conn.commit()

        logger.debug(
            f"Saved snapshot: equity={snapshot.total_equity:.2f}, "
            f"open_positions={snapshot.num_open_positions}"
        )

        return snapshot_id

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent state snapshot."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM state_snapshots ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()

        if not row:
            return None

        return StateSnapshot(
            id=row["id"],
            total_equity=row["total_equity"],
            available_capital=row["available_capital"],
            allocated_capital=row["allocated_capital"],
            total_unrealized_pnl=row["total_unrealized_pnl"],
            total_realized_pnl=row["total_realized_pnl"],
            num_open_positions=row["num_open_positions"],
            num_strategies_active=row["num_strategies_active"],
            timestamp=row["timestamp"],
            created_at=row["created_at"],
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

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO event_log (event_type, event_data, severity, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                json.dumps(event_data) if event_data else None,
                severity,
                now,
                now,
            ),
        )

        conn.commit()
