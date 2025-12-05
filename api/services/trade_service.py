"""Trade service for business logic."""

import logging
from typing import List, Optional, Tuple

from execution.persistence.state_manager import StateManager

from api.schemas.trades import TradeResponse, TradeListResponse

logger = logging.getLogger(__name__)


class TradeService:
    """Service for trade-related operations."""

    def __init__(self, state_manager: StateManager):
        """Initialize trade service.

        Args:
            state_manager: Database state manager
        """
        self.state_manager = state_manager

    def get_trades(
        self,
        strategy_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> TradeListResponse:
        """Get trades with optional filtering.

        Args:
            strategy_id: Filter by strategy ID
            limit: Maximum number of trades to return
            offset: Pagination offset

        Returns:
            Trade list response with pagination info
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            # Build query
            params = {"limit": limit, "offset": offset}

            if strategy_id:
                query = """
                    SELECT t.*, s.strategy_name, p.side as position_side, a.id as arena_id
                    FROM trades t
                    LEFT JOIN strategies s ON t.strategy_id = s.id
                    LEFT JOIN positions p ON t.position_id = p.id
                    LEFT JOIN arenas a ON t.starlisting_id = a.starlisting_id
                    WHERE t.strategy_id = :strategy_id
                    ORDER BY t.timestamp DESC, t.id DESC
                    LIMIT :limit OFFSET :offset
                """
                count_query = "SELECT COUNT(*) FROM trades WHERE strategy_id = :strategy_id"
                params["strategy_id"] = strategy_id
            else:
                query = """
                    SELECT t.*, s.strategy_name, p.side as position_side, a.id as arena_id
                    FROM trades t
                    LEFT JOIN strategies s ON t.strategy_id = s.id
                    LEFT JOIN positions p ON t.position_id = p.id
                    LEFT JOIN arenas a ON t.starlisting_id = a.starlisting_id
                    ORDER BY t.timestamp DESC, t.id DESC
                    LIMIT :limit OFFSET :offset
                """
                count_query = "SELECT COUNT(*) FROM trades"

            # Get total count
            total = conn.execute(text(count_query), params).scalar()

            # Get trades
            result = conn.execute(text(query), params)
            trades = []
            for row in result.mappings():
                trades.append(TradeResponse(
                    id=row["id"],
                    position_id=row["position_id"],
                    strategy_id=row["strategy_id"],
                    starlisting_id=row["starlisting_id"],
                    trade_type=row["trade_type"],
                    size=row["size"],
                    price=row["price"],
                    fees=row["fees"],
                    slippage=row["slippage"],
                    timestamp=row["timestamp"],
                    signal_id=row["signal_id"],
                    created_at=row["created_at"],
                    strategy_name=row["strategy_name"],
                    position_side=row["position_side"],
                    arena_id=row.get("arena_id"),
                ))

        return TradeListResponse(
            trades=trades,
            total=total or 0,
            limit=limit,
            offset=offset,
        )

    def get_trade_by_id(self, trade_id: int) -> Optional[TradeResponse]:
        """Get trade by ID.

        Args:
            trade_id: Trade ID

        Returns:
            Trade response or None if not found
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT t.*, s.strategy_name, p.side as position_side, a.id as arena_id
                    FROM trades t
                    LEFT JOIN strategies s ON t.strategy_id = s.id
                    LEFT JOIN positions p ON t.position_id = p.id
                    LEFT JOIN arenas a ON t.starlisting_id = a.starlisting_id
                    WHERE t.id = :trade_id
                """),
                {"trade_id": trade_id}
            )
            row = result.mappings().fetchone()

        if not row:
            return None

        return TradeResponse(
            id=row["id"],
            position_id=row["position_id"],
            strategy_id=row["strategy_id"],
            starlisting_id=row["starlisting_id"],
            trade_type=row["trade_type"],
            size=row["size"],
            price=row["price"],
            fees=row["fees"],
            slippage=row["slippage"],
            timestamp=row["timestamp"],
            signal_id=row["signal_id"],
            created_at=row["created_at"],
            strategy_name=row["strategy_name"],
            position_side=row["position_side"],
            arena_id=row.get("arena_id"),
        )

    def get_recent_trades(self, limit: int = 10) -> List[TradeResponse]:
        """Get most recent trades.

        Args:
            limit: Number of trades to return

        Returns:
            List of recent trades
        """
        result = self.get_trades(limit=limit, offset=0)
        return result.trades

    def get_trades_by_date_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        strategy_id: Optional[int] = None,
    ) -> List[TradeResponse]:
        """Get trades within a date range.

        Args:
            start_timestamp: Start timestamp (Unix ms)
            end_timestamp: End timestamp (Unix ms)
            strategy_id: Optional strategy filter

        Returns:
            List of trades in the date range
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            params = {
                "start_ts": start_timestamp,
                "end_ts": end_timestamp,
            }

            if strategy_id:
                query = """
                    SELECT t.*, s.strategy_name, p.side as position_side, a.id as arena_id
                    FROM trades t
                    LEFT JOIN strategies s ON t.strategy_id = s.id
                    LEFT JOIN positions p ON t.position_id = p.id
                    LEFT JOIN arenas a ON t.starlisting_id = a.starlisting_id
                    WHERE t.timestamp >= :start_ts AND t.timestamp <= :end_ts
                    AND t.strategy_id = :strategy_id
                    ORDER BY t.timestamp DESC, t.id DESC
                """
                params["strategy_id"] = strategy_id
            else:
                query = """
                    SELECT t.*, s.strategy_name, p.side as position_side, a.id as arena_id
                    FROM trades t
                    LEFT JOIN strategies s ON t.strategy_id = s.id
                    LEFT JOIN positions p ON t.position_id = p.id
                    LEFT JOIN arenas a ON t.starlisting_id = a.starlisting_id
                    WHERE t.timestamp >= :start_ts AND t.timestamp <= :end_ts
                    ORDER BY t.timestamp DESC, t.id DESC
                """

            result = conn.execute(text(query), params)
            trades = []
            for row in result.mappings():
                trades.append(TradeResponse(
                    id=row["id"],
                    position_id=row["position_id"],
                    strategy_id=row["strategy_id"],
                    starlisting_id=row["starlisting_id"],
                    trade_type=row["trade_type"],
                    size=row["size"],
                    price=row["price"],
                    fees=row["fees"],
                    slippage=row["slippage"],
                    timestamp=row["timestamp"],
                    signal_id=row["signal_id"],
                    created_at=row["created_at"],
                    strategy_name=row["strategy_name"],
                    position_side=row["position_side"],
                    arena_id=row.get("arena_id"),
                ))

        return trades
