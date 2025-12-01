"""Position service for business logic."""

import logging
from typing import List, Literal, Optional

from execution.persistence.state_manager import StateManager

from api.schemas.positions import PositionResponse, PositionListResponse

logger = logging.getLogger(__name__)


class PositionService:
    """Service for position-related operations."""

    def __init__(self, state_manager: StateManager):
        """Initialize position service.

        Args:
            state_manager: Database state manager
        """
        self.state_manager = state_manager

    def get_positions(
        self,
        strategy_id: Optional[int] = None,
        status: Optional[Literal["open", "closed"]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PositionListResponse:
        """Get positions with optional filtering.

        Args:
            strategy_id: Filter by strategy ID
            status: Filter by status (open/closed)
            limit: Maximum number of positions to return
            offset: Pagination offset

        Returns:
            Position list response with pagination info
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            # Build query dynamically
            conditions = ["1=1"]
            params = {"limit": limit, "offset": offset}

            if strategy_id:
                conditions.append("p.strategy_id = :strategy_id")
                params["strategy_id"] = strategy_id

            if status:
                conditions.append("p.status = :status")
                params["status"] = status

            where_clause = " AND ".join(conditions)

            # Get total count
            count_query = f"SELECT COUNT(*) FROM positions p WHERE {where_clause}"
            total = conn.execute(text(count_query), params).scalar()

            # Get positions with strategy info
            query = f"""
                SELECT p.*, s.strategy_name,
                       CASE
                           WHEN p.exit_timestamp IS NOT NULL AND p.entry_timestamp IS NOT NULL
                           THEN (p.exit_timestamp - p.entry_timestamp) / 60000.0
                           ELSE NULL
                       END as duration_minutes,
                       CASE
                           WHEN p.entry_price > 0 AND p.realized_pnl IS NOT NULL
                           THEN (p.realized_pnl / p.size) * 100
                           ELSE NULL
                       END as pnl_pct
                FROM positions p
                LEFT JOIN strategies s ON p.strategy_id = s.id
                WHERE {where_clause}
                ORDER BY p.entry_timestamp DESC
                LIMIT :limit OFFSET :offset
            """

            result = conn.execute(text(query), params)
            positions = []
            for row in result.mappings():
                positions.append(self._row_to_response(row))

        return PositionListResponse(
            positions=positions,
            total=total or 0,
            limit=limit,
            offset=offset,
        )

    def get_open_positions(
        self,
        strategy_id: Optional[int] = None,
    ) -> List[PositionResponse]:
        """Get all open positions.

        Args:
            strategy_id: Optional strategy filter

        Returns:
            List of open positions
        """
        result = self.get_positions(
            strategy_id=strategy_id,
            status="open",
            limit=1000,  # Reasonable max for open positions
            offset=0,
        )
        return result.positions

    def get_closed_positions(
        self,
        strategy_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> PositionListResponse:
        """Get closed positions (historical).

        Args:
            strategy_id: Optional strategy filter
            limit: Maximum number of positions to return
            offset: Pagination offset

        Returns:
            Position list response
        """
        return self.get_positions(
            strategy_id=strategy_id,
            status="closed",
            limit=limit,
            offset=offset,
        )

    def get_position_by_id(self, position_id: int) -> Optional[PositionResponse]:
        """Get position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position response or None if not found
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT p.*, s.strategy_name,
                           CASE
                               WHEN p.exit_timestamp IS NOT NULL AND p.entry_timestamp IS NOT NULL
                               THEN (p.exit_timestamp - p.entry_timestamp) / 60000.0
                               ELSE NULL
                           END as duration_minutes,
                           CASE
                               WHEN p.entry_price > 0 AND p.realized_pnl IS NOT NULL
                               THEN (p.realized_pnl / p.size) * 100
                               ELSE NULL
                           END as pnl_pct
                    FROM positions p
                    LEFT JOIN strategies s ON p.strategy_id = s.id
                    WHERE p.id = :position_id
                """),
                {"position_id": position_id}
            )
            row = result.mappings().fetchone()

        if not row:
            return None

        return self._row_to_response(row)

    def _row_to_response(self, row) -> PositionResponse:
        """Convert database row to PositionResponse."""
        return PositionResponse(
            id=row["id"],
            strategy_id=row["strategy_id"],
            starlisting_id=row["starlisting_id"],
            side=row["side"],
            size=row["size"],
            entry_price=row["entry_price"],
            entry_timestamp=row["entry_timestamp"],
            exit_price=row["exit_price"],
            exit_timestamp=row["exit_timestamp"],
            exit_reason=row["exit_reason"],
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
            fees_paid=row["fees_paid"],
            status=row["status"],
            stop_loss_price=row["stop_loss_price"],
            take_profit_price=row["take_profit_price"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            strategy_name=row.get("strategy_name"),
            duration_minutes=row.get("duration_minutes"),
            pnl_pct=row.get("pnl_pct"),
        )
