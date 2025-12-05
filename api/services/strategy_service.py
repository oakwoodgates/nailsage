"""Strategy service for business logic."""

import logging
from typing import List, Optional

from execution.persistence.state_manager import StateManager, Strategy

from api.schemas.strategies import StrategyResponse, StrategyWithStats
from api.schemas.arenas import ArenaSummary

logger = logging.getLogger(__name__)


class StrategyService:
    """Service for strategy-related operations."""

    def __init__(self, state_manager: StateManager):
        """Initialize strategy service.

        Args:
            state_manager: Database state manager
        """
        self.state_manager = state_manager

    def get_all_strategies(self, active_only: bool = True) -> List[StrategyResponse]:
        """Get all strategies.

        Args:
            active_only: If True, return only active strategies

        Returns:
            List of strategy responses
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        # Query with LEFT JOIN to arenas for arena summary data
        query = """
            SELECT
                s.*,
                a.id as arena_db_id,
                a.starlisting_id as arena_starlisting_id,
                a.trading_pair,
                a.interval as arena_interval,
                c.symbol as coin,
                c.name as coin_name,
                q.symbol as quote,
                q.name as quote_name,
                e.slug as exchange,
                e.display_name as exchange_name,
                mt.type as market_type,
                mt.display as market_name
            FROM strategies s
            LEFT JOIN arenas a ON s.arena_id = a.id
            LEFT JOIN coins c ON a.coin_id = c.id
            LEFT JOIN coins q ON a.quote_id = q.id
            LEFT JOIN exchanges e ON a.exchange_id = e.id
            LEFT JOIN market_types mt ON a.market_type_id = mt.id
        """

        if active_only:
            query += " WHERE s.is_active = TRUE"

        query += " ORDER BY s.created_at DESC"

        with engine.connect() as conn:
            result = conn.execute(text(query))

            strategies = []
            for row in result.mappings():
                # Build arena summary if arena_id exists
                arena = None
                if row.get("arena_id") and row.get("arena_db_id"):
                    arena = ArenaSummary(
                        id=row["arena_db_id"],
                        starlisting_id=row["arena_starlisting_id"],
                        trading_pair=row["trading_pair"],
                        interval=row["arena_interval"],
                        coin=row["coin"],
                        coin_name=row["coin_name"],
                        quote=row["quote"],
                        quote_name=row["quote_name"],
                        exchange=row["exchange"],
                        exchange_name=row["exchange_name"],
                        market_type=row["market_type"],
                        market_name=row["market_name"],
                    )

                strategies.append(StrategyResponse(
                    id=row["id"],
                    strategy_name=row["strategy_name"],
                    version=row["version"],
                    starlisting_id=row["starlisting_id"],
                    arena_id=row.get("arena_id"),
                    interval=row["interval"],
                    model_id=row["model_id"],
                    is_active=bool(row["is_active"]),
                    initial_bankroll=float(row["initial_bankroll"]) if row.get("initial_bankroll") else 10000.0,
                    current_bankroll=float(row["current_bankroll"]) if row.get("current_bankroll") else 10000.0,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    arena=arena,
                ))

        return strategies

    def get_strategies_by_arena(
        self, arena_id: int, active_only: bool = True
    ) -> List[StrategyResponse]:
        """Get strategies for a specific arena.

        Args:
            arena_id: Arena ID
            active_only: If True, return only active strategies

        Returns:
            List of strategy responses for the arena
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        # Query with LEFT JOIN to arenas for arena summary data
        query = """
            SELECT
                s.*,
                a.id as arena_db_id,
                a.starlisting_id as arena_starlisting_id,
                a.trading_pair,
                a.interval as arena_interval,
                c.symbol as coin,
                c.name as coin_name,
                q.symbol as quote,
                q.name as quote_name,
                e.slug as exchange,
                e.display_name as exchange_name,
                mt.type as market_type,
                mt.display as market_name
            FROM strategies s
            LEFT JOIN arenas a ON s.arena_id = a.id
            LEFT JOIN coins c ON a.coin_id = c.id
            LEFT JOIN coins q ON a.quote_id = q.id
            LEFT JOIN exchanges e ON a.exchange_id = e.id
            LEFT JOIN market_types mt ON a.market_type_id = mt.id
            WHERE s.arena_id = :arena_id
        """

        if active_only:
            query += " AND s.is_active = TRUE"

        query += " ORDER BY s.created_at DESC"

        with engine.connect() as conn:
            result = conn.execute(text(query), {"arena_id": arena_id})

            strategies = []
            for row in result.mappings():
                # Build arena summary if arena_id exists
                arena = None
                if row.get("arena_id") and row.get("arena_db_id"):
                    arena = ArenaSummary(
                        id=row["arena_db_id"],
                        starlisting_id=row["arena_starlisting_id"],
                        trading_pair=row["trading_pair"],
                        interval=row["arena_interval"],
                        coin=row["coin"],
                        coin_name=row["coin_name"],
                        quote=row["quote"],
                        quote_name=row["quote_name"],
                        exchange=row["exchange"],
                        exchange_name=row["exchange_name"],
                        market_type=row["market_type"],
                        market_name=row["market_name"],
                    )

                strategies.append(StrategyResponse(
                    id=row["id"],
                    strategy_name=row["strategy_name"],
                    version=row["version"],
                    starlisting_id=row["starlisting_id"],
                    arena_id=row.get("arena_id"),
                    interval=row["interval"],
                    model_id=row["model_id"],
                    is_active=bool(row["is_active"]),
                    initial_bankroll=float(row["initial_bankroll"]) if row.get("initial_bankroll") else 10000.0,
                    current_bankroll=float(row["current_bankroll"]) if row.get("current_bankroll") else 10000.0,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    arena=arena,
                ))

        return strategies

    def get_strategy_by_id(self, strategy_id: int) -> Optional[StrategyResponse]:
        """Get strategy by ID.

        Args:
            strategy_id: Strategy ID

        Returns:
            Strategy response or None if not found
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        # Query with LEFT JOIN to arenas for arena summary data
        query = """
            SELECT
                s.*,
                a.id as arena_db_id,
                a.starlisting_id as arena_starlisting_id,
                a.trading_pair,
                a.interval as arena_interval,
                c.symbol as coin,
                c.name as coin_name,
                q.symbol as quote,
                q.name as quote_name,
                e.slug as exchange,
                e.display_name as exchange_name,
                mt.type as market_type,
                mt.display as market_name
            FROM strategies s
            LEFT JOIN arenas a ON s.arena_id = a.id
            LEFT JOIN coins c ON a.coin_id = c.id
            LEFT JOIN coins q ON a.quote_id = q.id
            LEFT JOIN exchanges e ON a.exchange_id = e.id
            LEFT JOIN market_types mt ON a.market_type_id = mt.id
            WHERE s.id = :strategy_id
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"strategy_id": strategy_id})
            row = result.mappings().fetchone()

        if not row:
            return None

        # Build arena summary if arena_id exists
        arena = None
        if row.get("arena_id") and row.get("arena_db_id"):
            arena = ArenaSummary(
                id=row["arena_db_id"],
                starlisting_id=row["arena_starlisting_id"],
                trading_pair=row["trading_pair"],
                interval=row["arena_interval"],
                coin=row["coin"],
                coin_name=row["coin_name"],
                quote=row["quote"],
                quote_name=row["quote_name"],
                exchange=row["exchange"],
                exchange_name=row["exchange_name"],
                market_type=row["market_type"],
                market_name=row["market_name"],
            )

        return StrategyResponse(
            id=row["id"],
            strategy_name=row["strategy_name"],
            version=row["version"],
            starlisting_id=row["starlisting_id"],
            arena_id=row.get("arena_id"),
            interval=row["interval"],
            model_id=row["model_id"],
            is_active=bool(row["is_active"]),
            initial_bankroll=float(row["initial_bankroll"]) if row.get("initial_bankroll") else 10000.0,
            current_bankroll=float(row["current_bankroll"]) if row.get("current_bankroll") else 10000.0,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            arena=arena,
        )

    def update_strategy_bankroll(self, strategy_id: int, new_bankroll: float) -> None:
        """Update strategy bankroll.

        Args:
            strategy_id: Strategy ID
            new_bankroll: New bankroll value (USDT)
        """
        self.state_manager.update_strategy_bankroll(strategy_id, new_bankroll)
        logger.info(f"Updated strategy {strategy_id} bankroll to ${new_bankroll:.2f}")

    def get_strategy_with_stats(
        self,
        strategy_id: int,
        initial_capital: float = 100000.0,
    ) -> Optional[StrategyWithStats]:
        """Get strategy with computed performance statistics.

        Args:
            strategy_id: Strategy ID
            initial_capital: Initial capital for P&L percentage calculation

        Returns:
            Strategy with stats or None if not found
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        # Query strategy with LEFT JOIN to arenas for arena summary data
        strategy_query = """
            SELECT
                s.*,
                a.id as arena_db_id,
                a.starlisting_id as arena_starlisting_id,
                a.trading_pair,
                a.interval as arena_interval,
                c.symbol as coin,
                c.name as coin_name,
                q.symbol as quote,
                q.name as quote_name,
                e.slug as exchange,
                e.display_name as exchange_name,
                mt.type as market_type,
                mt.display as market_name
            FROM strategies s
            LEFT JOIN arenas a ON s.arena_id = a.id
            LEFT JOIN coins c ON a.coin_id = c.id
            LEFT JOIN coins q ON a.quote_id = q.id
            LEFT JOIN exchanges e ON a.exchange_id = e.id
            LEFT JOIN market_types mt ON a.market_type_id = mt.id
            WHERE s.id = :strategy_id
        """

        with engine.connect() as conn:
            # Get strategy with arena data
            result = conn.execute(text(strategy_query), {"strategy_id": strategy_id})
            strategy_row = result.mappings().fetchone()

            if not strategy_row:
                return None

            # Get trade counts and P&L metrics
            result = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as total_trades,
                        COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END), 0) as win_count,
                        COALESCE(SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END), 0) as loss_count,
                        COALESCE(SUM(realized_pnl), 0) as realized_pnl,
                        COALESCE(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END), 0) as avg_win,
                        COALESCE(AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END), 0) as avg_loss,
                        COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl END), 0) as gross_profit,
                        COALESCE(ABS(SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl END)), 0) as gross_loss
                    FROM positions
                    WHERE strategy_id = :strategy_id AND status = 'closed'
                """),
                {"strategy_id": strategy_id}
            )
            stats_row = result.mappings().fetchone()

            # Get open positions count and unrealized P&L
            result = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as open_positions,
                        COALESCE(SUM(unrealized_pnl), 0) as unrealized_pnl
                    FROM positions
                    WHERE strategy_id = :strategy_id AND status = 'open'
                """),
                {"strategy_id": strategy_id}
            )
            open_row = result.mappings().fetchone()

        # Build arena summary if arena_id exists
        arena = None
        if strategy_row.get("arena_id") and strategy_row.get("arena_db_id"):
            arena = ArenaSummary(
                id=strategy_row["arena_db_id"],
                starlisting_id=strategy_row["arena_starlisting_id"],
                trading_pair=strategy_row["trading_pair"],
                interval=strategy_row["arena_interval"],
                coin=strategy_row["coin"],
                coin_name=strategy_row["coin_name"],
                quote=strategy_row["quote"],
                quote_name=strategy_row["quote_name"],
                exchange=strategy_row["exchange"],
                exchange_name=strategy_row["exchange_name"],
                market_type=strategy_row["market_type"],
                market_name=strategy_row["market_name"],
            )

        # Calculate metrics
        total_trades = int(stats_row["total_trades"] or 0)
        win_count = int(stats_row["win_count"] or 0)
        loss_count = int(stats_row["loss_count"] or 0)
        realized_pnl = float(stats_row["realized_pnl"] or 0)
        unrealized_pnl = float(open_row["unrealized_pnl"] or 0)
        open_positions = int(open_row["open_positions"] or 0)

        total_pnl = realized_pnl + unrealized_pnl
        total_pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        gross_profit = float(stats_row["gross_profit"] or 0)
        gross_loss = float(stats_row["gross_loss"] or 0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        return StrategyWithStats(
            id=strategy_row["id"],
            strategy_name=strategy_row["strategy_name"],
            version=strategy_row["version"],
            starlisting_id=strategy_row["starlisting_id"],
            arena_id=strategy_row.get("arena_id"),
            interval=strategy_row["interval"],
            model_id=strategy_row["model_id"],
            is_active=bool(strategy_row["is_active"]),
            initial_bankroll=float(strategy_row["initial_bankroll"]) if strategy_row.get("initial_bankroll") else 10000.0,
            current_bankroll=float(strategy_row["current_bankroll"]) if strategy_row.get("current_bankroll") else 10000.0,
            created_at=strategy_row["created_at"],
            updated_at=strategy_row["updated_at"],
            arena=arena,
            total_trades=total_trades,
            open_positions=open_positions,
            total_pnl_usd=total_pnl,
            total_pnl_pct=total_pnl_pct,
            realized_pnl_usd=realized_pnl,
            unrealized_pnl_usd=unrealized_pnl,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_win_usd=float(stats_row["avg_win"] or 0),
            avg_loss_usd=float(stats_row["avg_loss"] or 0),
            profit_factor=profit_factor,
        )
