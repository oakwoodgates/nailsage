"""Strategy service for business logic."""

import logging
from typing import List, Optional

from execution.persistence.state_manager import StateManager, Strategy

from api.schemas.strategies import StrategyResponse, StrategyWithStats

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

        with engine.connect() as conn:
            if active_only:
                result = conn.execute(
                    text("SELECT * FROM strategies WHERE is_active = TRUE ORDER BY created_at DESC")
                )
            else:
                result = conn.execute(
                    text("SELECT * FROM strategies ORDER BY created_at DESC")
                )

            strategies = []
            for row in result.mappings():
                strategies.append(StrategyResponse(
                    id=row["id"],
                    strategy_name=row["strategy_name"],
                    version=row["version"],
                    starlisting_id=row["starlisting_id"],
                    interval=row["interval"],
                    model_id=row["model_id"],
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                ))

        return strategies

    def get_strategy_by_id(self, strategy_id: int) -> Optional[StrategyResponse]:
        """Get strategy by ID.

        Args:
            strategy_id: Strategy ID

        Returns:
            Strategy response or None if not found
        """
        strategy = self.state_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            return None

        return StrategyResponse(
            id=strategy.id,
            strategy_name=strategy.strategy_name,
            version=strategy.version,
            starlisting_id=strategy.starlisting_id,
            interval=strategy.interval,
            model_id=strategy.model_id,
            is_active=strategy.is_active,
            created_at=strategy.created_at,
            updated_at=strategy.updated_at,
        )

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
        strategy = self.state_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            return None

        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
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
            id=strategy.id,
            strategy_name=strategy.strategy_name,
            version=strategy.version,
            starlisting_id=strategy.starlisting_id,
            interval=strategy.interval,
            model_id=strategy.model_id,
            is_active=strategy.is_active,
            created_at=strategy.created_at,
            updated_at=strategy.updated_at,
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
