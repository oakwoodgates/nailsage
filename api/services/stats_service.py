"""Statistics service for financial calculations."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from execution.persistence.state_manager import StateManager

from api.schemas.stats import (
    FinancialSummary,
    DailyPnL,
    DailyPnLResponse,
    StrategyStats,
    LeaderboardEntry,
    LeaderboardResponse,
)

logger = logging.getLogger(__name__)


class StatsService:
    """Service for financial statistics and calculations."""

    def __init__(self, state_manager: StateManager, initial_capital: float = 100000.0):
        """Initialize stats service.

        Args:
            state_manager: Database state manager
            initial_capital: Initial capital for P&L percentage calculations
        """
        self.state_manager = state_manager
        self.initial_capital = initial_capital

    def get_financial_summary(
        self,
        strategy_id: Optional[int] = None,
    ) -> FinancialSummary:
        """Get comprehensive financial summary.

        Args:
            strategy_id: Optional strategy filter

        Returns:
            Financial summary with all metrics
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        now = int(datetime.now().timestamp() * 1000)

        with engine.connect() as conn:
            # Build strategy filter
            strategy_filter = ""
            params = {}
            if strategy_id:
                strategy_filter = "AND strategy_id = :strategy_id"
                params["strategy_id"] = strategy_id

            # Get closed position stats
            result = conn.execute(
                text(f"""
                    SELECT
                        COUNT(*) as total_trades,
                        COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
                        COALESCE(SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
                        COALESCE(SUM(realized_pnl), 0) as realized_pnl,
                        COALESCE(SUM(fees_paid), 0) as total_fees,
                        COALESCE(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END), 0) as avg_win,
                        COALESCE(AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END), 0) as avg_loss,
                        COALESCE(MAX(realized_pnl), 0) as largest_win,
                        COALESCE(MIN(realized_pnl), 0) as largest_loss,
                        COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl END), 0) as gross_profit,
                        COALESCE(ABS(SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl END)), 0) as gross_loss,
                        COALESCE(AVG(realized_pnl), 0) as avg_trade
                    FROM positions
                    WHERE status = 'closed' {strategy_filter}
                """),
                params
            )
            closed_stats = result.mappings().fetchone()

            # Get open position stats
            result = conn.execute(
                text(f"""
                    SELECT
                        COALESCE(SUM(unrealized_pnl), 0) as unrealized_pnl
                    FROM positions
                    WHERE status = 'open' {strategy_filter}
                """),
                params
            )
            open_stats = result.mappings().fetchone()

        # Calculate metrics
        total_trades = int(closed_stats["total_trades"] or 0)
        wins = int(closed_stats["wins"] or 0)
        losses = int(closed_stats["losses"] or 0)
        realized_pnl = float(closed_stats["realized_pnl"] or 0)
        unrealized_pnl = float(open_stats["unrealized_pnl"] or 0)
        total_pnl = realized_pnl + unrealized_pnl

        total_equity = self.initial_capital + total_pnl
        total_pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        gross_profit = float(closed_stats["gross_profit"] or 0)
        gross_loss = float(closed_stats["gross_loss"] or 0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        avg_trade = float(closed_stats["avg_trade"] or 0)
        expectancy = avg_trade if total_trades > 0 else 0

        return FinancialSummary(
            total_equity_usd=total_equity,
            total_pnl_usd=total_pnl,
            total_pnl_pct=total_pnl_pct,
            realized_pnl_usd=realized_pnl,
            unrealized_pnl_usd=unrealized_pnl,
            total_trades=total_trades,
            total_wins=wins,
            total_losses=losses,
            win_rate=win_rate,
            avg_win_usd=float(closed_stats["avg_win"] or 0),
            avg_loss_usd=float(closed_stats["avg_loss"] or 0),
            largest_win_usd=float(closed_stats["largest_win"] or 0),
            largest_loss_usd=float(closed_stats["largest_loss"] or 0),
            profit_factor=profit_factor,
            avg_trade_usd=avg_trade,
            expectancy_usd=expectancy,
            total_fees_paid=float(closed_stats["total_fees"] or 0),
            timestamp=now,
        )

    def get_daily_pnl(
        self,
        days: int = 30,
        strategy_id: Optional[int] = None,
    ) -> DailyPnLResponse:
        """Get daily P&L breakdown.

        Args:
            days: Number of days to include
            strategy_id: Optional strategy filter

        Returns:
            Daily P&L response
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        # Calculate date range
        end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = (end_date - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        with engine.connect() as conn:
            # Build strategy filter
            strategy_filter = ""
            params = {"start_ts": start_ts, "end_ts": end_ts}
            if strategy_id:
                strategy_filter = "AND strategy_id = :strategy_id"
                params["strategy_id"] = strategy_id

            # Get daily aggregated data
            # Note: This query groups by day based on exit_timestamp
            result = conn.execute(
                text(f"""
                    SELECT
                        DATE(datetime(exit_timestamp/1000, 'unixepoch')) as date,
                        COUNT(*) as trades,
                        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                        SUM(realized_pnl) as pnl
                    FROM positions
                    WHERE status = 'closed'
                    AND exit_timestamp >= :start_ts
                    AND exit_timestamp <= :end_ts
                    {strategy_filter}
                    GROUP BY DATE(datetime(exit_timestamp/1000, 'unixepoch'))
                    ORDER BY date
                """),
                params
            )

            daily_records = []
            cumulative_pnl = 0.0

            for row in result.mappings():
                date_str = row["date"]
                pnl = float(row["pnl"] or 0)
                cumulative_pnl += pnl

                # Parse date for timestamp calculation
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                day_start = int(date_obj.timestamp() * 1000)
                day_end = int((date_obj + timedelta(days=1) - timedelta(seconds=1)).timestamp() * 1000)

                starting_equity = self.initial_capital + cumulative_pnl - pnl
                ending_equity = self.initial_capital + cumulative_pnl

                daily_records.append(DailyPnL(
                    date=date_str,
                    timestamp_start=day_start,
                    timestamp_end=day_end,
                    pnl_usd=pnl,
                    pnl_pct=(pnl / starting_equity * 100) if starting_equity > 0 else 0,
                    cumulative_pnl_usd=cumulative_pnl,
                    trades=int(row["trades"] or 0),
                    wins=int(row["wins"] or 0),
                    losses=int(row["losses"] or 0),
                    starting_equity_usd=starting_equity,
                    ending_equity_usd=ending_equity,
                ))

        # Calculate summary stats
        total_days = len(daily_records)
        profitable_days = sum(1 for d in daily_records if d.pnl_usd > 0)
        losing_days = sum(1 for d in daily_records if d.pnl_usd <= 0)
        avg_daily_pnl = (sum(d.pnl_usd for d in daily_records) / total_days) if total_days > 0 else 0
        best_day = max((d.pnl_usd for d in daily_records), default=0)
        worst_day = min((d.pnl_usd for d in daily_records), default=0)

        return DailyPnLResponse(
            daily=daily_records,
            total_days=total_days,
            profitable_days=profitable_days,
            losing_days=losing_days,
            avg_daily_pnl_usd=avg_daily_pnl,
            best_day_usd=best_day,
            worst_day_usd=worst_day,
        )

    def get_stats_by_strategy(self) -> List[StrategyStats]:
        """Get statistics grouped by strategy.

        Returns:
            List of strategy statistics
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        s.id as strategy_id,
                        s.strategy_name,
                        s.is_active,
                        COALESCE(SUM(CASE WHEN p.status = 'closed' THEN p.realized_pnl END), 0) as realized_pnl,
                        COALESCE(SUM(CASE WHEN p.status = 'open' THEN p.unrealized_pnl END), 0) as unrealized_pnl,
                        COUNT(CASE WHEN p.status = 'closed' THEN 1 END) as total_trades,
                        COUNT(CASE WHEN p.status = 'open' THEN 1 END) as open_positions,
                        COUNT(CASE WHEN p.status = 'closed' AND p.realized_pnl > 0 THEN 1 END) as wins,
                        COUNT(CASE WHEN p.status = 'closed' AND p.realized_pnl <= 0 THEN 1 END) as losses,
                        AVG(CASE WHEN p.status = 'closed' AND p.realized_pnl > 0 THEN p.realized_pnl END) as avg_win,
                        AVG(CASE WHEN p.status = 'closed' AND p.realized_pnl < 0 THEN p.realized_pnl END) as avg_loss,
                        SUM(CASE WHEN p.status = 'closed' AND p.realized_pnl > 0 THEN p.realized_pnl END) as gross_profit,
                        ABS(SUM(CASE WHEN p.status = 'closed' AND p.realized_pnl < 0 THEN p.realized_pnl END)) as gross_loss
                    FROM strategies s
                    LEFT JOIN positions p ON s.id = p.strategy_id
                    GROUP BY s.id, s.strategy_name, s.is_active
                    ORDER BY realized_pnl DESC
                """)
            )

            stats = []
            for row in result.mappings():
                total_trades = int(row["total_trades"] or 0)
                wins = int(row["wins"] or 0)
                losses = int(row["losses"] or 0)
                realized_pnl = float(row["realized_pnl"] or 0)
                unrealized_pnl = float(row["unrealized_pnl"] or 0)

                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

                gross_profit = float(row["gross_profit"] or 0)
                gross_loss = float(row["gross_loss"] or 0)
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

                stats.append(StrategyStats(
                    strategy_id=row["strategy_id"],
                    strategy_name=row["strategy_name"],
                    is_active=bool(row["is_active"]),
                    total_pnl_usd=realized_pnl + unrealized_pnl,
                    realized_pnl_usd=realized_pnl,
                    unrealized_pnl_usd=unrealized_pnl,
                    total_trades=total_trades,
                    open_positions=int(row["open_positions"] or 0),
                    win_count=wins,
                    loss_count=losses,
                    win_rate=win_rate,
                    avg_win_usd=float(row["avg_win"] or 0),
                    avg_loss_usd=float(row["avg_loss"] or 0),
                    profit_factor=profit_factor,
                ))

        return stats

    def get_leaderboard(
        self,
        metric: str = "total_pnl",
        limit: int = 10,
    ) -> LeaderboardResponse:
        """Get strategy leaderboard.

        Args:
            metric: Metric to rank by (total_pnl, win_rate, profit_factor)
            limit: Number of entries to return

        Returns:
            Leaderboard response
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        # Validate metric
        valid_metrics = ["total_pnl", "win_rate", "profit_factor"]
        if metric not in valid_metrics:
            metric = "total_pnl"

        # Map metric to SQL column
        metric_column = {
            "total_pnl": "total_pnl",
            "win_rate": "win_rate",
            "profit_factor": "profit_factor",
        }[metric]

        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT
                        s.id as strategy_id,
                        s.strategy_name,
                        s.interval,
                        COALESCE(SUM(p.realized_pnl), 0) + COALESCE(SUM(CASE WHEN p.status = 'open' THEN p.unrealized_pnl END), 0) as total_pnl,
                        COUNT(CASE WHEN p.status = 'closed' THEN 1 END) as total_trades,
                        CASE
                            WHEN COUNT(CASE WHEN p.status = 'closed' THEN 1 END) > 0
                            THEN CAST(COUNT(CASE WHEN p.status = 'closed' AND p.realized_pnl > 0 THEN 1 END) AS FLOAT)
                                 / COUNT(CASE WHEN p.status = 'closed' THEN 1 END) * 100
                            ELSE 0
                        END as win_rate,
                        CASE
                            WHEN COALESCE(ABS(SUM(CASE WHEN p.status = 'closed' AND p.realized_pnl < 0 THEN p.realized_pnl END)), 0) > 0
                            THEN COALESCE(SUM(CASE WHEN p.status = 'closed' AND p.realized_pnl > 0 THEN p.realized_pnl END), 0)
                                 / ABS(SUM(CASE WHEN p.status = 'closed' AND p.realized_pnl < 0 THEN p.realized_pnl END))
                            ELSE 0
                        END as profit_factor,
                        CASE
                            WHEN COUNT(CASE WHEN p.status = 'closed' THEN 1 END) > 0
                            THEN AVG(CASE WHEN p.status = 'closed' THEN p.realized_pnl END)
                            ELSE 0
                        END as avg_trade
                    FROM strategies s
                    LEFT JOIN positions p ON s.id = p.strategy_id
                    WHERE s.is_active = TRUE
                    GROUP BY s.id, s.strategy_name, s.interval
                    ORDER BY {metric_column} DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )

            entries = []
            rank = 1
            for row in result.mappings():
                total_pnl = float(row["total_pnl"] or 0)
                total_pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

                entries.append(LeaderboardEntry(
                    rank=rank,
                    strategy_id=row["strategy_id"],
                    strategy_name=row["strategy_name"],
                    interval=row["interval"],
                    total_pnl_usd=total_pnl,
                    total_pnl_pct=total_pnl_pct,
                    win_rate=float(row["win_rate"] or 0),
                    total_trades=int(row["total_trades"] or 0),
                    profit_factor=float(row["profit_factor"] or 0),
                    avg_trade_usd=float(row["avg_trade"] or 0),
                ))
                rank += 1

            # Get total strategy count
            total_result = conn.execute(
                text("SELECT COUNT(*) FROM strategies WHERE is_active = TRUE")
            )
            total_strategies = total_result.scalar()

        return LeaderboardResponse(
            entries=entries,
            metric=metric,
            total_strategies=total_strategies or 0,
        )
