"""Portfolio service for aggregated portfolio data."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from execution.persistence.state_manager import StateManager

from api.schemas.portfolio import (
    PortfolioSummary,
    AllocationItem,
    AllocationResponse,
    ExposureSummary,
    EquityPoint,
    EquityHistoryResponse,
)

logger = logging.getLogger(__name__)


class PortfolioService:
    """Service for portfolio-level operations."""

    def __init__(
        self,
        state_manager: StateManager,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
    ):
        """Initialize portfolio service.

        Args:
            state_manager: Database state manager
            initial_capital: Initial capital for calculations
            max_positions: Maximum allowed positions
        """
        self.state_manager = state_manager
        self.initial_capital = initial_capital
        self.max_positions = max_positions

    def get_summary(self) -> PortfolioSummary:
        """Get complete portfolio summary.

        Returns:
            Portfolio summary with all metrics
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        now = int(datetime.now().timestamp() * 1000)
        today_start = int(datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp() * 1000)

        with engine.connect() as conn:
            # Get closed position stats
            result = conn.execute(
                text("""
                    SELECT
                        COALESCE(SUM(realized_pnl), 0) as realized_pnl,
                        COALESCE(SUM(fees_paid), 0) as total_fees
                    FROM positions
                    WHERE status = 'closed'
                """)
            )
            closed_stats = result.mappings().fetchone()

            # Get open position stats
            result = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as open_count,
                        COALESCE(SUM(unrealized_pnl), 0) as unrealized_pnl,
                        COALESCE(SUM(size), 0) as allocated_capital,
                        COALESCE(SUM(CASE WHEN side = 'long' THEN size ELSE 0 END), 0) as long_exposure,
                        COALESCE(SUM(CASE WHEN side = 'short' THEN size ELSE 0 END), 0) as short_exposure
                    FROM positions
                    WHERE status = 'open'
                """)
            )
            open_stats = result.mappings().fetchone()

            # Get strategy counts
            result = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN is_active THEN 1 END) as active
                    FROM strategies
                """)
            )
            strategy_stats = result.mappings().fetchone()

            # Get trades today
            result = conn.execute(
                text("""
                    SELECT COUNT(*) as count
                    FROM trades
                    WHERE timestamp >= :today_start
                """),
                {"today_start": today_start}
            )
            trades_today = result.scalar() or 0

        # Calculate metrics
        realized_pnl = float(closed_stats["realized_pnl"] or 0)
        unrealized_pnl = float(open_stats["unrealized_pnl"] or 0)
        total_pnl = realized_pnl + unrealized_pnl

        total_equity = self.initial_capital + total_pnl
        allocated_capital = float(open_stats["allocated_capital"] or 0)
        available_capital = total_equity - allocated_capital

        open_positions = int(open_stats["open_count"] or 0)
        position_utilization = (open_positions / self.max_positions * 100) if self.max_positions > 0 else 0

        long_exposure = float(open_stats["long_exposure"] or 0)
        short_exposure = float(open_stats["short_exposure"] or 0)
        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        return PortfolioSummary(
            initial_capital_usd=self.initial_capital,
            total_equity_usd=total_equity,
            available_capital_usd=available_capital,
            allocated_capital_usd=allocated_capital,
            total_pnl_usd=total_pnl,
            total_pnl_pct=(total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            realized_pnl_usd=realized_pnl,
            unrealized_pnl_usd=unrealized_pnl,
            total_open_positions=open_positions,
            max_positions=self.max_positions,
            position_utilization_pct=position_utilization,
            total_exposure_usd=total_exposure,
            long_exposure_usd=long_exposure,
            short_exposure_usd=short_exposure,
            net_exposure_usd=net_exposure,
            active_strategies=int(strategy_stats["active"] or 0),
            total_strategies=int(strategy_stats["total"] or 0),
            trades_today=trades_today,
            total_fees_paid=float(closed_stats["total_fees"] or 0),
            timestamp=now,
        )

    def get_allocation(self) -> AllocationResponse:
        """Get capital allocation by strategy.

        Returns:
            Allocation response with per-strategy breakdown
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        s.id as strategy_id,
                        s.strategy_name,
                        COALESCE(SUM(p.size), 0) as allocated,
                        COUNT(p.id) as open_positions,
                        COALESCE(SUM(p.unrealized_pnl), 0) as unrealized_pnl
                    FROM strategies s
                    LEFT JOIN positions p ON s.id = p.strategy_id AND p.status = 'open'
                    WHERE s.is_active = TRUE
                    GROUP BY s.id, s.strategy_name
                    ORDER BY allocated DESC
                """)
            )

            allocations = []
            total_allocated = 0.0

            for row in result.mappings():
                allocated = float(row["allocated"] or 0)
                total_allocated += allocated

                allocations.append(AllocationItem(
                    strategy_id=row["strategy_id"],
                    strategy_name=row["strategy_name"],
                    allocated_usd=allocated,
                    allocation_pct=0,  # Will calculate after total is known
                    open_positions=int(row["open_positions"] or 0),
                    unrealized_pnl_usd=float(row["unrealized_pnl"] or 0),
                ))

        # Calculate percentages
        for item in allocations:
            if total_allocated > 0:
                item.allocation_pct = (item.allocated_usd / total_allocated * 100)

        # Calculate unallocated
        summary = self.get_summary()
        unallocated = summary.available_capital_usd

        return AllocationResponse(
            allocations=allocations,
            total_allocated_usd=total_allocated,
            unallocated_usd=unallocated,
        )

    def get_exposure(self) -> ExposureSummary:
        """Get current market exposure summary.

        Returns:
            Exposure summary
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            # Get exposure by side
            result = conn.execute(
                text("""
                    SELECT
                        COALESCE(SUM(CASE WHEN side = 'long' THEN size ELSE 0 END), 0) as long_exposure,
                        COALESCE(SUM(CASE WHEN side = 'short' THEN size ELSE 0 END), 0) as short_exposure
                    FROM positions
                    WHERE status = 'open'
                """)
            )
            exposure_stats = result.mappings().fetchone()

            # Get exposure by starlisting
            result = conn.execute(
                text("""
                    SELECT
                        starlisting_id,
                        side,
                        SUM(size) as exposure
                    FROM positions
                    WHERE status = 'open'
                    GROUP BY starlisting_id, side
                """)
            )

            by_starlisting = []
            for row in result.mappings():
                by_starlisting.append({
                    "starlisting_id": row["starlisting_id"],
                    "side": row["side"],
                    "exposure_usd": float(row["exposure"] or 0),
                })

        long_exposure = float(exposure_stats["long_exposure"] or 0)
        short_exposure = float(exposure_stats["short_exposure"] or 0)
        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        # Get current equity for percentage calculation
        summary = self.get_summary()
        exposure_pct = (gross_exposure / summary.total_equity_usd * 100) if summary.total_equity_usd > 0 else 0

        return ExposureSummary(
            total_exposure_usd=total_exposure,
            long_exposure_usd=long_exposure,
            short_exposure_usd=short_exposure,
            net_exposure_usd=net_exposure,
            gross_exposure_usd=gross_exposure,
            exposure_pct_of_equity=exposure_pct,
            by_starlisting=by_starlisting if by_starlisting else None,
        )

    def get_equity_history(
        self,
        limit: int = 1000,
    ) -> EquityHistoryResponse:
        """Get equity curve from state snapshots.

        Args:
            limit: Maximum number of points to return

        Returns:
            Equity history response
        """
        engine = self.state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        timestamp,
                        total_equity,
                        total_realized_pnl,
                        total_unrealized_pnl,
                        num_open_positions
                    FROM state_snapshots
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )

            points = []
            max_equity = self.initial_capital
            min_equity = self.initial_capital
            max_drawdown_usd = 0.0
            peak_equity = self.initial_capital

            for row in result.mappings():
                equity = float(row["total_equity"] or self.initial_capital)

                points.append(EquityPoint(
                    timestamp=row["timestamp"],
                    equity_usd=equity,
                    realized_pnl_usd=float(row["total_realized_pnl"] or 0),
                    unrealized_pnl_usd=float(row["total_unrealized_pnl"] or 0),
                    open_positions=int(row["num_open_positions"] or 0),
                ))

                # Track min/max
                max_equity = max(max_equity, equity)
                min_equity = min(min_equity, equity)

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = peak_equity - equity
                max_drawdown_usd = max(max_drawdown_usd, drawdown)

        # Reverse to chronological order
        points.reverse()

        # Get current equity
        current_equity = points[-1].equity_usd if points else self.initial_capital

        max_drawdown_pct = (max_drawdown_usd / peak_equity * 100) if peak_equity > 0 else 0

        return EquityHistoryResponse(
            points=points,
            initial_capital_usd=self.initial_capital,
            current_equity_usd=current_equity,
            max_equity_usd=max_equity,
            min_equity_usd=min_equity,
            max_drawdown_usd=max_drawdown_usd,
            max_drawdown_pct=max_drawdown_pct,
        )
