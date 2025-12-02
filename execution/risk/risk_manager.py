"""Centralized risk management for trading operations.

Coordinates capital allocation and exposure tracking to provide
unified pre-trade risk checks.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from execution.risk.capital_allocator import CapitalAllocator
from execution.risk.exposure_tracker import ExposureTracker

logger = logging.getLogger(__name__)


class RiskCheckStatus(Enum):
    """Status of a risk check."""

    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"  # Approved with warning


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check.

    Attributes:
        status: APPROVED, REJECTED, or WARNING
        approved: True if trade can proceed
        reason: Explanation (why rejected or warning)
        warnings: List of warning messages
        max_position_size: Maximum position size allowed (USD)
    """

    status: RiskCheckStatus
    approved: bool
    reason: Optional[str] = None
    warnings: list[str] = None
    max_position_size: Optional[float] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RiskManager:
    """
    Centralized risk management for trading.

    Coordinates capital allocation, exposure tracking, and risk limits
    to provide unified pre-trade validation.

    Attributes:
        capital_allocator: Manages capital allocation
        exposure_tracker: Tracks asset/strategy exposure
        enable_checks: If False, all checks return approved (for testing)
    """

    def __init__(
        self,
        capital_allocator: CapitalAllocator,
        exposure_tracker: ExposureTracker,
        enable_checks: bool = True,
    ):
        """
        Initialize risk manager.

        Args:
            capital_allocator: CapitalAllocator instance
            exposure_tracker: ExposureTracker instance
            enable_checks: Enable risk checks (default: True)
        """
        self.capital_allocator = capital_allocator
        self.exposure_tracker = exposure_tracker
        self.enable_checks = enable_checks

        logger.info(
            f"Initialized RiskManager (checks_enabled={enable_checks})"
        )

    def check_new_position(
        self,
        asset: str,
        strategy: str,
        position_size_usd: float,
        price: float,
    ) -> RiskCheckResult:
        """
        Check if a new position can be opened.

        Args:
            asset: Asset symbol (e.g., 'BTC/USDT')
            strategy: Strategy name
            position_size_usd: Desired position size in USD
            price: Current asset price

        Returns:
            RiskCheckResult indicating approval/rejection
        """
        if not self.enable_checks:
            return RiskCheckResult(
                status=RiskCheckStatus.APPROVED,
                approved=True,
                reason="Risk checks disabled",
                max_position_size=position_size_usd,
            )

        warnings = []

        # Check capital availability
        can_allocate, capital_reason = self.capital_allocator.can_allocate(
            position_size_usd
        )

        if not can_allocate:
            return RiskCheckResult(
                status=RiskCheckStatus.REJECTED,
                approved=False,
                reason=f"Capital check failed: {capital_reason}",
            )

        # Check exposure limits
        capital_state = self.capital_allocator.get_state()
        can_add, exposure_reason = self.exposure_tracker.can_add_position(
            asset=asset,
            strategy=strategy,
            exposure_usd=position_size_usd,
            total_capital=capital_state.total_capital,
        )

        if not can_add:
            return RiskCheckResult(
                status=RiskCheckStatus.REJECTED,
                approved=False,
                reason=f"Exposure check failed: {exposure_reason}",
            )

        # Check for high capital utilization (warning only)
        utilization = capital_state.utilization_pct()
        if utilization > 80.0:
            warnings.append(
                f"High capital utilization: {utilization:.1f}%"
            )

        # Check for concentrated exposure (warning only)
        asset_exposure = self.exposure_tracker.get_asset_exposure(asset)
        new_asset_exposure = asset_exposure + position_size_usd
        asset_exposure_pct = (
            (new_asset_exposure / capital_state.total_capital) * 100.0
        )

        if asset_exposure_pct > 15.0:
            warnings.append(
                f"Concentrated exposure to {asset}: {asset_exposure_pct:.1f}%"
            )

        # Calculate max position size
        max_size = self.capital_allocator.get_max_position_size(price)

        # Determine final status
        if warnings:
            status = RiskCheckStatus.WARNING
            reason = f"Approved with {len(warnings)} warning(s)"
        else:
            status = RiskCheckStatus.APPROVED
            reason = "All risk checks passed"

        return RiskCheckResult(
            status=status,
            approved=True,
            reason=reason,
            warnings=warnings,
            max_position_size=max_size,
        )

    def allocate_position(
        self,
        position_id: int,
        asset: str,
        strategy: str,
        position_size_usd: float,
    ) -> None:
        """
        Allocate capital and track exposure for a new position.

        This should be called after check_new_position() approves the trade
        and the position is actually opened.

        Args:
            position_id: Position identifier
            asset: Asset symbol
            strategy: Strategy name
            position_size_usd: Position size in USD

        Raises:
            ValueError: If allocation fails
        """
        # Allocate capital
        self.capital_allocator.allocate(position_id, position_size_usd)

        # Track exposure
        self.exposure_tracker.add_position(
            position_id=position_id,
            asset=asset,
            strategy=strategy,
            exposure_usd=position_size_usd,
        )

        logger.info(
            f"Allocated position {position_id}: {asset} via {strategy}, "
            f"size=${position_size_usd:,.2f}"
        )

    def deallocate_position(self, position_id: int) -> None:
        """
        Deallocate capital and remove exposure tracking for a closed position.

        Args:
            position_id: Position identifier

        Raises:
            ValueError: If position not found
        """
        # Deallocate capital
        self.capital_allocator.deallocate(position_id)

        # Remove exposure tracking
        self.exposure_tracker.remove_position(position_id)

        logger.info(f"Deallocated position {position_id}")

    def update_equity(self, new_equity: float) -> None:
        """
        Update equity after P&L changes.

        Args:
            new_equity: New equity value

        Raises:
            ValueError: If equity is invalid
        """
        self.capital_allocator.update_equity(new_equity)

    def update_position_exposure(
        self,
        position_id: int,
        new_exposure_usd: float,
    ) -> None:
        """
        Update exposure for a position (e.g., after price changes).

        Args:
            position_id: Position identifier
            new_exposure_usd: New exposure amount

        Raises:
            ValueError: If position not found or exposure invalid
        """
        self.exposure_tracker.update_exposure(position_id, new_exposure_usd)

    def get_risk_summary(self) -> dict:
        """
        Get comprehensive risk summary.

        Returns:
            Dict with capital and exposure state
        """
        capital_state = self.capital_allocator.get_state()
        exposure_state = self.exposure_tracker.get_state()

        return {
            "capital": {
                "total": capital_state.total_capital,
                "allocated": capital_state.allocated_capital,
                "available": capital_state.available_capital,
                "utilization_pct": capital_state.utilization_pct(),
            },
            "exposure": {
                "total_usd": exposure_state.total_exposure_usd,
                "num_positions": exposure_state.num_positions,
                "by_asset": exposure_state.by_asset,
                "by_strategy": exposure_state.by_strategy,
            },
            "checks_enabled": self.enable_checks,
        }

    def reset(self, new_capital: Optional[float] = None) -> None:
        """
        Reset risk manager (useful for testing or strategy reset).

        Args:
            new_capital: New capital amount (uses initial if None)
        """
        self.capital_allocator.reset(new_capital)
        self.exposure_tracker.reset()
        logger.info("Risk manager reset")

    def __repr__(self) -> str:
        capital_state = self.capital_allocator.get_state()
        exposure_state = self.exposure_tracker.get_state()

        return (
            f"RiskManager("
            f"capital=${capital_state.total_capital:,.2f}, "
            f"utilization={capital_state.utilization_pct():.1f}%, "
            f"positions={exposure_state.num_positions})"
        )
