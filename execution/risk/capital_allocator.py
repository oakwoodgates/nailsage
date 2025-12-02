"""Capital allocation and tracking for risk management.

Tracks available vs allocated capital to prevent over-leveraging.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CapitalState:
    """Current capital allocation state.

    Attributes:
        total_capital: Total available capital
        allocated_capital: Capital currently allocated to open positions
        available_capital: Capital available for new positions
        reserved_capital: Capital reserved for margin/fees
    """

    total_capital: float
    allocated_capital: float
    available_capital: float
    reserved_capital: float = 0.0

    def utilization_pct(self) -> float:
        """Get capital utilization percentage."""
        if self.total_capital <= 0:
            return 0.0
        return (self.allocated_capital / self.total_capital) * 100.0


class CapitalAllocator:
    """
    Manages capital allocation for trading strategies.

    Tracks how much capital is allocated to positions vs available
    for new trades, preventing over-leveraging.

    Attributes:
        initial_capital: Starting capital amount
        max_allocation_pct: Maximum % of capital that can be allocated (default: 95%)
        reserve_for_fees_pct: % of capital to reserve for fees/slippage (default: 5%)
    """

    def __init__(
        self,
        initial_capital: float,
        max_allocation_pct: float = 95.0,
        reserve_for_fees_pct: float = 5.0,
    ):
        """
        Initialize capital allocator.

        Args:
            initial_capital: Starting capital amount
            max_allocation_pct: Max % of capital that can be allocated (0-100)
            reserve_for_fees_pct: % to reserve for fees/slippage (0-100)

        Raises:
            ValueError: If parameters are invalid
        """
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be > 0, got {initial_capital}")

        if not 0 < max_allocation_pct <= 100:
            raise ValueError(
                f"Max allocation % must be in (0, 100], got {max_allocation_pct}"
            )

        if not 0 <= reserve_for_fees_pct < 100:
            raise ValueError(
                f"Reserve % must be in [0, 100), got {reserve_for_fees_pct}"
            )

        if max_allocation_pct + reserve_for_fees_pct > 100:
            raise ValueError(
                f"Max allocation ({max_allocation_pct}%) + reserve ({reserve_for_fees_pct}%) "
                f"cannot exceed 100%"
            )

        self.initial_capital = initial_capital
        self.max_allocation_pct = max_allocation_pct
        self.reserve_for_fees_pct = reserve_for_fees_pct

        # Current allocation tracking
        self._allocated_by_position: Dict[int, float] = {}  # position_id -> allocated_amount
        self._current_equity = initial_capital  # Track P&L

        logger.info(
            f"Initialized CapitalAllocator: ${initial_capital:,.2f}, "
            f"max_alloc={max_allocation_pct}%, reserve={reserve_for_fees_pct}%"
        )

    def get_state(self) -> CapitalState:
        """
        Get current capital allocation state.

        Returns:
            CapitalState with allocation details
        """
        allocated = sum(self._allocated_by_position.values())
        reserved = self._current_equity * (self.reserve_for_fees_pct / 100.0)
        available = max(0, self._current_equity - allocated - reserved)

        return CapitalState(
            total_capital=self._current_equity,
            allocated_capital=allocated,
            available_capital=available,
            reserved_capital=reserved,
        )

    def can_allocate(self, amount: float) -> tuple[bool, Optional[str]]:
        """
        Check if capital amount can be allocated.

        Args:
            amount: Capital amount to allocate

        Returns:
            Tuple of (can_allocate, reason_if_not)
        """
        if amount <= 0:
            return False, f"Invalid allocation amount: ${amount:,.2f}"

        state = self.get_state()

        if amount > state.available_capital:
            return False, (
                f"Insufficient capital: need ${amount:,.2f}, "
                f"available ${state.available_capital:,.2f}"
            )

        # Check if allocation would exceed max allocation %
        new_allocated = state.allocated_capital + amount
        max_allowed = self._current_equity * (self.max_allocation_pct / 100.0)

        if new_allocated > max_allowed:
            return False, (
                f"Would exceed max allocation: "
                f"${new_allocated:,.2f} > ${max_allowed:,.2f} "
                f"({self.max_allocation_pct}%)"
            )

        return True, None

    def allocate(self, position_id: int, amount: float) -> None:
        """
        Allocate capital to a position.

        Args:
            position_id: Position identifier
            amount: Amount to allocate

        Raises:
            ValueError: If allocation fails validation
        """
        can_allocate, reason = self.can_allocate(amount)
        if not can_allocate:
            raise ValueError(f"Cannot allocate capital: {reason}")

        if position_id in self._allocated_by_position:
            raise ValueError(
                f"Position {position_id} already has allocated capital: "
                f"${self._allocated_by_position[position_id]:,.2f}"
            )

        self._allocated_by_position[position_id] = amount

        logger.info(
            f"Allocated ${amount:,.2f} to position {position_id}. "
            f"Utilization: {self.get_state().utilization_pct():.1f}%"
        )

    def deallocate(self, position_id: int) -> float:
        """
        Deallocate capital from a closed position.

        Args:
            position_id: Position identifier

        Returns:
            Amount that was deallocated

        Raises:
            ValueError: If position not found
        """
        if position_id not in self._allocated_by_position:
            raise ValueError(f"Position {position_id} has no allocated capital")

        amount = self._allocated_by_position.pop(position_id)

        logger.info(
            f"Deallocated ${amount:,.2f} from position {position_id}. "
            f"Utilization: {self.get_state().utilization_pct():.1f}%"
        )

        return amount

    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity (after P&L changes).

        Args:
            new_equity: New equity value

        Raises:
            ValueError: If new_equity is invalid
        """
        if new_equity < 0:
            raise ValueError(f"Equity cannot be negative: ${new_equity:,.2f}")

        old_equity = self._current_equity
        self._current_equity = new_equity
        pnl = new_equity - old_equity

        logger.info(
            f"Equity updated: ${old_equity:,.2f} → ${new_equity:,.2f} "
            f"(P&L: {pnl:+,.2f})"
        )

    def get_max_position_size(
        self,
        price: float,
        max_position_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate maximum position size that can be opened.

        Args:
            price: Asset price
            max_position_pct: Optional max % of available capital per position

        Returns:
            Maximum position size in USD

        Raises:
            ValueError: If price is invalid
        """
        if price <= 0:
            raise ValueError(f"Price must be > 0, got {price}")

        state = self.get_state()
        available = state.available_capital

        # Apply position size limit if specified
        if max_position_pct:
            if not 0 < max_position_pct <= 100:
                raise ValueError(
                    f"max_position_pct must be in (0, 100], got {max_position_pct}"
                )
            max_from_pct = state.total_capital * (max_position_pct / 100.0)
            available = min(available, max_from_pct)

        return available

    def get_allocated_for_position(self, position_id: int) -> float:
        """
        Get allocated capital for a specific position.

        Args:
            position_id: Position identifier

        Returns:
            Allocated amount (0 if position not found)
        """
        return self._allocated_by_position.get(position_id, 0.0)

    def reset(self, new_capital: Optional[float] = None) -> None:
        """
        Reset allocator (useful for testing or strategy reset).

        Args:
            new_capital: New capital amount (uses initial_capital if None)
        """
        capital = new_capital if new_capital is not None else self.initial_capital
        self._allocated_by_position.clear()
        self._current_equity = capital

        logger.info(f"Capital allocator reset to ${capital:,.2f}")

    def __repr__(self) -> str:
        state = self.get_state()
        return (
            f"CapitalAllocator("
            f"equity=${state.total_capital:,.2f}, "
            f"allocated=${state.allocated_capital:,.2f}, "
            f"available=${state.available_capital:,.2f}, "
            f"utilization={state.utilization_pct():.1f}%)"
        )
