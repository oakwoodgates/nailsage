"""Position tracking and P&L calculation for paper trading.

This module provides:
- Position management (open, close, update)
- Unrealized P&L calculation (mark-to-market)
- Realized P&L calculation (on position close)
- Position state persistence
- Integration with StateManager

Example usage:
    tracker = PositionTracker(state_manager=state_manager)

    # Open a new position
    position = tracker.open_position(
        strategy_id=1,
        starlisting_id=1,
        side='long',
        size=0.1,
        entry_price=89500.0,
        entry_timestamp=timestamp,
        fees_paid=8.95,
    )

    # Update position with current price
    tracker.update_position_pnl(position.id, current_price=90000.0)

    # Close position
    tracker.close_position(
        position_id=position.id,
        exit_price=90500.0,
        exit_timestamp=timestamp,
        fees_paid=9.05,
        exit_reason='signal',
    )
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from execution.persistence.state_manager import Position, StateManager

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Tracks positions and calculates P&L.

    This class handles:
    - Position lifecycle (open, update, close)
    - Unrealized P&L calculation (mark-to-market)
    - Realized P&L calculation (on close)
    - Position persistence via StateManager
    - Position lookup and querying

    Attributes:
        state_manager: StateManager for persistence
        _open_positions: Cache of currently open positions {position_id: Position}
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize position tracker.

        Args:
            state_manager: StateManager instance for persistence
        """
        self.state_manager = state_manager
        self._open_positions: Dict[int, Position] = {}

        # Load existing open positions from database
        self._load_open_positions()

        logger.info(
            f"Initialized PositionTracker with {len(self._open_positions)} open positions"
        )

    def _load_open_positions(self) -> None:
        """Load open positions from database into memory cache."""
        positions = self.state_manager.get_open_positions()
        for position in positions:
            self._open_positions[position.id] = position

        logger.info(f"Loaded {len(positions)} open positions from database")

    def open_position(
        self,
        strategy_id: int,
        starlisting_id: int,
        side: str,
        size: float,
        entry_price: float,
        entry_timestamp: int,
        fees_paid: float = 0.0,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Position:
        """
        Open a new position.

        Args:
            strategy_id: Strategy ID
            starlisting_id: Starlisting ID
            side: 'long' or 'short'
            size: Position size in base currency
            entry_price: Entry price
            entry_timestamp: Entry timestamp (Unix ms)
            fees_paid: Fees paid on entry
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price

        Returns:
            Position record with assigned ID

        Raises:
            ValueError: If position parameters are invalid
        """
        # Validate inputs
        if side not in ('long', 'short'):
            raise ValueError(f"Invalid side: {side}. Must be 'long' or 'short'")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        # Create position
        position = Position(
            id=None,  # Will be assigned by database
            strategy_id=strategy_id,
            starlisting_id=starlisting_id,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            fees_paid=fees_paid,
            status='open',
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        # Save to database
        position_id = self.state_manager.save_position(position)
        position.id = position_id

        # Add to cache
        self._open_positions[position_id] = position

        notional_usd = size * entry_price
        logger.info(
            f"Opened {side} position #{position_id}",
            extra={
                "strategy_id": strategy_id,
                "starlisting_id": starlisting_id,
                "size": f"{size:.6f}",
                "entry_price": f"${entry_price:,.2f}",
                "notional_usd": f"${notional_usd:,.2f}",
            }
        )

        return position

    def update_position_pnl(
        self,
        position_id: int,
        current_price: float,
    ) -> Position:
        """
        Update position's unrealized P&L based on current price.

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            Updated Position

        Raises:
            ValueError: If position not found or not open
        """
        if position_id not in self._open_positions:
            raise ValueError(f"Position {position_id} not found or not open")

        position = self._open_positions[position_id]

        # Calculate unrealized P&L
        unrealized_pnl = self._calculate_unrealized_pnl(
            position.side,
            position.size,
            position.entry_price,
            current_price,
        )

        # Update position
        position.unrealized_pnl = unrealized_pnl
        position.updated_at = int(datetime.now().timestamp() * 1000)

        # Save to database
        self.state_manager.save_position(position)

        logger.debug(
            f"Updated position #{position_id} P&L: ${unrealized_pnl:,.2f}",
            extra={
                "entry_price": position.entry_price,
                "current_price": current_price,
                "pnl": unrealized_pnl,
            }
        )

        return position

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_timestamp: int,
        fees_paid: float = 0.0,
        exit_reason: Optional[str] = None,
    ) -> Position:
        """
        Close an open position.

        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_timestamp: Exit timestamp (Unix ms)
            fees_paid: Fees paid on exit
            exit_reason: Reason for closing (e.g., 'signal', 'stop_loss')

        Returns:
            Closed Position

        Raises:
            ValueError: If position not found or already closed
        """
        if position_id not in self._open_positions:
            raise ValueError(f"Position {position_id} not found or not open")

        position = self._open_positions[position_id]

        # Calculate realized P&L
        realized_pnl = self._calculate_realized_pnl(
            position.side,
            position.size,
            position.entry_price,
            exit_price,
            position.fees_paid + fees_paid,
        )

        # Update position
        position.exit_price = exit_price
        position.exit_timestamp = exit_timestamp
        position.realized_pnl = realized_pnl
        position.unrealized_pnl = 0.0  # No longer unrealized
        position.fees_paid += fees_paid
        position.status = 'closed'
        position.exit_reason = exit_reason
        position.updated_at = int(datetime.now().timestamp() * 1000)

        # Save to database
        self.state_manager.save_position(position)

        # Remove from cache
        del self._open_positions[position_id]

        logger.info(
            f"Closed {position.side} position #{position_id}",
            extra={
                "entry_price": f"${position.entry_price:,.2f}",
                "exit_price": f"${exit_price:,.2f}",
                "realized_pnl": f"${realized_pnl:,.2f}",
                "exit_reason": exit_reason,
            }
        )

        return position

    def _calculate_unrealized_pnl(
        self,
        side: str,
        size: float,
        entry_price: float,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized P&L for an open position.

        Args:
            side: 'long' or 'short'
            size: Position size in base currency
            entry_price: Entry price
            current_price: Current market price

        Returns:
            Unrealized P&L in USD (positive = profit, negative = loss)
        """
        if side == 'long':
            # Long: profit when price goes up
            pnl = (current_price - entry_price) * size
        else:  # short
            # Short: profit when price goes down
            pnl = (entry_price - current_price) * size

        return pnl

    def _calculate_realized_pnl(
        self,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float,
        total_fees: float,
    ) -> float:
        """
        Calculate realized P&L for a closed position.

        Args:
            side: 'long' or 'short'
            size: Position size in base currency
            entry_price: Entry price
            exit_price: Exit price
            total_fees: Total fees paid (entry + exit)

        Returns:
            Realized P&L in USD (after fees)
        """
        if side == 'long':
            # Long: profit when price goes up
            gross_pnl = (exit_price - entry_price) * size
        else:  # short
            # Short: profit when price goes down
            gross_pnl = (entry_price - exit_price) * size

        # Subtract fees
        net_pnl = gross_pnl - total_fees

        return net_pnl

    def get_position(self, position_id: int) -> Optional[Position]:
        """
        Get position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position if found, None otherwise
        """
        if position_id in self._open_positions:
            return self._open_positions[position_id]

        # Check database for closed positions
        return self.state_manager.get_position(position_id)

    def get_open_positions(
        self,
        strategy_id: Optional[int] = None,
        starlisting_id: Optional[int] = None,
    ) -> List[Position]:
        """
        Get all open positions, optionally filtered.

        Args:
            strategy_id: Filter by strategy ID
            starlisting_id: Filter by starlisting ID

        Returns:
            List of open positions
        """
        positions = list(self._open_positions.values())

        if strategy_id is not None:
            positions = [p for p in positions if p.strategy_id == strategy_id]

        if starlisting_id is not None:
            positions = [p for p in positions if p.starlisting_id == starlisting_id]

        return positions

    def get_total_unrealized_pnl(self, strategy_id: Optional[int] = None) -> float:
        """
        Get total unrealized P&L across open positions.

        Args:
            strategy_id: Optional strategy ID to filter positions

        Returns:
            Total unrealized P&L in USD
        """
        total = sum(
            p.unrealized_pnl or 0.0
            for p in self._open_positions.values()
            if strategy_id is None or p.strategy_id == strategy_id
        )
        return total

    def get_total_exposure_usd(self, strategy_id: Optional[int] = None) -> float:
        """
        Get total exposure (notional value) across open positions.

        Args:
            strategy_id: Optional strategy ID to filter positions

        Returns:
            Total exposure in USD
        """
        total = 0.0
        for position in self._open_positions.values():
            if strategy_id is not None and position.strategy_id != strategy_id:
                continue
            # For both longs and shorts, exposure is size * entry_price
            notional = position.size * position.entry_price
            total += notional

        return total

    def has_open_position(
        self,
        strategy_id: int,
        starlisting_id: int,
    ) -> bool:
        """
        Check if strategy has open position for a starlisting.

        Args:
            strategy_id: Strategy ID
            starlisting_id: Starlisting ID

        Returns:
            True if open position exists
        """
        for position in self._open_positions.values():
            if (position.strategy_id == strategy_id and
                position.starlisting_id == starlisting_id):
                return True
        return False

    def get_stats(self, strategy_id: Optional[int] = None) -> dict:
        """
        Get statistics about tracked positions.

        Args:
            strategy_id: Optional strategy ID to filter stats

        Returns:
            Dict with statistics including P&L and win rate
        """
        # Get realized P&L and closed position stats from database
        total_realized_pnl = self.state_manager.get_total_realized_pnl(strategy_id)
        closed_stats = self.state_manager.get_closed_positions_stats(strategy_id)

        # Calculate win rate
        total_closed = closed_stats['total_closed']
        num_wins = closed_stats['num_wins']
        num_losses = closed_stats['num_losses']
        win_rate = (num_wins / total_closed * 100) if total_closed > 0 else 0.0

        # Filter open positions by strategy_id if specified
        open_positions = [
            p for p in self._open_positions.values()
            if strategy_id is None or p.strategy_id == strategy_id
        ]

        return {
            "num_open_positions": len(open_positions),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(strategy_id),
            "total_exposure_usd": self.get_total_exposure_usd(strategy_id),
            "total_realized_pnl": total_realized_pnl,
            "num_wins": num_wins,
            "num_losses": num_losses,
            "total_closed": total_closed,
            "win_rate": win_rate,
        }
