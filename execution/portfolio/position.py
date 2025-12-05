"""Position tracking for active trades."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    """
    Represents an active trading position.

    Tracks the details of an open position including entry price,
    size, and current profit/loss.

    Attributes:
        strategy_name: Name of the strategy that opened this position
        asset: Trading pair (e.g., 'BTC/USDT', 'SOL/USDT')
        direction: Position direction: -1 (short), 1 (long)
        size_usd: Position size in USD
        entry_time: When the position was opened
        entry_price: Price at which position was entered
        current_price: Current market price (updated)
        unrealized_pnl: Current unrealized profit/loss in USD

    Example:
        >>> position = Position(
        ...     strategy_name='momentum_classifier_v1',
        ...     asset='BTC/USDT',
        ...     direction=1,  # long
        ...     size_usd=10000.0,
        ...     entry_time=datetime.now(),
        ...     entry_price=50000.0,
        ...     current_price=51000.0,
        ...     unrealized_pnl=200.0
        ... )
    """

    strategy_name: str
    asset: str
    direction: int  # -1 (short), 1 (long)
    size_usd: float
    entry_time: datetime
    entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None

    def __post_init__(self):
        """Validate position fields."""
        if self.direction not in (-1, 1):
            raise ValueError(f"Direction must be -1 or 1, got {self.direction}")
        if self.size_usd <= 0:
            raise ValueError(f"Position size must be positive, got {self.size_usd}")
        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {self.entry_price}")

    @property
    def direction_name(self) -> str:
        """Human-readable direction name."""
        return "short" if self.direction == -1 else "long"

    def update_pnl(self, current_price: float) -> None:
        """
        Update unrealized PnL based on current price.

        Args:
            current_price: Current market price
        """
        self.current_price = current_price

        # Calculate PnL
        # Long: profit if price goes up
        # Short: profit if price goes down
        price_change = (current_price - self.entry_price) / self.entry_price
        if self.direction == -1:  # short
            price_change = -price_change

        self.unrealized_pnl = self.size_usd * price_change

    def to_dict(self) -> dict:
        """Convert position to dictionary for serialization."""
        return {
            'strategy': self.strategy_name,
            'asset': self.asset,
            'direction': self.direction_name,
            'size_usd': self.size_usd,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
        }

    def __repr__(self) -> str:
        """String representation of position."""
        pnl_str = f"PnL=${self.unrealized_pnl:,.2f}" if self.unrealized_pnl is not None else "PnL=N/A"
        return (
            f"Position({self.strategy_name}, {self.asset}, "
            f"{self.direction_name}, ${self.size_usd:,.0f}, {pnl_str})"
        )
