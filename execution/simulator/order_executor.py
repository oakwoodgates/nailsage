"""Order execution simulator for paper trading.

This module simulates realistic order execution with:
- Slippage modeling based on order size
- Transaction fees (maker/taker fees)
- Market order simulation (filled at current price + slippage)
- Fill price calculation
- Trade record creation

Example usage:
    executor = OrderExecutor(
        fee_rate=0.001,  # 0.1% taker fee
        slippage_bps=5,  # 5 basis points
    )

    # Execute a buy order
    trade = executor.execute_market_order(
        side='long',
        size_usd=10000.0,
        current_price=89500.0,
        timestamp=int(datetime.now().timestamp() * 1000),
        position_id=1,
        strategy_id=1,
        starlisting_id=1,
    )
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from execution.persistence.state_manager import Trade

logger = logging.getLogger(__name__)


@dataclass
class OrderExecutorConfig:
    """Configuration for order execution simulation.

    Attributes:
        fee_rate: Transaction fee rate (e.g., 0.001 = 0.1%)
        slippage_bps: Slippage in basis points (e.g., 5 = 0.05%)
        min_order_size_usd: Minimum order size in USD
        max_order_size_usd: Maximum order size in USD (for safety)
    """

    fee_rate: float = 0.001  # 0.1% taker fee (typical for crypto exchanges)
    slippage_bps: float = 5.0  # 5 basis points (0.05%)
    min_order_size_usd: float = 10.0
    max_order_size_usd: float = 1_000_000.0

    def __post_init__(self):
        """Validate config."""
        if self.fee_rate < 0 or self.fee_rate > 0.1:
            raise ValueError(
                f"fee_rate must be between 0 and 0.1, got {self.fee_rate}"
            )
        if self.slippage_bps < 0:
            raise ValueError(
                f"slippage_bps must be non-negative, got {self.slippage_bps}"
            )
        if self.min_order_size_usd <= 0:
            raise ValueError(
                f"min_order_size_usd must be positive, got {self.min_order_size_usd}"
            )


@dataclass
class OrderResult:
    """Result of order execution.

    Attributes:
        trade: Trade record
        fill_price: Actual fill price (including slippage)
        size: Position size in base currency
        notional_usd: Total value in USD
        fees_usd: Fees paid in USD
        slippage_usd: Slippage cost in USD
        success: Whether order was successfully executed
        rejection_reason: Reason if order was rejected
    """

    trade: Optional[Trade]
    fill_price: float
    size: float
    notional_usd: float
    fees_usd: float
    slippage_usd: float
    success: bool
    rejection_reason: Optional[str] = None


class OrderExecutor:
    """
    Simulates order execution for paper trading.

    This class provides realistic simulation of:
    - Market orders (immediate execution at current price)
    - Slippage (based on order size and market conditions)
    - Transaction fees (proportional to order value)
    - Fill price calculation

    Attributes:
        config: OrderExecutorConfig
        _orders_executed: Total count of executed orders
    """

    def __init__(self, config: OrderExecutorConfig):
        """
        Initialize order executor.

        Args:
            config: OrderExecutorConfig instance
        """
        self.config = config
        self._orders_executed: int = 0

        logger.info(
            "Initialized OrderExecutor",
            extra={
                "fee_rate": config.fee_rate,
                "slippage_bps": config.slippage_bps,
            }
        )

    def execute_market_order(
        self,
        side: Literal['long', 'short'],
        size_usd: float,
        current_price: float,
        timestamp: int,
        position_id: int,
        strategy_id: int,
        starlisting_id: int,
        trade_type: Literal[
            'open_long', 'open_short', 'close_long', 'close_short'
        ],
        signal_id: Optional[int] = None,
    ) -> OrderResult:
        """
        Execute a market order (immediate fill).

        Args:
            side: 'long' (buy) or 'short' (sell)
            size_usd: Order size in USD
            current_price: Current market price
            timestamp: Execution timestamp (Unix ms)
            position_id: Associated position ID
            strategy_id: Strategy ID
            starlisting_id: Starlisting ID
            trade_type: Type of trade
            signal_id: Optional signal ID that triggered this trade

        Returns:
            OrderResult with trade details or rejection reason
        """
        # Validate order size
        if size_usd < self.config.min_order_size_usd:
            return OrderResult(
                trade=None,
                fill_price=current_price,
                size=0.0,
                notional_usd=0.0,
                fees_usd=0.0,
                slippage_usd=0.0,
                success=False,
                rejection_reason=(
                    f"Order size ${size_usd:.2f} below minimum "
                    f"${self.config.min_order_size_usd:.2f}"
                )
            )

        if size_usd > self.config.max_order_size_usd:
            return OrderResult(
                trade=None,
                fill_price=current_price,
                size=0.0,
                notional_usd=0.0,
                fees_usd=0.0,
                slippage_usd=0.0,
                success=False,
                rejection_reason=(
                    f"Order size ${size_usd:.2f} exceeds maximum "
                    f"${self.config.max_order_size_usd:.2f}"
                )
            )

        # Calculate slippage
        # For longs (buys), slippage increases price
        # For shorts (sells), slippage decreases price
        slippage_multiplier = self.config.slippage_bps / 10000.0  # Convert bps to decimal

        if side == 'long':
            fill_price = current_price * (1 + slippage_multiplier)
        else:  # short
            fill_price = current_price * (1 - slippage_multiplier)

        # Calculate position size in base currency
        size = size_usd / fill_price

        # Calculate fees (on notional value)
        fees = size_usd * self.config.fee_rate

        # Calculate slippage cost
        slippage_cost = abs(fill_price - current_price) * size

        # Create trade record
        trade = Trade(
            id=None,  # Will be assigned by database
            position_id=position_id,
            strategy_id=strategy_id,
            starlisting_id=starlisting_id,
            trade_type=trade_type,
            size=size,
            price=fill_price,
            fees=fees,
            slippage=slippage_cost,
            timestamp=timestamp,
            signal_id=signal_id,
            created_at=None,  # Will be set by database
        )

        self._orders_executed += 1

        logger.info(
            f"Executed {trade_type} order #{self._orders_executed}",
            extra={
                "side": side,
                "size": f"{size:.6f}",
                "fill_price": f"${fill_price:,.2f}",
                "notional_usd": f"${size_usd:,.2f}",
                "fees": f"${fees:.2f}",
                "slippage": f"${slippage_cost:.2f}",
            }
        )

        return OrderResult(
            trade=trade,
            fill_price=fill_price,
            size=size,
            notional_usd=size_usd,
            fees_usd=fees,
            slippage_usd=slippage_cost,
            success=True,
            rejection_reason=None,
        )

    def calculate_fill_price(
        self,
        side: Literal['long', 'short'],
        current_price: float,
    ) -> float:
        """
        Calculate expected fill price for an order.

        Args:
            side: 'long' (buy) or 'short' (sell)
            current_price: Current market price

        Returns:
            Expected fill price including slippage
        """
        slippage_multiplier = self.config.slippage_bps / 10000.0

        if side == 'long':
            return current_price * (1 + slippage_multiplier)
        else:  # short
            return current_price * (1 - slippage_multiplier)

    def calculate_fees(self, notional_usd: float) -> float:
        """
        Calculate fees for an order.

        Args:
            notional_usd: Order value in USD

        Returns:
            Fees in USD
        """
        return notional_usd * self.config.fee_rate

    def get_stats(self) -> dict:
        """
        Get statistics about order execution.

        Returns:
            Dict with statistics
        """
        return {
            "orders_executed": self._orders_executed,
            "fee_rate": self.config.fee_rate,
            "slippage_bps": self.config.slippage_bps,
        }
