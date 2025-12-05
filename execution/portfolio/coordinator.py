"""Portfolio coordination and position management.

Phase 1 MVP: Simple pass-through coordinator with basic safety checks.
This coordinator tracks positions and enforces risk limits without
attempting to optimize or coordinate between strategies.
"""

from datetime import datetime
from typing import Dict, List, Optional

from execution.portfolio.position import Position
from execution.portfolio.signal import StrategySignal
from utils.logger import get_logger

logger = get_logger("portfolio.coordinator")


class PortfolioCoordinator:
    """
    Portfolio coordinator for managing multiple strategy positions.

    Phase 1 MVP Design:
    - Pass-through approach: Each strategy gets its own positions
    - Basic safety checks: Max positions, max total exposure
    - No optimization or signal coordination
    - Simple position tracking and reporting

    Future phases will add:
    - Portfolio-level optimization
    - Signal coordination between strategies
    - Dynamic position sizing
    - Risk-adjusted allocations

    Attributes:
        max_positions: Maximum number of concurrent positions across all strategies
        max_total_exposure: Maximum total USD exposure across all positions
        positions: Dictionary mapping (strategy_name, asset) to Position objects

    Example:
        >>> coordinator = PortfolioCoordinator(
        ...     max_positions=10,
        ...     max_total_exposure=100000.0
        ... )
        >>> signals = [signal1, signal2]
        >>> approved_signals = coordinator.process_signals(signals)
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_total_exposure: float = 100000.0
    ):
        """
        Initialize portfolio coordinator.

        Args:
            max_positions: Maximum concurrent positions allowed
            max_total_exposure: Maximum total USD exposure allowed
        """
        if max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        if max_total_exposure <= 0:
            raise ValueError(f"max_total_exposure must be positive, got {max_total_exposure}")

        self.max_positions = max_positions
        self.max_total_exposure = max_total_exposure

        # Key: (strategy_name, asset), Value: Position
        self.positions: Dict[tuple[str, str], Position] = {}

        logger.info(
            f"Portfolio Coordinator initialized: "
            f"max_positions={max_positions}, "
            f"max_total_exposure=${max_total_exposure:,.0f}"
        )

    def process_signals(
        self,
        signals: List[StrategySignal]
    ) -> List[StrategySignal]:
        """
        Process trading signals with basic safety checks.

        Phase 1: Pass-through with limits
        - Allow signals if within position and exposure limits
        - Each strategy manages its own positions independently
        - No coordination or optimization between strategies

        Args:
            signals: List of strategy signals to process

        Returns:
            List of approved signals (subset of input signals)
        """
        approved_signals = []
        current_exposure = self.get_total_exposure()
        current_positions = len(self.positions)

        logger.info(
            f"Processing {len(signals)} signals. "
            f"Current: {current_positions}/{self.max_positions} positions, "
            f"${current_exposure:,.0f}/${self.max_total_exposure:,.0f} exposure"
        )

        for signal in signals:
            # Check if this is a close signal (always allow)
            if signal.signal == 0:
                approved_signals.append(signal)
                logger.debug(f"Approved close signal: {signal.strategy_name} {signal.asset}")
                continue

            # Check if we already have a position for this (strategy, asset)
            position_key = (signal.strategy_name, signal.asset)
            has_position = position_key in self.positions

            # If opening a new position, check limits
            if not has_position:
                # Check position count limit
                if current_positions >= self.max_positions:
                    logger.warning(
                        f"Rejected signal {signal.strategy_name} {signal.asset}: "
                        f"Max positions limit reached ({self.max_positions})"
                    )
                    continue

                # Check exposure limit
                new_exposure = current_exposure + signal.position_size_usd
                if new_exposure > self.max_total_exposure:
                    logger.warning(
                        f"Rejected signal {signal.strategy_name} {signal.asset}: "
                        f"Would exceed max exposure "
                        f"(${new_exposure:,.0f} > ${self.max_total_exposure:,.0f})"
                    )
                    continue

                # Signal approved - will open new position
                current_positions += 1
                current_exposure += signal.position_size_usd

            # Signal approved
            approved_signals.append(signal)
            logger.info(
                f"Approved signal: {signal.strategy_name} {signal.asset} "
                f"{signal.direction_name} ${signal.position_size_usd:,.0f}"
            )

        logger.info(f"Approved {len(approved_signals)}/{len(signals)} signals")
        return approved_signals

    def update_position(
        self,
        strategy_name: str,
        asset: str,
        position: Optional[Position]
    ) -> None:
        """
        Update or close a position.

        Args:
            strategy_name: Name of the strategy
            asset: Trading pair
            position: New position object, or None to close position
        """
        position_key = (strategy_name, asset)

        if position is None:
            # Close position
            if position_key in self.positions:
                old_position = self.positions.pop(position_key)
                logger.info(
                    f"Closed position: {strategy_name} {asset} "
                    f"({old_position.direction_name}, "
                    f"PnL=${old_position.unrealized_pnl or 0:,.2f})"
                )
            else:
                logger.warning(
                    f"Attempted to close non-existent position: {strategy_name} {asset}"
                )
        else:
            # Open or update position
            action = "Updated" if position_key in self.positions else "Opened"
            self.positions[position_key] = position
            logger.info(
                f"{action} position: {strategy_name} {asset} "
                f"{position.direction_name} ${position.size_usd:,.0f} "
                f"@ ${position.entry_price:,.2f}"
            )

    def get_total_exposure(self) -> float:
        """
        Calculate total USD exposure across all positions.

        Returns:
            Total position size in USD
        """
        return sum(pos.size_usd for pos in self.positions.values())

    def get_positions_by_strategy(self, strategy_name: str) -> List[Position]:
        """
        Get all positions for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            List of positions for the strategy
        """
        return [
            pos for (strat, _), pos in self.positions.items()
            if strat == strategy_name
        ]

    def get_positions_by_asset(self, asset: str) -> List[Position]:
        """
        Get all positions for a specific asset across all strategies.

        Args:
            asset: Trading pair (e.g., 'BTC/USDT')

        Returns:
            List of positions for the asset
        """
        return [
            pos for (_, pos_asset), pos in self.positions.items()
            if pos_asset == asset
        ]

    def get_snapshot(self) -> dict:
        """
        Get current portfolio state snapshot.

        Returns:
            Dictionary containing portfolio metrics and positions
        """
        total_exposure = self.get_total_exposure()
        total_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
            if pos.unrealized_pnl is not None
        )

        # Count positions by strategy
        strategy_counts = {}
        for (strategy_name, _), _ in self.positions.items():
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1

        return {
            'timestamp': datetime.now().isoformat(),
            'total_positions': len(self.positions),
            'max_positions': self.max_positions,
            'total_exposure_usd': total_exposure,
            'max_exposure_usd': self.max_total_exposure,
            'total_unrealized_pnl': total_pnl,
            'utilization_pct': (len(self.positions) / self.max_positions) * 100,
            'exposure_pct': (total_exposure / self.max_total_exposure) * 100,
            'positions_by_strategy': strategy_counts,
            'positions': [
                {
                    **pos.to_dict(),
                    'key': f"{strategy}_{asset}",
                }
                for (strategy, asset), pos in self.positions.items()
            ]
        }

    def __repr__(self) -> str:
        """String representation of coordinator."""
        total_exposure = self.get_total_exposure()
        return (
            f"PortfolioCoordinator("
            f"{len(self.positions)}/{self.max_positions} positions, "
            f"${total_exposure:,.0f}/${self.max_total_exposure:,.0f} exposure)"
        )
