"""Exposure tracking for risk management.

Tracks exposure across assets and strategies to prevent over-concentration.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExposureState:
    """Current exposure state.

    Attributes:
        total_exposure_usd: Total exposure across all positions
        by_asset: Exposure by asset (e.g., 'BTC/USDT' -> $10,000)
        by_strategy: Exposure by strategy name
        num_positions: Total number of open positions
    """

    total_exposure_usd: float
    by_asset: Dict[str, float]
    by_strategy: Dict[str, int]  # Strategy -> position count
    num_positions: int


class ExposureTracker:
    """
    Tracks trading exposure across assets and strategies.

    Prevents over-concentration in single assets or strategies.

    Attributes:
        max_positions: Maximum total open positions (0 = unlimited)
        max_positions_per_asset: Max positions per asset (0 = unlimited)
        max_positions_per_strategy: Max positions per strategy (0 = unlimited)
        max_exposure_per_asset_pct: Max % of capital exposed to one asset
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_positions_per_asset: int = 3,
        max_positions_per_strategy: int = 5,
        max_exposure_per_asset_pct: float = 25.0,
    ):
        """
        Initialize exposure tracker.

        Args:
            max_positions: Max total open positions (0 = unlimited)
            max_positions_per_asset: Max positions per asset (0 = unlimited)
            max_positions_per_strategy: Max positions per strategy (0 = unlimited)
            max_exposure_per_asset_pct: Max % of capital per asset (0-100)

        Raises:
            ValueError: If parameters are invalid
        """
        if max_positions < 0:
            raise ValueError(f"max_positions must be >= 0, got {max_positions}")

        if max_positions_per_asset < 0:
            raise ValueError(
                f"max_positions_per_asset must be >= 0, got {max_positions_per_asset}"
            )

        if max_positions_per_strategy < 0:
            raise ValueError(
                f"max_positions_per_strategy must be >= 0, got {max_positions_per_strategy}"
            )

        if not 0 <= max_exposure_per_asset_pct <= 100:
            raise ValueError(
                f"max_exposure_per_asset_pct must be in [0, 100], "
                f"got {max_exposure_per_asset_pct}"
            )

        self.max_positions = max_positions
        self.max_positions_per_asset = max_positions_per_asset
        self.max_positions_per_strategy = max_positions_per_strategy
        self.max_exposure_per_asset_pct = max_exposure_per_asset_pct

        # Tracking dictionaries
        self._positions: Dict[int, dict] = {}  # position_id -> {asset, strategy, exposure_usd}
        self._by_asset: Dict[str, List[int]] = defaultdict(list)  # asset -> [position_ids]
        self._by_strategy: Dict[str, List[int]] = defaultdict(list)  # strategy -> [position_ids]

        logger.info(
            f"Initialized ExposureTracker: "
            f"max_pos={max_positions}, "
            f"max_per_asset={max_positions_per_asset}, "
            f"max_per_strategy={max_positions_per_strategy}"
        )

    def get_state(self) -> ExposureState:
        """
        Get current exposure state.

        Returns:
            ExposureState with exposure details
        """
        total_exposure = sum(pos['exposure_usd'] for pos in self._positions.values())

        exposure_by_asset = {}
        for asset, position_ids in self._by_asset.items():
            exposure_by_asset[asset] = sum(
                self._positions[pid]['exposure_usd'] for pid in position_ids
            )

        positions_by_strategy = {
            strategy: len(position_ids)
            for strategy, position_ids in self._by_strategy.items()
        }

        return ExposureState(
            total_exposure_usd=total_exposure,
            by_asset=exposure_by_asset,
            by_strategy=positions_by_strategy,
            num_positions=len(self._positions),
        )

    def can_add_position(
        self,
        asset: str,
        strategy: str,
        exposure_usd: float,
        total_capital: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a new position can be added.

        Args:
            asset: Asset symbol (e.g., 'BTC/USDT')
            strategy: Strategy name
            exposure_usd: Position exposure in USD
            total_capital: Total capital for % limits (optional)

        Returns:
            Tuple of (can_add, reason_if_not)
        """
        if exposure_usd <= 0:
            return False, f"Invalid exposure: ${exposure_usd:,.2f}"

        # Check max total positions
        if self.max_positions > 0 and len(self._positions) >= self.max_positions:
            return False, (
                f"Max total positions reached: "
                f"{len(self._positions)}/{self.max_positions}"
            )

        # Check max positions per asset
        if self.max_positions_per_asset > 0:
            current_asset_positions = len(self._by_asset.get(asset, []))
            if current_asset_positions >= self.max_positions_per_asset:
                return False, (
                    f"Max positions for {asset} reached: "
                    f"{current_asset_positions}/{self.max_positions_per_asset}"
                )

        # Check max positions per strategy
        if self.max_positions_per_strategy > 0:
            current_strategy_positions = len(self._by_strategy.get(strategy, []))
            if current_strategy_positions >= self.max_positions_per_strategy:
                return False, (
                    f"Max positions for strategy '{strategy}' reached: "
                    f"{current_strategy_positions}/{self.max_positions_per_strategy}"
                )

        # Check max exposure per asset %
        if total_capital and self.max_exposure_per_asset_pct > 0:
            current_asset_exposure = sum(
                self._positions[pid]['exposure_usd']
                for pid in self._by_asset.get(asset, [])
            )
            new_asset_exposure = current_asset_exposure + exposure_usd
            max_allowed = total_capital * (self.max_exposure_per_asset_pct / 100.0)

            if new_asset_exposure > max_allowed:
                return False, (
                    f"Would exceed max exposure for {asset}: "
                    f"${new_asset_exposure:,.2f} > ${max_allowed:,.2f} "
                    f"({self.max_exposure_per_asset_pct}%)"
                )

        return True, None

    def add_position(
        self,
        position_id: int,
        asset: str,
        strategy: str,
        exposure_usd: float,
    ) -> None:
        """
        Add a new position to tracking.

        Args:
            position_id: Position identifier
            asset: Asset symbol
            strategy: Strategy name
            exposure_usd: Position exposure in USD

        Raises:
            ValueError: If position cannot be added
        """
        if position_id in self._positions:
            raise ValueError(f"Position {position_id} is already being tracked")

        # Note: can_add_position validation should be done before calling this
        # (this method doesn't have total_capital for % validation)

        self._positions[position_id] = {
            'asset': asset,
            'strategy': strategy,
            'exposure_usd': exposure_usd,
        }
        self._by_asset[asset].append(position_id)
        self._by_strategy[strategy].append(position_id)

        logger.info(
            f"Added position {position_id}: {asset} via {strategy}, "
            f"exposure=${exposure_usd:,.2f}"
        )

    def remove_position(self, position_id: int) -> None:
        """
        Remove a closed position from tracking.

        Args:
            position_id: Position identifier

        Raises:
            ValueError: If position not found
        """
        if position_id not in self._positions:
            raise ValueError(f"Position {position_id} is not being tracked")

        position = self._positions.pop(position_id)
        asset = position['asset']
        strategy = position['strategy']

        self._by_asset[asset].remove(position_id)
        if not self._by_asset[asset]:
            del self._by_asset[asset]

        self._by_strategy[strategy].remove(position_id)
        if not self._by_strategy[strategy]:
            del self._by_strategy[strategy]

        logger.info(
            f"Removed position {position_id}: {asset} via {strategy}"
        )

    def update_exposure(self, position_id: int, new_exposure_usd: float) -> None:
        """
        Update exposure for an existing position.

        Args:
            position_id: Position identifier
            new_exposure_usd: New exposure amount

        Raises:
            ValueError: If position not found or exposure invalid
        """
        if position_id not in self._positions:
            raise ValueError(f"Position {position_id} is not being tracked")

        if new_exposure_usd < 0:
            raise ValueError(f"Exposure cannot be negative: ${new_exposure_usd:,.2f}")

        old_exposure = self._positions[position_id]['exposure_usd']
        self._positions[position_id]['exposure_usd'] = new_exposure_usd

        logger.debug(
            f"Updated exposure for position {position_id}: "
            f"${old_exposure:,.2f} → ${new_exposure_usd:,.2f}"
        )

    def get_asset_exposure(self, asset: str) -> float:
        """
        Get total exposure for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Total exposure in USD
        """
        position_ids = self._by_asset.get(asset, [])
        return sum(
            self._positions[pid]['exposure_usd']
            for pid in position_ids
        )

    def get_strategy_count(self, strategy: str) -> int:
        """
        Get number of positions for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Number of positions
        """
        return len(self._by_strategy.get(strategy, []))

    def get_positions_for_asset(self, asset: str) -> List[int]:
        """
        Get all position IDs for an asset.

        Args:
            asset: Asset symbol

        Returns:
            List of position IDs
        """
        return self._by_asset.get(asset, []).copy()

    def get_positions_for_strategy(self, strategy: str) -> List[int]:
        """
        Get all position IDs for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            List of position IDs
        """
        return self._by_strategy.get(strategy, []).copy()

    def reset(self) -> None:
        """Reset tracker (useful for testing or strategy reset)."""
        self._positions.clear()
        self._by_asset.clear()
        self._by_strategy.clear()
        logger.info("Exposure tracker reset")

    def __repr__(self) -> str:
        state = self.get_state()
        return (
            f"ExposureTracker("
            f"positions={state.num_positions}, "
            f"total_exposure=${state.total_exposure_usd:,.2f}, "
            f"assets={len(state.by_asset)})"
        )
