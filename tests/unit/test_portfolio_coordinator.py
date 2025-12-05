"""Unit tests for Portfolio Coordinator."""

import pytest
from datetime import datetime

from execution.portfolio.coordinator import PortfolioCoordinator
from execution.portfolio.position import Position
from execution.portfolio.signal import StrategySignal


class TestPortfolioCoordinator:
    """Test suite for PortfolioCoordinator class."""

    def test_init_valid(self):
        """Test coordinator initialization with valid parameters."""
        coordinator = PortfolioCoordinator(
            max_positions=5,
            max_total_exposure=50000.0
        )
        assert coordinator.max_positions == 5
        assert coordinator.max_total_exposure == 50000.0
        assert len(coordinator.positions) == 0

    def test_init_invalid_max_positions(self):
        """Test coordinator initialization with invalid max_positions."""
        with pytest.raises(ValueError, match="max_positions must be positive"):
            PortfolioCoordinator(max_positions=0, max_total_exposure=10000.0)

        with pytest.raises(ValueError, match="max_positions must be positive"):
            PortfolioCoordinator(max_positions=-5, max_total_exposure=10000.0)

    def test_init_invalid_max_exposure(self):
        """Test coordinator initialization with invalid max_total_exposure."""
        with pytest.raises(ValueError, match="max_total_exposure must be positive"):
            PortfolioCoordinator(max_positions=10, max_total_exposure=0.0)

        with pytest.raises(ValueError, match="max_total_exposure must be positive"):
            PortfolioCoordinator(max_positions=10, max_total_exposure=-1000.0)

    def test_process_signals_empty(self):
        """Test processing empty signal list."""
        coordinator = PortfolioCoordinator()
        approved = coordinator.process_signals([])
        assert len(approved) == 0

    def test_process_signals_single_new_position(self):
        """Test processing a single signal that opens a new position."""
        coordinator = PortfolioCoordinator(
            max_positions=5,
            max_total_exposure=100000.0
        )

        signal = StrategySignal(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            signal=1,  # long
            confidence=0.85,
            timestamp=datetime.now(),
            position_size_usd=10000.0
        )

        approved = coordinator.process_signals([signal])
        assert len(approved) == 1
        assert approved[0] == signal

    def test_process_signals_close_always_approved(self):
        """Test that close signals (signal=0) are always approved."""
        # Create coordinator at max capacity
        coordinator = PortfolioCoordinator(
            max_positions=1,
            max_total_exposure=1000.0
        )

        # Add a position to fill capacity
        position = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=1,
            size_usd=1000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        coordinator.update_position("momentum_v1", "BTC/USDT", position)

        # Now we're at max capacity, but close signals should still be approved
        close_signal = StrategySignal(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            signal=0,  # close
            confidence=1.0,
            timestamp=datetime.now(),
            position_size_usd=0.0
        )

        approved = coordinator.process_signals([close_signal])
        assert len(approved) == 1
        assert approved[0] == close_signal

    def test_process_signals_max_positions_limit(self):
        """Test that signals are rejected when max_positions limit is reached."""
        coordinator = PortfolioCoordinator(
            max_positions=2,
            max_total_exposure=100000.0
        )

        signals = [
            StrategySignal(
                strategy_name=f"strategy_{i}",
                asset=f"ASSET{i}/USDT",
                signal=1,
                confidence=0.8,
                timestamp=datetime.now(),
                position_size_usd=5000.0
            )
            for i in range(3)
        ]

        approved = coordinator.process_signals(signals)
        # Only first 2 should be approved
        assert len(approved) == 2
        assert approved[0] == signals[0]
        assert approved[1] == signals[1]

    def test_process_signals_max_exposure_limit(self):
        """Test that signals are rejected when max_total_exposure limit is reached."""
        coordinator = PortfolioCoordinator(
            max_positions=10,
            max_total_exposure=20000.0
        )

        signals = [
            StrategySignal(
                strategy_name=f"strategy_{i}",
                asset=f"ASSET{i}/USDT",
                signal=1,
                confidence=0.8,
                timestamp=datetime.now(),
                position_size_usd=12000.0  # Each signal requests 12k
            )
            for i in range(3)
        ]

        approved = coordinator.process_signals(signals)
        # Only first signal should be approved (12k < 20k limit)
        # Second signal would bring total to 24k (exceeds 20k limit)
        assert len(approved) == 1
        assert approved[0] == signals[0]

    def test_process_signals_existing_position_allowed(self):
        """Test that signals for existing positions bypass new position limits."""
        coordinator = PortfolioCoordinator(
            max_positions=1,  # Only 1 position allowed
            max_total_exposure=10000.0
        )

        # First signal - opens position
        signal1 = StrategySignal(
            strategy_name="strategy_1",
            asset="BTC/USDT",
            signal=1,
            confidence=0.8,
            timestamp=datetime.now(),
            position_size_usd=5000.0
        )

        approved1 = coordinator.process_signals([signal1])
        assert len(approved1) == 1

        # Manually add position to simulate existing position
        coordinator.update_position(
            strategy_name="strategy_1",
            asset="BTC/USDT",
            position=Position(
                strategy_name="strategy_1",
                asset="BTC/USDT",
                direction=1,
                size_usd=5000.0,
                entry_time=datetime.now(),
                entry_price=50000.0
            )
        )

        # Second signal - updates existing position (should be allowed)
        signal2 = StrategySignal(
            strategy_name="strategy_1",
            asset="BTC/USDT",
            signal=-1,  # Reverse to short
            confidence=0.9,
            timestamp=datetime.now(),
            position_size_usd=5000.0
        )

        # Third signal - new position (should be rejected due to max_positions=1)
        signal3 = StrategySignal(
            strategy_name="strategy_2",
            asset="ETH/USDT",
            signal=1,
            confidence=0.8,
            timestamp=datetime.now(),
            position_size_usd=3000.0
        )

        approved2 = coordinator.process_signals([signal2, signal3])
        # Signal 2 should be approved (updates existing), signal 3 rejected (new position)
        assert len(approved2) == 1
        assert approved2[0] == signal2

    def test_update_position_open(self):
        """Test opening a new position."""
        coordinator = PortfolioCoordinator()

        position = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )

        coordinator.update_position("momentum_v1", "BTC/USDT", position)

        assert len(coordinator.positions) == 1
        assert ("momentum_v1", "BTC/USDT") in coordinator.positions
        assert coordinator.positions[("momentum_v1", "BTC/USDT")] == position

    def test_update_position_update(self):
        """Test updating an existing position."""
        coordinator = PortfolioCoordinator()

        # Open position
        position1 = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        coordinator.update_position("momentum_v1", "BTC/USDT", position1)

        # Update position
        position2 = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=-1,  # Reversed direction
            size_usd=12000.0,
            entry_time=datetime.now(),
            entry_price=51000.0
        )
        coordinator.update_position("momentum_v1", "BTC/USDT", position2)

        assert len(coordinator.positions) == 1
        assert coordinator.positions[("momentum_v1", "BTC/USDT")] == position2

    def test_update_position_close(self):
        """Test closing a position."""
        coordinator = PortfolioCoordinator()

        # Open position
        position = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        coordinator.update_position("momentum_v1", "BTC/USDT", position)
        assert len(coordinator.positions) == 1

        # Close position
        coordinator.update_position("momentum_v1", "BTC/USDT", None)
        assert len(coordinator.positions) == 0

    def test_update_position_close_nonexistent(self):
        """Test closing a position that doesn't exist (should not raise error)."""
        coordinator = PortfolioCoordinator()

        # Should not raise an error
        coordinator.update_position("strategy_1", "BTC/USDT", None)
        assert len(coordinator.positions) == 0

    def test_get_total_exposure_empty(self):
        """Test total exposure calculation with no positions."""
        coordinator = PortfolioCoordinator()
        assert coordinator.get_total_exposure() == 0.0

    def test_get_total_exposure_single_position(self):
        """Test total exposure calculation with one position."""
        coordinator = PortfolioCoordinator()

        position = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        coordinator.update_position("momentum_v1", "BTC/USDT", position)

        assert coordinator.get_total_exposure() == 10000.0

    def test_get_total_exposure_multiple_positions(self):
        """Test total exposure calculation with multiple positions."""
        coordinator = PortfolioCoordinator()

        positions = [
            Position(
                strategy_name=f"strategy_{i}",
                asset=f"ASSET{i}/USDT",
                direction=1,
                size_usd=5000.0 + i * 1000,
                entry_time=datetime.now(),
                entry_price=50000.0
            )
            for i in range(3)
        ]

        for i, pos in enumerate(positions):
            coordinator.update_position(f"strategy_{i}", f"ASSET{i}/USDT", pos)

        # 5000 + 6000 + 7000 = 18000
        assert coordinator.get_total_exposure() == 18000.0

    def test_get_positions_by_strategy(self):
        """Test retrieving positions for a specific strategy."""
        coordinator = PortfolioCoordinator()

        # Add positions for multiple strategies
        pos1 = Position(
            strategy_name="strategy_1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        pos2 = Position(
            strategy_name="strategy_1",
            asset="ETH/USDT",
            direction=-1,
            size_usd=5000.0,
            entry_time=datetime.now(),
            entry_price=3000.0
        )
        pos3 = Position(
            strategy_name="strategy_2",
            asset="SOL/USDT",
            direction=1,
            size_usd=3000.0,
            entry_time=datetime.now(),
            entry_price=100.0
        )

        coordinator.update_position("strategy_1", "BTC/USDT", pos1)
        coordinator.update_position("strategy_1", "ETH/USDT", pos2)
        coordinator.update_position("strategy_2", "SOL/USDT", pos3)

        # Get positions for strategy_1
        strategy1_positions = coordinator.get_positions_by_strategy("strategy_1")
        assert len(strategy1_positions) == 2
        assert pos1 in strategy1_positions
        assert pos2 in strategy1_positions

        # Get positions for strategy_2
        strategy2_positions = coordinator.get_positions_by_strategy("strategy_2")
        assert len(strategy2_positions) == 1
        assert pos3 in strategy2_positions

        # Get positions for non-existent strategy
        strategy3_positions = coordinator.get_positions_by_strategy("strategy_3")
        assert len(strategy3_positions) == 0

    def test_get_positions_by_asset(self):
        """Test retrieving positions for a specific asset."""
        coordinator = PortfolioCoordinator()

        # Add positions for multiple assets
        pos1 = Position(
            strategy_name="strategy_1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        pos2 = Position(
            strategy_name="strategy_2",
            asset="BTC/USDT",
            direction=-1,
            size_usd=5000.0,
            entry_time=datetime.now(),
            entry_price=51000.0
        )
        pos3 = Position(
            strategy_name="strategy_1",
            asset="ETH/USDT",
            direction=1,
            size_usd=3000.0,
            entry_time=datetime.now(),
            entry_price=3000.0
        )

        coordinator.update_position("strategy_1", "BTC/USDT", pos1)
        coordinator.update_position("strategy_2", "BTC/USDT", pos2)
        coordinator.update_position("strategy_1", "ETH/USDT", pos3)

        # Get positions for BTC/USDT
        btc_positions = coordinator.get_positions_by_asset("BTC/USDT")
        assert len(btc_positions) == 2
        assert pos1 in btc_positions
        assert pos2 in btc_positions

        # Get positions for ETH/USDT
        eth_positions = coordinator.get_positions_by_asset("ETH/USDT")
        assert len(eth_positions) == 1
        assert pos3 in eth_positions

    def test_get_snapshot_empty(self):
        """Test snapshot with no positions."""
        coordinator = PortfolioCoordinator(
            max_positions=10,
            max_total_exposure=100000.0
        )

        snapshot = coordinator.get_snapshot()

        assert snapshot['total_positions'] == 0
        assert snapshot['max_positions'] == 10
        assert snapshot['total_exposure_usd'] == 0.0
        assert snapshot['max_exposure_usd'] == 100000.0
        assert snapshot['total_unrealized_pnl'] == 0.0
        assert snapshot['utilization_pct'] == 0.0
        assert snapshot['exposure_pct'] == 0.0
        assert snapshot['positions_by_strategy'] == {}
        assert len(snapshot['positions']) == 0
        assert 'timestamp' in snapshot

    def test_get_snapshot_with_positions(self):
        """Test snapshot with positions."""
        coordinator = PortfolioCoordinator(
            max_positions=10,
            max_total_exposure=100000.0
        )

        # Add positions
        pos1 = Position(
            strategy_name="strategy_1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        pos1.update_pnl(51000.0)  # Profitable position

        pos2 = Position(
            strategy_name="strategy_1",
            asset="ETH/USDT",
            direction=-1,
            size_usd=5000.0,
            entry_time=datetime.now(),
            entry_price=3000.0
        )
        pos2.update_pnl(2900.0)  # Profitable short

        coordinator.update_position("strategy_1", "BTC/USDT", pos1)
        coordinator.update_position("strategy_1", "ETH/USDT", pos2)

        snapshot = coordinator.get_snapshot()

        assert snapshot['total_positions'] == 2
        assert snapshot['max_positions'] == 10
        assert snapshot['total_exposure_usd'] == 15000.0
        assert snapshot['max_exposure_usd'] == 100000.0
        assert snapshot['utilization_pct'] == 20.0  # 2/10 * 100
        assert snapshot['exposure_pct'] == 15.0  # 15000/100000 * 100
        assert snapshot['positions_by_strategy'] == {'strategy_1': 2}
        assert len(snapshot['positions']) == 2

        # Check total unrealized PnL
        expected_pnl = pos1.unrealized_pnl + pos2.unrealized_pnl
        assert abs(snapshot['total_unrealized_pnl'] - expected_pnl) < 0.01

    def test_repr(self):
        """Test string representation."""
        coordinator = PortfolioCoordinator(
            max_positions=5,
            max_total_exposure=50000.0
        )

        # Empty coordinator
        repr_str = repr(coordinator)
        assert "0/5 positions" in repr_str
        assert "$0/$50,000 exposure" in repr_str

        # Add a position
        position = Position(
            strategy_name="momentum_v1",
            asset="BTC/USDT",
            direction=1,
            size_usd=10000.0,
            entry_time=datetime.now(),
            entry_price=50000.0
        )
        coordinator.update_position("momentum_v1", "BTC/USDT", position)

        repr_str = repr(coordinator)
        assert "1/5 positions" in repr_str
        assert "$10,000/$50,000 exposure" in repr_str
