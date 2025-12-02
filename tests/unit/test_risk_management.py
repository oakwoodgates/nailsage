"""Unit tests for risk management system."""

import pytest

from execution.risk.capital_allocator import CapitalAllocator, CapitalState
from execution.risk.exposure_tracker import ExposureTracker, ExposureState
from execution.risk.risk_manager import RiskManager, RiskCheckStatus


class TestCapitalAllocator:
    """Tests for CapitalAllocator."""

    def test_init_valid(self):
        """Test valid initialization."""
        allocator = CapitalAllocator(
            initial_capital=10000.0,
            max_allocation_pct=95.0,
            reserve_for_fees_pct=5.0,
        )

        assert allocator.initial_capital == 10000.0
        assert allocator.max_allocation_pct == 95.0
        assert allocator.reserve_for_fees_pct == 5.0

    def test_init_invalid_capital(self):
        """Test initialization with invalid capital."""
        with pytest.raises(ValueError, match="capital must be > 0"):
            CapitalAllocator(initial_capital=0)

    def test_init_invalid_max_allocation(self):
        """Test initialization with invalid max allocation."""
        with pytest.raises(ValueError, match="Max allocation"):
            CapitalAllocator(initial_capital=10000, max_allocation_pct=150)

    def test_init_exceeds_100_percent(self):
        """Test initialization when max + reserve > 100%."""
        with pytest.raises(ValueError, match="cannot exceed 100%"):
            CapitalAllocator(
                initial_capital=10000,
                max_allocation_pct=96,
                reserve_for_fees_pct=5,
            )

    def test_get_state_initial(self):
        """Test getting state with no allocations."""
        allocator = CapitalAllocator(
            initial_capital=10000.0,
            max_allocation_pct=95.0,
            reserve_for_fees_pct=5.0,
        )

        state = allocator.get_state()

        assert state.total_capital == 10000.0
        assert state.allocated_capital == 0.0
        assert state.reserved_capital == 500.0  # 5% of 10000
        assert state.available_capital == 9500.0  # 10000 - 0 - 500

    def test_can_allocate_sufficient_capital(self):
        """Test checking allocation with sufficient capital."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        can_allocate, reason = allocator.can_allocate(5000.0)

        assert can_allocate is True
        assert reason is None

    def test_can_allocate_insufficient_capital(self):
        """Test checking allocation with insufficient capital."""
        allocator = CapitalAllocator(
            initial_capital=10000.0,
            max_allocation_pct=95.0,
            reserve_for_fees_pct=5.0,
        )

        # Try to allocate more than available (9500)
        can_allocate, reason = allocator.can_allocate(10000.0)

        assert can_allocate is False
        assert "Insufficient capital" in reason

    def test_can_allocate_exceeds_max_pct(self):
        """Test allocation exceeding max allocation %."""
        allocator = CapitalAllocator(
            initial_capital=10000.0,
            max_allocation_pct=50.0,  # Only 50% can be allocated
        )

        # Try to allocate 60% of capital
        can_allocate, reason = allocator.can_allocate(6000.0)

        assert can_allocate is False
        assert "exceed max allocation" in reason

    def test_allocate_success(self):
        """Test successful allocation."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        allocator.allocate(position_id=1, amount=5000.0)

        state = allocator.get_state()
        assert state.allocated_capital == 5000.0
        assert allocator.get_allocated_for_position(1) == 5000.0

    def test_allocate_duplicate_position(self):
        """Test allocating to same position twice."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        allocator.allocate(position_id=1, amount=5000.0)

        with pytest.raises(ValueError, match="already has allocated capital"):
            allocator.allocate(position_id=1, amount=2000.0)

    def test_deallocate_success(self):
        """Test successful deallocation."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        allocator.allocate(position_id=1, amount=5000.0)
        amount = allocator.deallocate(position_id=1)

        assert amount == 5000.0
        assert allocator.get_state().allocated_capital == 0.0

    def test_deallocate_not_found(self):
        """Test deallocating position that doesn't exist."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        with pytest.raises(ValueError, match="has no allocated capital"):
            allocator.deallocate(position_id=999)

    def test_update_equity(self):
        """Test updating equity."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        allocator.update_equity(12000.0)

        state = allocator.get_state()
        assert state.total_capital == 12000.0

    def test_get_max_position_size(self):
        """Test calculating max position size."""
        allocator = CapitalAllocator(
            initial_capital=10000.0,
            max_allocation_pct=90.0,  # Adjust to accommodate reserve
            reserve_for_fees_pct=10.0,
        )

        max_size = allocator.get_max_position_size(price=100.0)

        # Available: 10000 - 1000 (reserve) = 9000
        assert max_size == 9000.0

    def test_get_max_position_size_with_limit(self):
        """Test max position size with percentage limit."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        max_size = allocator.get_max_position_size(
            price=100.0,
            max_position_pct=20.0,  # Max 20% of capital
        )

        # 20% of 10000 = 2000
        assert max_size == 2000.0

    def test_utilization_pct(self):
        """Test capital utilization percentage."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        allocator.allocate(1, 3000.0)
        allocator.allocate(2, 2000.0)

        state = allocator.get_state()
        assert state.utilization_pct() == 50.0  # 5000/10000

    def test_reset(self):
        """Test resetting allocator."""
        allocator = CapitalAllocator(initial_capital=10000.0)

        allocator.allocate(1, 5000.0)
        allocator.update_equity(12000.0)

        allocator.reset()

        state = allocator.get_state()
        assert state.total_capital == 10000.0
        assert state.allocated_capital == 0.0


class TestExposureTracker:
    """Tests for ExposureTracker."""

    def test_init_valid(self):
        """Test valid initialization."""
        tracker = ExposureTracker(
            max_positions=10,
            max_positions_per_asset=3,
            max_positions_per_strategy=5,
            max_exposure_per_asset_pct=25.0,
        )

        assert tracker.max_positions == 10
        assert tracker.max_positions_per_asset == 3

    def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            ExposureTracker(max_positions=-1)

        with pytest.raises(ValueError):
            ExposureTracker(max_exposure_per_asset_pct=150)

    def test_get_state_initial(self):
        """Test getting state with no positions."""
        tracker = ExposureTracker()

        state = tracker.get_state()

        assert state.total_exposure_usd == 0.0
        assert state.num_positions == 0
        assert len(state.by_asset) == 0

    def test_can_add_position_success(self):
        """Test checking if position can be added."""
        tracker = ExposureTracker(max_positions=5)

        can_add, reason = tracker.can_add_position(
            asset="BTC/USDT",
            strategy="test_strategy",
            exposure_usd=1000.0,
        )

        assert can_add is True
        assert reason is None

    def test_can_add_position_max_total(self):
        """Test max total positions limit."""
        tracker = ExposureTracker(max_positions=2)

        # Add 2 positions
        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.add_position(2, "ETH/USDT", "strategy1", 1000.0)

        # Try to add 3rd
        can_add, reason = tracker.can_add_position(
            asset="SOL/USDT",
            strategy="strategy1",
            exposure_usd=1000.0,
        )

        assert can_add is False
        assert "Max total positions reached" in reason

    def test_can_add_position_max_per_asset(self):
        """Test max positions per asset limit."""
        tracker = ExposureTracker(max_positions_per_asset=2)

        # Add 2 BTC positions
        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.add_position(2, "BTC/USDT", "strategy2", 1000.0)

        # Try to add 3rd BTC position
        can_add, reason = tracker.can_add_position(
            asset="BTC/USDT",
            strategy="strategy3",
            exposure_usd=1000.0,
        )

        assert can_add is False
        assert "Max positions for BTC/USDT reached" in reason

    def test_can_add_position_max_per_strategy(self):
        """Test max positions per strategy limit."""
        tracker = ExposureTracker(max_positions_per_strategy=2)

        # Add 2 positions for strategy1
        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.add_position(2, "ETH/USDT", "strategy1", 1000.0)

        # Try to add 3rd position for strategy1
        can_add, reason = tracker.can_add_position(
            asset="SOL/USDT",
            strategy="strategy1",
            exposure_usd=1000.0,
        )

        assert can_add is False
        assert "Max positions for strategy" in reason

    def test_can_add_position_max_exposure_pct(self):
        """Test max exposure % per asset."""
        tracker = ExposureTracker(max_exposure_per_asset_pct=20.0)

        # Add position with 15% exposure
        tracker.add_position(1, "BTC/USDT", "strategy1", 1500.0)

        # Try to add another with 10% exposure (total would be 25%)
        can_add, reason = tracker.can_add_position(
            asset="BTC/USDT",
            strategy="strategy2",
            exposure_usd=1000.0,
            total_capital=10000.0,
        )

        assert can_add is False
        assert "exceed max exposure for BTC/USDT" in reason

    def test_add_position_success(self):
        """Test adding a position."""
        tracker = ExposureTracker()

        tracker.add_position(
            position_id=1,
            asset="BTC/USDT",
            strategy="test_strategy",
            exposure_usd=5000.0,
        )

        state = tracker.get_state()
        assert state.num_positions == 1
        assert state.by_asset["BTC/USDT"] == 5000.0
        assert state.by_strategy["test_strategy"] == 1

    def test_add_position_duplicate(self):
        """Test adding same position twice."""
        tracker = ExposureTracker()

        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)

        with pytest.raises(ValueError, match="already being tracked"):
            tracker.add_position(1, "ETH/USDT", "strategy1", 2000.0)

    def test_remove_position_success(self):
        """Test removing a position."""
        tracker = ExposureTracker()

        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.remove_position(1)

        state = tracker.get_state()
        assert state.num_positions == 0
        assert "BTC/USDT" not in state.by_asset

    def test_remove_position_not_found(self):
        """Test removing position that doesn't exist."""
        tracker = ExposureTracker()

        with pytest.raises(ValueError, match="not being tracked"):
            tracker.remove_position(999)

    def test_update_exposure(self):
        """Test updating position exposure."""
        tracker = ExposureTracker()

        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.update_exposure(1, 1500.0)

        exposure = tracker.get_asset_exposure("BTC/USDT")
        assert exposure == 1500.0

    def test_get_asset_exposure(self):
        """Test getting total exposure for an asset."""
        tracker = ExposureTracker()

        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.add_position(2, "BTC/USDT", "strategy2", 2000.0)

        exposure = tracker.get_asset_exposure("BTC/USDT")
        assert exposure == 3000.0

    def test_reset(self):
        """Test resetting tracker."""
        tracker = ExposureTracker()

        tracker.add_position(1, "BTC/USDT", "strategy1", 1000.0)
        tracker.reset()

        state = tracker.get_state()
        assert state.num_positions == 0


class TestRiskManager:
    """Tests for RiskManager."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager for testing."""
        allocator = CapitalAllocator(
            initial_capital=10000.0,
            max_allocation_pct=90.0,
        )
        tracker = ExposureTracker(
            max_positions=5,
            max_positions_per_asset=2,
        )
        return RiskManager(allocator, tracker, enable_checks=True)

    def test_init(self, risk_manager):
        """Test initialization."""
        assert risk_manager.enable_checks is True

    def test_check_new_position_approved(self, risk_manager):
        """Test checking new position - approved."""
        result = risk_manager.check_new_position(
            asset="BTC/USDT",
            strategy="test_strategy",
            position_size_usd=1000.0,
            price=50000.0,
        )

        assert result.status == RiskCheckStatus.APPROVED
        assert result.approved is True
        assert result.max_position_size > 0

    def test_check_new_position_rejected_capital(self, risk_manager):
        """Test checking new position - rejected due to capital."""
        result = risk_manager.check_new_position(
            asset="BTC/USDT",
            strategy="test_strategy",
            position_size_usd=15000.0,  # More than available
            price=50000.0,
        )

        assert result.status == RiskCheckStatus.REJECTED
        assert result.approved is False
        assert "Capital check failed" in result.reason

    def test_check_new_position_warning(self, risk_manager):
        """Test checking new position - approved with warning."""
        # Allocate 8100 to get high utilization (81% > 80%)
        risk_manager.allocate_position(1, "ETH/USDT", "strategy1", 8100.0)

        result = risk_manager.check_new_position(
            asset="BTC/USDT",
            strategy="test_strategy",
            position_size_usd=500.0,
            price=50000.0,
        )

        assert result.status == RiskCheckStatus.WARNING
        assert result.approved is True
        assert len(result.warnings) > 0

    def test_allocate_position(self, risk_manager):
        """Test allocating a position."""
        risk_manager.allocate_position(
            position_id=1,
            asset="BTC/USDT",
            strategy="test_strategy",
            position_size_usd=5000.0,
        )

        # Check capital allocated
        allocated = risk_manager.capital_allocator.get_allocated_for_position(1)
        assert allocated == 5000.0

        # Check exposure tracked
        exposure = risk_manager.exposure_tracker.get_asset_exposure("BTC/USDT")
        assert exposure == 5000.0

    def test_deallocate_position(self, risk_manager):
        """Test deallocating a position."""
        risk_manager.allocate_position(1, "BTC/USDT", "strategy1", 5000.0)
        risk_manager.deallocate_position(1)

        # Check capital deallocated
        allocated = risk_manager.capital_allocator.get_allocated_for_position(1)
        assert allocated == 0.0

        # Check exposure removed
        state = risk_manager.exposure_tracker.get_state()
        assert state.num_positions == 0

    def test_update_equity(self, risk_manager):
        """Test updating equity."""
        risk_manager.update_equity(12000.0)

        capital_state = risk_manager.capital_allocator.get_state()
        assert capital_state.total_capital == 12000.0

    def test_get_risk_summary(self, risk_manager):
        """Test getting risk summary."""
        risk_manager.allocate_position(1, "BTC/USDT", "strategy1", 5000.0)

        summary = risk_manager.get_risk_summary()

        assert summary["capital"]["total"] == 10000.0
        assert summary["capital"]["allocated"] == 5000.0
        assert summary["exposure"]["num_positions"] == 1
        assert summary["checks_enabled"] is True

    def test_checks_disabled(self):
        """Test with checks disabled."""
        allocator = CapitalAllocator(initial_capital=10000.0)
        tracker = ExposureTracker()
        manager = RiskManager(allocator, tracker, enable_checks=False)

        # Should approve anything
        result = manager.check_new_position(
            asset="BTC/USDT",
            strategy="test",
            position_size_usd=100000.0,  # Way over capital
            price=50000.0,
        )

        assert result.approved is True
        assert "disabled" in result.reason

    def test_reset(self, risk_manager):
        """Test resetting risk manager."""
        risk_manager.allocate_position(1, "BTC/USDT", "strategy1", 5000.0)
        risk_manager.reset()

        summary = risk_manager.get_risk_summary()
        assert summary["capital"]["allocated"] == 0.0
        assert summary["exposure"]["num_positions"] == 0
