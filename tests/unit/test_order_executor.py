"""Unit tests for OrderExecutor."""

import pytest
from datetime import datetime

from execution.simulator.order_executor import (
    OrderExecutor,
    OrderExecutorConfig,
    OrderResult,
)
from execution.persistence.state_manager import Trade


class TestOrderExecutorConfig:
    """Test suite for OrderExecutorConfig validation."""

    def test_config_valid(self):
        """Test config initialization with valid parameters."""
        config = OrderExecutorConfig(
            fee_rate=0.001,
            slippage_bps=5.0,
            min_order_size_usd=10.0,
            max_order_size_usd=1_000_000.0,
        )

        assert config.fee_rate == 0.001
        assert config.slippage_bps == 5.0
        assert config.min_order_size_usd == 10.0
        assert config.max_order_size_usd == 1_000_000.0

    def test_config_defaults(self):
        """Test config defaults."""
        config = OrderExecutorConfig()

        assert config.fee_rate == 0.001  # 0.1%
        assert config.slippage_bps == 5.0  # 5 bps
        assert config.min_order_size_usd == 10.0
        assert config.max_order_size_usd == 1_000_000.0

    def test_config_invalid_fee_rate_negative(self):
        """Test config validation with negative fee rate."""
        with pytest.raises(ValueError, match="fee_rate must be between 0 and 0.1"):
            OrderExecutorConfig(fee_rate=-0.001)

    def test_config_invalid_fee_rate_too_high(self):
        """Test config validation with fee rate > 10%."""
        with pytest.raises(ValueError, match="fee_rate must be between 0 and 0.1"):
            OrderExecutorConfig(fee_rate=0.15)

    def test_config_invalid_slippage_negative(self):
        """Test config validation with negative slippage."""
        with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
            OrderExecutorConfig(slippage_bps=-1.0)

    def test_config_invalid_min_order_size(self):
        """Test config validation with invalid min order size."""
        with pytest.raises(ValueError, match="min_order_size_usd must be positive"):
            OrderExecutorConfig(min_order_size_usd=0.0)


class TestOrderExecutor:
    """Test suite for OrderExecutor class."""

    @pytest.fixture
    def config(self):
        """Create default config for testing."""
        return OrderExecutorConfig(
            fee_rate=0.001,  # 0.1%
            slippage_bps=5.0,  # 5 bps = 0.05%
            min_order_size_usd=10.0,
            max_order_size_usd=100_000.0,
        )

    @pytest.fixture
    def executor(self, config):
        """Create order executor for testing."""
        return OrderExecutor(config)

    def test_init(self, config):
        """Test order executor initialization."""
        executor = OrderExecutor(config)

        assert executor.config == config
        assert executor._orders_executed == 0

    # Test: Successful Long (Buy) Order
    def test_execute_long_order_success(self, executor):
        """Test successful long (buy) market order."""
        result = executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
            signal_id=1,
        )

        # Check success
        assert result.success is True
        assert result.rejection_reason is None

        # Check fill price (should be higher due to slippage for buy)
        # 5 bps = 0.0005 = 0.05%
        expected_fill_price = 90000.0 * 1.0005
        assert result.fill_price == pytest.approx(expected_fill_price)
        assert result.fill_price > 90000.0  # Slippage increases price for longs

        # Check size
        expected_size = 10000.0 / expected_fill_price
        assert result.size == pytest.approx(expected_size)

        # Check fees (0.1% of 10000 = $10)
        assert result.fees_usd == pytest.approx(10.0)

        # Check slippage cost
        slippage_diff = result.fill_price - 90000.0
        expected_slippage_cost = slippage_diff * result.size
        assert result.slippage_usd == pytest.approx(expected_slippage_cost)

        # Check trade object
        assert result.trade is not None
        assert result.trade.trade_type == 'open_long'
        assert result.trade.size == pytest.approx(expected_size)
        assert result.trade.price == pytest.approx(expected_fill_price)
        assert result.trade.fees == pytest.approx(10.0)
        assert result.trade.position_id == 1
        assert result.trade.strategy_id == 1
        assert result.trade.signal_id == 1

    # Test: Successful Short (Sell) Order
    def test_execute_short_order_success(self, executor):
        """Test successful short (sell) market order."""
        result = executor.execute_market_order(
            side='short',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=2,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_short',
        )

        # Check success
        assert result.success is True

        # Check fill price (should be lower due to slippage for sell)
        expected_fill_price = 90000.0 * 0.9995
        assert result.fill_price == pytest.approx(expected_fill_price)
        assert result.fill_price < 90000.0  # Slippage decreases price for shorts

        # Check size
        expected_size = 10000.0 / expected_fill_price
        assert result.size == pytest.approx(expected_size)

        # Check fees
        assert result.fees_usd == pytest.approx(10.0)

        # Check trade
        assert result.trade.trade_type == 'open_short'

    # Test: Order Rejection - Below Minimum
    def test_execute_order_below_minimum(self, executor):
        """Test order rejection when size is below minimum."""
        result = executor.execute_market_order(
            side='long',
            size_usd=5.0,  # Below min of $10
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        # Check rejection
        assert result.success is False
        assert result.trade is None
        assert "below minimum" in result.rejection_reason
        assert result.size == 0.0
        assert result.fees_usd == 0.0
        assert result.slippage_usd == 0.0

    # Test: Order Rejection - Above Maximum
    def test_execute_order_above_maximum(self, executor):
        """Test order rejection when size exceeds maximum."""
        result = executor.execute_market_order(
            side='long',
            size_usd=200_000.0,  # Above max of $100,000
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        # Check rejection
        assert result.success is False
        assert result.trade is None
        assert "exceeds maximum" in result.rejection_reason

    # Test: Different Trade Types
    def test_execute_close_long_order(self, executor):
        """Test closing a long position."""
        result = executor.execute_market_order(
            side='short',  # Sell to close long
            size_usd=10000.0,
            current_price=95000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='close_long',
        )

        assert result.success is True
        assert result.trade.trade_type == 'close_long'

    def test_execute_close_short_order(self, executor):
        """Test closing a short position."""
        result = executor.execute_market_order(
            side='long',  # Buy to close short
            size_usd=10000.0,
            current_price=85000.0,
            timestamp=1700000000000,
            position_id=2,
            strategy_id=1,
            starlisting_id=1,
            trade_type='close_short',
        )

        assert result.success is True
        assert result.trade.trade_type == 'close_short'

    # Test: Fee Calculation
    def test_calculate_fees(self, executor):
        """Test fee calculation."""
        fees = executor.calculate_fees(notional_usd=10000.0)
        assert fees == pytest.approx(10.0)  # 0.1% of 10000

        fees = executor.calculate_fees(notional_usd=50000.0)
        assert fees == pytest.approx(50.0)  # 0.1% of 50000

    # Test: Fill Price Calculation
    def test_calculate_fill_price_long(self, executor):
        """Test fill price calculation for long orders."""
        fill_price = executor.calculate_fill_price(
            side='long',
            current_price=90000.0
        )

        expected = 90000.0 * 1.0005  # +0.05% slippage
        assert fill_price == pytest.approx(expected)
        assert fill_price > 90000.0

    def test_calculate_fill_price_short(self, executor):
        """Test fill price calculation for short orders."""
        fill_price = executor.calculate_fill_price(
            side='short',
            current_price=90000.0
        )

        expected = 90000.0 * 0.9995  # -0.05% slippage
        assert fill_price == pytest.approx(expected)
        assert fill_price < 90000.0

    # Test: Order Counting
    def test_orders_executed_counter(self, executor):
        """Test that executed orders are counted."""
        assert executor._orders_executed == 0

        # Execute first order
        executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )
        assert executor._orders_executed == 1

        # Execute second order
        executor.execute_market_order(
            side='short',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=2,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_short',
        )
        assert executor._orders_executed == 2

        # Rejected orders should not count
        executor.execute_market_order(
            side='long',
            size_usd=5.0,  # Below minimum
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=3,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )
        assert executor._orders_executed == 2  # Still 2

    # Test: Statistics
    def test_get_stats(self, executor):
        """Test getting executor statistics."""
        stats = executor.get_stats()

        assert stats["orders_executed"] == 0
        assert stats["fee_rate"] == 0.001
        assert stats["slippage_bps"] == 5.0

        # Execute some orders
        executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        stats = executor.get_stats()
        assert stats["orders_executed"] == 1

    # Test: Different Slippage Settings
    def test_zero_slippage(self):
        """Test with zero slippage."""
        config = OrderExecutorConfig(
            fee_rate=0.001,
            slippage_bps=0.0,  # No slippage
        )
        executor = OrderExecutor(config)

        result = executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        # Fill price should equal current price (no slippage)
        assert result.fill_price == pytest.approx(90000.0)
        assert result.slippage_usd == pytest.approx(0.0)

    def test_high_slippage(self):
        """Test with high slippage."""
        config = OrderExecutorConfig(
            fee_rate=0.001,
            slippage_bps=50.0,  # 50 bps = 0.5%
        )
        executor = OrderExecutor(config)

        result = executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        # Fill price should be 0.5% higher
        expected_fill_price = 90000.0 * 1.005
        assert result.fill_price == pytest.approx(expected_fill_price)

    # Test: Different Fee Rates
    def test_zero_fees(self):
        """Test with zero fees."""
        config = OrderExecutorConfig(
            fee_rate=0.0,  # No fees
            slippage_bps=5.0,
        )
        executor = OrderExecutor(config)

        result = executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        assert result.fees_usd == pytest.approx(0.0)

    def test_high_fees(self):
        """Test with high fees."""
        config = OrderExecutorConfig(
            fee_rate=0.01,  # 1% fees
            slippage_bps=5.0,
        )
        executor = OrderExecutor(config)

        result = executor.execute_market_order(
            side='long',
            size_usd=10000.0,
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        # Fees should be 1% of notional
        assert result.fees_usd == pytest.approx(100.0)

    # Test: Edge Cases
    def test_exactly_at_minimum(self, executor):
        """Test order exactly at minimum size."""
        result = executor.execute_market_order(
            side='long',
            size_usd=10.0,  # Exactly at minimum
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        assert result.success is True
        assert result.trade is not None

    def test_exactly_at_maximum(self, executor):
        """Test order exactly at maximum size."""
        result = executor.execute_market_order(
            side='long',
            size_usd=100_000.0,  # Exactly at maximum
            current_price=90000.0,
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        assert result.success is True
        assert result.trade is not None

    def test_very_low_price(self, executor):
        """Test with very low price."""
        result = executor.execute_market_order(
            side='long',
            size_usd=1000.0,
            current_price=0.001,  # Very low price
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        assert result.success is True
        assert result.size > 0

    def test_very_high_price(self, executor):
        """Test with very high price."""
        result = executor.execute_market_order(
            side='long',
            size_usd=1000.0,
            current_price=1_000_000.0,  # Very high price
            timestamp=1700000000000,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
        )

        assert result.success is True
        assert result.size > 0
        assert result.size < 1.0  # Small size due to high price


class TestOrderResult:
    """Test suite for OrderResult dataclass."""

    def test_order_result_success(self):
        """Test OrderResult for successful order."""
        trade = Trade(
            id=None,
            position_id=1,
            strategy_id=1,
            starlisting_id=1,
            trade_type='open_long',
            size=0.1,
            price=90000.0,
            fees=10.0,
            slippage=5.0,
            timestamp=1700000000000,
        )

        result = OrderResult(
            trade=trade,
            fill_price=90000.0,
            size=0.1,
            notional_usd=9000.0,
            fees_usd=10.0,
            slippage_usd=5.0,
            success=True,
        )

        assert result.success is True
        assert result.rejection_reason is None
        assert result.trade == trade
        assert result.fill_price == 90000.0

    def test_order_result_rejection(self):
        """Test OrderResult for rejected order."""
        result = OrderResult(
            trade=None,
            fill_price=90000.0,
            size=0.0,
            notional_usd=0.0,
            fees_usd=0.0,
            slippage_usd=0.0,
            success=False,
            rejection_reason="Order size below minimum",
        )

        assert result.success is False
        assert result.trade is None
        assert result.rejection_reason is not None
        assert "below minimum" in result.rejection_reason
