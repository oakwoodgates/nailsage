"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from api.schemas.common import PaginationParams, PaginatedResponse, ErrorResponse
from api.schemas.strategies import StrategyResponse, StrategyWithStats, BankrollResponse, BankrollUpdateRequest
from api.schemas.arenas import ArenaSummary
from api.schemas.trades import TradeResponse
from api.schemas.positions import PositionResponse
from api.schemas.portfolio import PortfolioSummary, AllocationItem
from api.schemas.stats import FinancialSummary, LeaderboardEntry
from api.schemas.websocket import SubscribeRequest, WebSocketMessage


class TestPaginationParams:
    """Tests for PaginationParams schema."""

    def test_default_values(self):
        """Test default pagination values."""
        params = PaginationParams()
        assert params.limit == 100
        assert params.offset == 0

    def test_custom_values(self):
        """Test custom pagination values."""
        params = PaginationParams(limit=50, offset=10)
        assert params.limit == 50
        assert params.offset == 10

    def test_limit_validation_min(self):
        """Test limit minimum validation."""
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)

    def test_limit_validation_max(self):
        """Test limit maximum validation."""
        with pytest.raises(ValidationError):
            PaginationParams(limit=1001)

    def test_offset_validation(self):
        """Test offset cannot be negative."""
        with pytest.raises(ValidationError):
            PaginationParams(offset=-1)


class TestStrategyResponse:
    """Tests for StrategyResponse schema."""

    def test_valid_strategy(self):
        """Test valid strategy response."""
        strategy = StrategyResponse(
            id=1,
            strategy_name="test_strategy",
            version="v1",
            starlisting_id=123,
            interval="15m",
            model_id="model_123",
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
        )
        assert strategy.id == 1
        assert strategy.strategy_name == "test_strategy"
        assert strategy.is_active is True

    def test_optional_model_id(self):
        """Test model_id is optional."""
        strategy = StrategyResponse(
            id=1,
            strategy_name="test",
            version="v1",
            starlisting_id=123,
            interval="1m",
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
        )
        assert strategy.model_id is None

    def test_optional_arena(self):
        """Test arena is optional."""
        strategy = StrategyResponse(
            id=1,
            strategy_name="test",
            version="v1",
            starlisting_id=123,
            interval="1m",
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
        )
        assert strategy.arena is None
        assert strategy.arena_id is None

    def test_with_arena(self):
        """Test strategy with arena summary."""
        arena = ArenaSummary(
            id=1,
            starlisting_id=123,
            trading_pair="BTC/USD",
            interval="15m",
            coin="BTC",
            coin_name="Bitcoin",
            quote="USD",
            quote_name="US Dollar",
            exchange="hyperliquid",
            exchange_name="Hyperliquid",
            market_type="perps",
            market_name="Perpetuals",
        )
        strategy = StrategyResponse(
            id=1,
            strategy_name="test",
            version="v1",
            starlisting_id=123,
            arena_id=1,
            interval="15m",
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
            arena=arena,
        )
        assert strategy.arena_id == 1
        assert strategy.arena is not None
        assert strategy.arena.trading_pair == "BTC/USD"
        assert strategy.arena.coin == "BTC"
        assert strategy.arena.coin_name == "Bitcoin"
        assert strategy.arena.exchange == "hyperliquid"
        assert strategy.arena.exchange_name == "Hyperliquid"


class TestStrategyWithStats:
    """Tests for StrategyWithStats schema."""

    def test_default_stats_values(self):
        """Test default statistics values."""
        strategy = StrategyWithStats(
            id=1,
            strategy_name="test",
            version="v1",
            starlisting_id=123,
            interval="1m",
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
        )
        assert strategy.total_trades == 0
        assert strategy.total_pnl_usd == 0.0
        assert strategy.win_rate == 0.0

    def test_with_stats(self):
        """Test strategy with populated stats."""
        strategy = StrategyWithStats(
            id=1,
            strategy_name="test",
            version="v1",
            starlisting_id=123,
            interval="1m",
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
            total_trades=100,
            win_count=60,
            loss_count=40,
            win_rate=60.0,
            total_pnl_usd=5000.0,
            total_pnl_pct=5.0,
        )
        assert strategy.total_trades == 100
        assert strategy.win_rate == 60.0


class TestTradeResponse:
    """Tests for TradeResponse schema."""

    def test_valid_trade(self):
        """Test valid trade response."""
        trade = TradeResponse(
            id=1,
            position_id=1,
            strategy_id=1,
            starlisting_id=123,
            trade_type="open_long",
            size=1000.0,
            price=50000.0,
            fees=1.0,
            slippage=0.5,
            timestamp=1700000000000,
        )
        assert trade.id == 1
        assert trade.trade_type == "open_long"
        assert trade.size == 1000.0

    def test_trade_with_arena_id(self):
        """Test trade response with arena_id."""
        trade = TradeResponse(
            id=1,
            position_id=1,
            strategy_id=1,
            starlisting_id=123,
            trade_type="open_long",
            size=1000.0,
            price=50000.0,
            fees=1.0,
            slippage=0.5,
            timestamp=1700000000000,
            arena_id=1,
            strategy_name="test_strategy",
            position_side="long",
        )
        assert trade.arena_id == 1
        assert trade.strategy_name == "test_strategy"
        assert trade.position_side == "long"

    def test_trade_without_arena_id(self):
        """Test trade response without arena_id (null)."""
        trade = TradeResponse(
            id=1,
            position_id=1,
            strategy_id=1,
            starlisting_id=123,
            trade_type="close_short",
            size=500.0,
            price=48000.0,
            fees=0.5,
            slippage=0.25,
            timestamp=1700000000000,
        )
        assert trade.arena_id is None


class TestPositionResponse:
    """Tests for PositionResponse schema."""

    def test_open_position(self):
        """Test open position response."""
        position = PositionResponse(
            id=1,
            strategy_id=1,
            starlisting_id=123,
            side="long",
            size=1000.0,
            entry_price=50000.0,
            entry_timestamp=1700000000000,
            status="open",
        )
        assert position.status == "open"
        assert position.exit_price is None
        assert position.realized_pnl is None

    def test_closed_position(self):
        """Test closed position response."""
        position = PositionResponse(
            id=1,
            strategy_id=1,
            starlisting_id=123,
            side="long",
            size=1000.0,
            entry_price=50000.0,
            entry_timestamp=1700000000000,
            exit_price=51000.0,
            exit_timestamp=1700001000000,
            status="closed",
            realized_pnl=20.0,
            exit_reason="signal",
        )
        assert position.status == "closed"
        assert position.exit_price == 51000.0
        assert position.realized_pnl == 20.0


class TestPortfolioSummary:
    """Tests for PortfolioSummary schema."""

    def test_valid_portfolio(self):
        """Test valid portfolio summary."""
        portfolio = PortfolioSummary(
            initial_capital_usd=100000.0,
            total_equity_usd=105000.0,
            available_capital_usd=95000.0,
            allocated_capital_usd=10000.0,
            total_pnl_usd=5000.0,
            total_pnl_pct=5.0,
            realized_pnl_usd=3000.0,
            unrealized_pnl_usd=2000.0,
            total_open_positions=5,
            max_positions=10,
            position_utilization_pct=50.0,
            total_exposure_usd=10000.0,
            long_exposure_usd=7000.0,
            short_exposure_usd=3000.0,
            net_exposure_usd=4000.0,
            active_strategies=3,
            total_strategies=5,
            trades_today=10,
            total_fees_paid=50.0,
            timestamp=1700000000000,
        )
        assert portfolio.total_equity_usd == 105000.0
        assert portfolio.total_pnl_pct == 5.0


class TestFinancialSummary:
    """Tests for FinancialSummary schema."""

    def test_valid_financial_summary(self):
        """Test valid financial summary."""
        summary = FinancialSummary(
            total_equity_usd=105000.0,
            total_pnl_usd=5000.0,
            total_pnl_pct=5.0,
            realized_pnl_usd=3000.0,
            unrealized_pnl_usd=2000.0,
            total_trades=100,
            total_wins=60,
            total_losses=40,
            win_rate=60.0,
            avg_win_usd=100.0,
            avg_loss_usd=-50.0,
            largest_win_usd=500.0,
            largest_loss_usd=-200.0,
            profit_factor=2.0,
            avg_trade_usd=50.0,
            expectancy_usd=50.0,
            total_fees_paid=100.0,
            timestamp=1700000000000,
        )
        assert summary.win_rate == 60.0
        assert summary.profit_factor == 2.0


class TestWebSocketSchemas:
    """Tests for WebSocket schemas."""

    def test_subscribe_request(self):
        """Test subscribe request."""
        request = SubscribeRequest(
            action="subscribe",
            channels=["trades", "positions"],
        )
        assert request.action == "subscribe"
        assert len(request.channels) == 2

    def test_websocket_message(self):
        """Test WebSocket message."""
        message = WebSocketMessage(
            type="trade.new",
            channel="trades",
            timestamp=1700000000000,
            data={"trade_id": 1, "price": 50000.0},
        )
        assert message.type == "trade.new"
        assert message.channel == "trades"


class TestBankrollSchemas:
    """Tests for Bankroll schemas."""

    def test_bankroll_response_profitable(self):
        """Test bankroll response with profit."""
        response = BankrollResponse(
            strategy_id=1,
            strategy_name="test_strategy",
            initial_bankroll=10000.0,
            current_bankroll=12000.0,
            pnl=2000.0,
            pnl_pct=20.0,
            is_active=True,
        )
        assert response.strategy_id == 1
        assert response.initial_bankroll == 10000.0
        assert response.current_bankroll == 12000.0
        assert response.pnl == 2000.0
        assert response.pnl_pct == 20.0
        assert response.is_active is True

    def test_bankroll_response_loss(self):
        """Test bankroll response with loss."""
        response = BankrollResponse(
            strategy_id=2,
            strategy_name="losing_strategy",
            initial_bankroll=10000.0,
            current_bankroll=7500.0,
            pnl=-2500.0,
            pnl_pct=-25.0,
            is_active=True,
        )
        assert response.pnl == -2500.0
        assert response.pnl_pct == -25.0
        assert response.is_active is True

    def test_bankroll_response_depleted(self):
        """Test bankroll response when depleted (inactive)."""
        response = BankrollResponse(
            strategy_id=3,
            strategy_name="depleted_strategy",
            initial_bankroll=10000.0,
            current_bankroll=0.0,
            pnl=-10000.0,
            pnl_pct=-100.0,
            is_active=False,
        )
        assert response.current_bankroll == 0.0
        assert response.is_active is False

    def test_bankroll_update_request_valid(self):
        """Test valid bankroll update request."""
        request = BankrollUpdateRequest(bankroll=15000.0)
        assert request.bankroll == 15000.0

    def test_bankroll_update_request_minimum(self):
        """Test bankroll update with small positive value."""
        request = BankrollUpdateRequest(bankroll=0.01)
        assert request.bankroll == 0.01

    def test_bankroll_update_request_zero_invalid(self):
        """Test bankroll update with zero is invalid."""
        with pytest.raises(ValidationError):
            BankrollUpdateRequest(bankroll=0)

    def test_bankroll_update_request_negative_invalid(self):
        """Test bankroll update with negative value is invalid."""
        with pytest.raises(ValidationError):
            BankrollUpdateRequest(bankroll=-1000.0)
