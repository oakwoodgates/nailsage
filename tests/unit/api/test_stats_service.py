"""Unit tests for StatsService."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from api.services.stats_service import StatsService
from api.schemas.stats import FinancialSummary, DailyPnLResponse, LeaderboardResponse


class TestStatsService:
    """Tests for StatsService."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock StateManager."""
        mock = Mock()
        mock._get_engine.return_value = MagicMock()
        return mock

    @pytest.fixture
    def stats_service(self, mock_state_manager):
        """Create StatsService with mock dependencies."""
        return StatsService(
            state_manager=mock_state_manager,
            initial_capital=100000.0,
        )

    def test_get_financial_summary_empty(self, stats_service, mock_state_manager):
        """Test financial summary with no data."""
        # Mock database responses
        mock_engine = mock_state_manager._get_engine.return_value
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock closed stats
        mock_closed_result = MagicMock()
        mock_closed_result.mappings.return_value.fetchone.return_value = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "realized_pnl": 0,
            "total_fees": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "avg_trade": 0,
        }

        # Mock open stats
        mock_open_result = MagicMock()
        mock_open_result.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 0,
        }

        mock_conn.execute.side_effect = [mock_closed_result, mock_open_result]

        summary = stats_service.get_financial_summary()

        assert isinstance(summary, FinancialSummary)
        assert summary.total_trades == 0
        assert summary.win_rate == 0.0
        assert summary.total_pnl_usd == 0.0

    def test_get_financial_summary_with_data(self, stats_service, mock_state_manager):
        """Test financial summary with trading data."""
        mock_engine = mock_state_manager._get_engine.return_value
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock closed stats
        mock_closed_result = MagicMock()
        mock_closed_result.mappings.return_value.fetchone.return_value = {
            "total_trades": 100,
            "wins": 60,
            "losses": 40,
            "realized_pnl": 5000.0,
            "total_fees": 100.0,
            "avg_win": 150.0,
            "avg_loss": -75.0,
            "largest_win": 500.0,
            "largest_loss": -200.0,
            "gross_profit": 9000.0,
            "gross_loss": 4000.0,
            "avg_trade": 50.0,
        }

        # Mock open stats
        mock_open_result = MagicMock()
        mock_open_result.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 1000.0,
        }

        mock_conn.execute.side_effect = [mock_closed_result, mock_open_result]

        summary = stats_service.get_financial_summary()

        assert summary.total_trades == 100
        assert summary.total_wins == 60
        assert summary.total_losses == 40
        assert summary.win_rate == 60.0
        assert summary.realized_pnl_usd == 5000.0
        assert summary.unrealized_pnl_usd == 1000.0
        assert summary.total_pnl_usd == 6000.0
        assert summary.profit_factor == 9000.0 / 4000.0

    def test_win_rate_calculation(self, stats_service, mock_state_manager):
        """Test win rate calculation."""
        mock_engine = mock_state_manager._get_engine.return_value
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # 75 wins, 25 losses = 75% win rate
        mock_closed_result = MagicMock()
        mock_closed_result.mappings.return_value.fetchone.return_value = {
            "total_trades": 100,
            "wins": 75,
            "losses": 25,
            "realized_pnl": 0,
            "total_fees": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "avg_trade": 0,
        }

        mock_open_result = MagicMock()
        mock_open_result.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 0,
        }

        mock_conn.execute.side_effect = [mock_closed_result, mock_open_result]

        summary = stats_service.get_financial_summary()

        assert summary.win_rate == 75.0

    def test_profit_factor_calculation(self, stats_service, mock_state_manager):
        """Test profit factor calculation."""
        mock_engine = mock_state_manager._get_engine.return_value
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Gross profit 10000, gross loss 5000 = profit factor 2.0
        mock_closed_result = MagicMock()
        mock_closed_result.mappings.return_value.fetchone.return_value = {
            "total_trades": 50,
            "wins": 30,
            "losses": 20,
            "realized_pnl": 5000.0,
            "total_fees": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "gross_profit": 10000.0,
            "gross_loss": 5000.0,
            "avg_trade": 0,
        }

        mock_open_result = MagicMock()
        mock_open_result.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 0,
        }

        mock_conn.execute.side_effect = [mock_closed_result, mock_open_result]

        summary = stats_service.get_financial_summary()

        assert summary.profit_factor == 2.0

    def test_profit_factor_zero_loss(self, stats_service, mock_state_manager):
        """Test profit factor when no losses (should be 0)."""
        mock_engine = mock_state_manager._get_engine.return_value
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_closed_result = MagicMock()
        mock_closed_result.mappings.return_value.fetchone.return_value = {
            "total_trades": 10,
            "wins": 10,
            "losses": 0,
            "realized_pnl": 1000.0,
            "total_fees": 0,
            "avg_win": 100.0,
            "avg_loss": 0,
            "largest_win": 200.0,
            "largest_loss": 0,
            "gross_profit": 1000.0,
            "gross_loss": 0,  # No losses
            "avg_trade": 100.0,
        }

        mock_open_result = MagicMock()
        mock_open_result.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 0,
        }

        mock_conn.execute.side_effect = [mock_closed_result, mock_open_result]

        summary = stats_service.get_financial_summary()

        # Profit factor is 0 when there are no losses (division by zero protection)
        assert summary.profit_factor == 0

    def test_total_pnl_percentage(self, stats_service, mock_state_manager):
        """Test total P&L percentage calculation."""
        mock_engine = mock_state_manager._get_engine.return_value
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # 10% return on 100000 initial capital
        mock_closed_result = MagicMock()
        mock_closed_result.mappings.return_value.fetchone.return_value = {
            "total_trades": 50,
            "wins": 30,
            "losses": 20,
            "realized_pnl": 8000.0,
            "total_fees": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "avg_trade": 0,
        }

        mock_open_result = MagicMock()
        mock_open_result.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 2000.0,
        }

        mock_conn.execute.side_effect = [mock_closed_result, mock_open_result]

        summary = stats_service.get_financial_summary()

        # Total P&L: 8000 + 2000 = 10000 = 10% of 100000
        assert summary.total_pnl_usd == 10000.0
        assert summary.total_pnl_pct == 10.0
