"""Integration tests for API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import create_app


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch("api.dependencies._state_manager", None):
            with patch("api.routers.health.get_state_manager") as mock_get_sm:
                mock_sm = MagicMock()
                mock_engine = MagicMock()
                mock_sm._get_engine.return_value = mock_engine

                # Mock successful DB connection
                mock_conn = MagicMock()
                mock_engine.connect.return_value.__enter__.return_value = mock_conn

                mock_get_sm.return_value = mock_sm

                # Store for access in tests
                self.mock_engine = mock_engine

                app = create_app()
                yield TestClient(app)

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert "timestamp" in data

    def test_ready_check(self, client):
        """Test readiness endpoint."""
        with patch("sqlalchemy.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.get_table_names.return_value = [
                "strategies",
                "positions",
                "trades",
                "signals",
            ]
            mock_inspect.return_value = mock_inspector

            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is True
            assert data["checks"]["database"] is True


class TestStrategyEndpoints:
    """Tests for strategy endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("api.dependencies._state_manager", None):
            with patch("api.dependencies.get_state_manager") as mock_get_sm:
                mock_sm = MagicMock()
                mock_engine = MagicMock()
                mock_sm._get_engine.return_value = mock_engine
                mock_get_sm.return_value = mock_sm

                # Store for access in tests
                self.mock_engine = mock_engine

                app = create_app()
                yield TestClient(app)

    def test_list_strategies(self, client):
        """Test listing strategies."""
        # Mock database response
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_result = MagicMock()
        mock_result.mappings.return_value = [
            {
                "id": 1,
                "strategy_name": "test_strategy",
                "version": "v1",
                "starlisting_id": 123,
                "interval": "15m",
                "model_id": "model_1",
                "is_active": True,
                "created_at": 1700000000000,
                "updated_at": 1700000000000,
            }
        ]
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/strategies")

        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert "total" in data

    def test_list_strategies_filter_active(self, client):
        """Test listing only active strategies."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/strategies?active_only=true")

        assert response.status_code == 200


class TestTradeEndpoints:
    """Tests for trade endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("api.dependencies._state_manager", None):
            with patch("api.dependencies.get_state_manager") as mock_get_sm:
                mock_sm = MagicMock()
                mock_engine = MagicMock()
                mock_sm._get_engine.return_value = mock_engine
                mock_get_sm.return_value = mock_sm

                self.mock_engine = mock_engine

                app = create_app()
                yield TestClient(app)

    def test_list_trades(self, client):
        """Test listing trades."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock count query
        mock_conn.execute.return_value.scalar.return_value = 0

        # Mock trades query
        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/trades")

        assert response.status_code == 200
        data = response.json()
        assert "trades" in data
        assert "total" in data

    def test_list_trades_with_pagination(self, client):
        """Test trade pagination."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/trades?limit=50&offset=10")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 50
        assert data["offset"] == 10

    def test_get_recent_trades(self, client):
        """Test getting recent trades."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/trades/recent?limit=5")

        assert response.status_code == 200


class TestPositionEndpoints:
    """Tests for position endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("api.dependencies._state_manager", None):
            with patch("api.dependencies.get_state_manager") as mock_get_sm:
                mock_sm = MagicMock()
                mock_engine = MagicMock()
                mock_sm._get_engine.return_value = mock_engine
                mock_get_sm.return_value = mock_sm

                self.mock_engine = mock_engine

                app = create_app()
                yield TestClient(app)

    def test_list_positions(self, client):
        """Test listing positions."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/positions")

        assert response.status_code == 200
        data = response.json()
        assert "positions" in data

    def test_get_open_positions(self, client):
        """Test getting open positions."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/positions/open")

        assert response.status_code == 200

    def test_get_closed_positions(self, client):
        """Test getting closed positions."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_conn.execute.return_value.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result

        response = client.get("/api/v1/positions/closed")

        assert response.status_code == 200


class TestPortfolioEndpoints:
    """Tests for portfolio endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("api.dependencies._state_manager", None):
            with patch("api.dependencies.get_state_manager") as mock_get_sm:
                mock_sm = MagicMock()
                mock_engine = MagicMock()
                mock_sm._get_engine.return_value = mock_engine
                mock_get_sm.return_value = mock_sm

                self.mock_engine = mock_engine

                app = create_app()
                yield TestClient(app)

    def test_get_portfolio_summary(self, client):
        """Test getting portfolio summary."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock closed stats
        mock_closed = MagicMock()
        mock_closed.mappings.return_value.fetchone.return_value = {
            "realized_pnl": 0,
            "total_fees": 0,
        }

        # Mock open stats
        mock_open = MagicMock()
        mock_open.mappings.return_value.fetchone.return_value = {
            "open_count": 0,
            "unrealized_pnl": 0,
            "allocated_capital": 0,
            "long_exposure": 0,
            "short_exposure": 0,
        }

        # Mock strategy stats
        mock_strategy = MagicMock()
        mock_strategy.mappings.return_value.fetchone.return_value = {
            "total": 0,
            "active": 0,
        }

        # Mock trades today
        mock_trades = MagicMock()
        mock_trades.scalar.return_value = 0

        mock_conn.execute.side_effect = [
            mock_closed,
            mock_open,
            mock_strategy,
            mock_trades,
        ]

        response = client.get("/api/v1/portfolio/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_equity_usd" in data
        assert "total_pnl_usd" in data


class TestStatsEndpoints:
    """Tests for statistics endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("api.dependencies._state_manager", None):
            with patch("api.dependencies.get_state_manager") as mock_get_sm:
                mock_sm = MagicMock()
                mock_engine = MagicMock()
                mock_sm._get_engine.return_value = mock_engine
                mock_get_sm.return_value = mock_sm

                self.mock_engine = mock_engine

                app = create_app()
                yield TestClient(app)

    def test_get_financial_summary(self, client):
        """Test getting financial summary."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock closed stats
        mock_closed = MagicMock()
        mock_closed.mappings.return_value.fetchone.return_value = {
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
        mock_open = MagicMock()
        mock_open.mappings.return_value.fetchone.return_value = {
            "unrealized_pnl": 0,
        }

        mock_conn.execute.side_effect = [mock_closed, mock_open]

        response = client.get("/api/v1/stats/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_equity_usd" in data
        assert "win_rate" in data

    def test_get_leaderboard(self, client):
        """Test getting leaderboard."""
        mock_conn = MagicMock()
        self.mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_result = MagicMock()
        mock_result.mappings.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_conn.execute.return_value.scalar.return_value = 0

        response = client.get("/api/v1/stats/leaderboard")

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "metric" in data
