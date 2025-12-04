"""Unit tests for Arena schemas."""

import pytest
from pydantic import ValidationError

from api.schemas.arenas import (
    ExchangeResponse,
    CoinResponse,
    MarketTypeResponse,
    ArenaResponse,
    ArenaSummary,
    ArenaListResponse,
    ArenaSyncRequest,
    ArenaSyncResponse,
    ExchangeListResponse,
    CoinListResponse,
    MarketTypeListResponse,
)


class TestExchangeResponse:
    """Tests for ExchangeResponse schema."""

    def test_valid_exchange(self):
        """Test valid exchange response."""
        exchange = ExchangeResponse(
            id=1,
            slug="hyperliquid",
            name="Hyperliquid",
        )
        assert exchange.id == 1
        assert exchange.slug == "hyperliquid"
        assert exchange.name == "Hyperliquid"

    def test_exchange_missing_field(self):
        """Test exchange with missing required field."""
        with pytest.raises(ValidationError):
            ExchangeResponse(id=1, slug="binance")


class TestCoinResponse:
    """Tests for CoinResponse schema."""

    def test_valid_coin(self):
        """Test valid coin response."""
        coin = CoinResponse(
            id=1,
            symbol="BTC",
            name="Bitcoin",
        )
        assert coin.id == 1
        assert coin.symbol == "BTC"
        assert coin.name == "Bitcoin"

    def test_coin_missing_field(self):
        """Test coin with missing required field."""
        with pytest.raises(ValidationError):
            CoinResponse(id=1, symbol="ETH")


class TestMarketTypeResponse:
    """Tests for MarketTypeResponse schema."""

    def test_valid_market_type(self):
        """Test valid market type response."""
        market_type = MarketTypeResponse(
            id=1,
            type="perps",
            name="Perpetuals",
        )
        assert market_type.id == 1
        assert market_type.type == "perps"
        assert market_type.name == "Perpetuals"

    def test_market_type_spot(self):
        """Test spot market type."""
        market_type = MarketTypeResponse(
            id=2,
            type="spot",
            name="Spot",
        )
        assert market_type.type == "spot"


class TestArenaResponse:
    """Tests for ArenaResponse schema."""

    def test_valid_arena(self):
        """Test valid arena response with nested objects."""
        arena = ArenaResponse(
            id=1,
            starlisting_id=123,
            trading_pair="BTC/USD",
            trading_pair_id=1,
            coin=CoinResponse(id=1, symbol="BTC", name="Bitcoin"),
            quote=CoinResponse(id=2, symbol="USD", name="US Dollar"),
            exchange=ExchangeResponse(id=1, slug="hyperliquid", name="Hyperliquid"),
            market_type=MarketTypeResponse(id=1, type="perps", name="Perpetuals"),
            interval="15m",
            interval_seconds=900,
            is_active=True,
            last_synced_at=1700000000000,
            created_at=1700000000000,
            updated_at=1700000000000,
        )
        assert arena.id == 1
        assert arena.trading_pair == "BTC/USD"
        assert arena.coin.symbol == "BTC"
        assert arena.quote.symbol == "USD"
        assert arena.exchange.slug == "hyperliquid"
        assert arena.market_type.type == "perps"
        assert arena.interval_seconds == 900

    def test_arena_optional_trading_pair_id(self):
        """Test arena with optional trading_pair_id."""
        arena = ArenaResponse(
            id=1,
            starlisting_id=123,
            trading_pair="SOL/USD",
            trading_pair_id=None,
            coin=CoinResponse(id=3, symbol="SOL", name="Solana"),
            quote=CoinResponse(id=2, symbol="USD", name="US Dollar"),
            exchange=ExchangeResponse(id=1, slug="hyperliquid", name="Hyperliquid"),
            market_type=MarketTypeResponse(id=1, type="perps", name="Perpetuals"),
            interval="1m",
            interval_seconds=60,
            is_active=True,
            created_at=1700000000000,
            updated_at=1700000000000,
        )
        assert arena.trading_pair_id is None
        assert arena.last_synced_at is None


class TestArenaSummary:
    """Tests for ArenaSummary schema."""

    def test_valid_arena_summary(self):
        """Test valid arena summary with all fields."""
        summary = ArenaSummary(
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
        assert summary.id == 1
        assert summary.coin == "BTC"
        assert summary.coin_name == "Bitcoin"
        assert summary.quote == "USD"
        assert summary.quote_name == "US Dollar"
        assert summary.exchange == "hyperliquid"
        assert summary.exchange_name == "Hyperliquid"
        assert summary.market_type == "perps"
        assert summary.market_name == "Perpetuals"

    def test_arena_summary_missing_name_field(self):
        """Test arena summary with missing name field."""
        with pytest.raises(ValidationError):
            ArenaSummary(
                id=1,
                starlisting_id=123,
                trading_pair="BTC/USD",
                interval="15m",
                coin="BTC",
                # missing coin_name
                quote="USD",
                quote_name="US Dollar",
                exchange="hyperliquid",
                exchange_name="Hyperliquid",
                market_type="perps",
                market_name="Perpetuals",
            )


class TestArenaListResponse:
    """Tests for ArenaListResponse schema."""

    def test_valid_arena_list(self):
        """Test valid arena list response."""
        arena_list = ArenaListResponse(
            arenas=[
                ArenaResponse(
                    id=1,
                    starlisting_id=1,
                    trading_pair="BTC/USD",
                    trading_pair_id=1,
                    coin=CoinResponse(id=1, symbol="BTC", name="Bitcoin"),
                    quote=CoinResponse(id=2, symbol="USD", name="US Dollar"),
                    exchange=ExchangeResponse(id=1, slug="hyperliquid", name="Hyperliquid"),
                    market_type=MarketTypeResponse(id=1, type="perps", name="Perpetuals"),
                    interval="15m",
                    interval_seconds=900,
                    is_active=True,
                    created_at=1700000000000,
                    updated_at=1700000000000,
                )
            ],
            total=1,
        )
        assert len(arena_list.arenas) == 1
        assert arena_list.total == 1

    def test_empty_arena_list(self):
        """Test empty arena list response."""
        arena_list = ArenaListResponse(arenas=[], total=0)
        assert len(arena_list.arenas) == 0
        assert arena_list.total == 0


class TestArenaSyncRequest:
    """Tests for ArenaSyncRequest schema."""

    def test_valid_sync_request(self):
        """Test valid sync request."""
        request = ArenaSyncRequest(starlisting_id=123)
        assert request.starlisting_id == 123

    def test_sync_request_missing_starlisting_id(self):
        """Test sync request with missing starlisting_id."""
        with pytest.raises(ValidationError):
            ArenaSyncRequest()


class TestLookupListResponses:
    """Tests for lookup table list responses."""

    def test_exchange_list_response(self):
        """Test exchange list response."""
        response = ExchangeListResponse(
            exchanges=[
                ExchangeResponse(id=1, slug="hyperliquid", name="Hyperliquid"),
                ExchangeResponse(id=2, slug="binance", name="Binance"),
            ],
            total=2,
        )
        assert len(response.exchanges) == 2
        assert response.total == 2

    def test_coin_list_response(self):
        """Test coin list response."""
        response = CoinListResponse(
            coins=[
                CoinResponse(id=1, symbol="BTC", name="Bitcoin"),
                CoinResponse(id=2, symbol="USD", name="US Dollar"),
                CoinResponse(id=3, symbol="SOL", name="Solana"),
            ],
            total=3,
        )
        assert len(response.coins) == 3
        assert response.total == 3

    def test_market_type_list_response(self):
        """Test market type list response."""
        response = MarketTypeListResponse(
            market_types=[
                MarketTypeResponse(id=1, type="perps", name="Perpetuals"),
                MarketTypeResponse(id=2, type="spot", name="Spot"),
            ],
            total=2,
        )
        assert len(response.market_types) == 2
        assert response.total == 2
