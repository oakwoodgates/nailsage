"""Arena service for business logic and Kirby API integration."""

import logging
from typing import List, Optional, Tuple

import httpx
from sqlalchemy import text

from execution.persistence.state_manager import StateManager
from api.config import get_config, APIConfig
from api.schemas.arenas import (
    ArenaResponse,
    ArenaSummary,
    ExchangeResponse,
    CoinResponse,
    MarketTypeResponse,
)

logger = logging.getLogger(__name__)


class ArenaService:
    """Service for arena-related operations."""

    def __init__(self, state_manager: StateManager):
        """Initialize arena service.

        Args:
            state_manager: Database state manager
        """
        self.state_manager = state_manager
        self.config = get_config()

    # =========================================================================
    # Lookup Table Methods
    # =========================================================================

    def get_all_exchanges(self) -> List[ExchangeResponse]:
        """Get all exchanges."""
        engine = self.state_manager._get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM exchanges ORDER BY slug")
            )
            return [
                ExchangeResponse(
                    id=row["id"],
                    slug=row["slug"],
                    name=row["display_name"],
                )
                for row in result.mappings()
            ]

    def get_all_coins(self) -> List[CoinResponse]:
        """Get all coins."""
        engine = self.state_manager._get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM coins ORDER BY symbol")
            )
            return [
                CoinResponse(
                    id=row["id"],
                    symbol=row["symbol"],
                    name=row["name"],
                )
                for row in result.mappings()
            ]

    def get_all_market_types(self) -> List[MarketTypeResponse]:
        """Get all market types."""
        engine = self.state_manager._get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM market_types ORDER BY type")
            )
            return [
                MarketTypeResponse(
                    id=row["id"],
                    type=row["type"],
                    name=row["display"],
                )
                for row in result.mappings()
            ]

    # =========================================================================
    # Arena Methods
    # =========================================================================

    def get_all_arenas(
        self,
        active_only: bool = True,
        exchange_id: Optional[int] = None,
        coin_id: Optional[int] = None,
    ) -> List[ArenaResponse]:
        """Get all arenas with optional filters.

        Args:
            active_only: If True, return only active arenas
            exchange_id: Filter by exchange ID
            coin_id: Filter by coin ID (base asset)

        Returns:
            List of arena responses
        """
        engine = self.state_manager._get_engine()

        # Build query with JOINs
        query = """
            SELECT
                a.*,
                c.id as coin_id, c.symbol as coin_symbol, c.name as coin_name,
                q.id as quote_id, q.symbol as quote_symbol, q.name as quote_name,
                e.id as exchange_id, e.slug as exchange_slug, e.display_name as exchange_display,
                mt.id as market_type_id, mt.type as market_type_type, mt.display as market_type_display
            FROM arenas a
            JOIN coins c ON a.coin_id = c.id
            JOIN coins q ON a.quote_id = q.id
            JOIN exchanges e ON a.exchange_id = e.id
            JOIN market_types mt ON a.market_type_id = mt.id
            WHERE 1=1
        """
        params = {}

        if active_only:
            query += " AND a.is_active = TRUE"

        if exchange_id is not None:
            query += " AND a.exchange_id = :exchange_id"
            params["exchange_id"] = exchange_id

        if coin_id is not None:
            query += " AND a.coin_id = :coin_id"
            params["coin_id"] = coin_id

        query += " ORDER BY a.trading_pair, a.interval"

        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [self._row_to_arena_response(row) for row in result.mappings()]

    def get_arena_by_id(self, arena_id: int) -> Optional[ArenaResponse]:
        """Get arena by ID with JOINs to lookup tables.

        Args:
            arena_id: Arena ID

        Returns:
            Arena response or None if not found
        """
        engine = self.state_manager._get_engine()

        query = """
            SELECT
                a.*,
                c.id as coin_id, c.symbol as coin_symbol, c.name as coin_name,
                q.id as quote_id, q.symbol as quote_symbol, q.name as quote_name,
                e.id as exchange_id, e.slug as exchange_slug, e.display_name as exchange_display,
                mt.id as market_type_id, mt.type as market_type_type, mt.display as market_type_display
            FROM arenas a
            JOIN coins c ON a.coin_id = c.id
            JOIN coins q ON a.quote_id = q.id
            JOIN exchanges e ON a.exchange_id = e.id
            JOIN market_types mt ON a.market_type_id = mt.id
            WHERE a.id = :arena_id
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"arena_id": arena_id})
            row = result.mappings().fetchone()

        return self._row_to_arena_response(row) if row else None

    def get_arena_by_starlisting(self, starlisting_id: int) -> Optional[ArenaResponse]:
        """Get arena by Kirby starlisting ID.

        Args:
            starlisting_id: Kirby starlisting ID

        Returns:
            Arena response or None if not found
        """
        engine = self.state_manager._get_engine()

        query = """
            SELECT
                a.*,
                c.id as coin_id, c.symbol as coin_symbol, c.name as coin_name,
                q.id as quote_id, q.symbol as quote_symbol, q.name as quote_name,
                e.id as exchange_id, e.slug as exchange_slug, e.display_name as exchange_display,
                mt.id as market_type_id, mt.type as market_type_type, mt.display as market_type_display
            FROM arenas a
            JOIN coins c ON a.coin_id = c.id
            JOIN coins q ON a.quote_id = q.id
            JOIN exchanges e ON a.exchange_id = e.id
            JOIN market_types mt ON a.market_type_id = mt.id
            WHERE a.starlisting_id = :starlisting_id
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"starlisting_id": starlisting_id})
            row = result.mappings().fetchone()

        return self._row_to_arena_response(row) if row else None

    def get_arena_summary(self, arena_id: int) -> Optional[ArenaSummary]:
        """Get lightweight arena summary for embedding.

        Args:
            arena_id: Arena ID

        Returns:
            Arena summary or None if not found
        """
        engine = self.state_manager._get_engine()

        query = """
            SELECT
                a.id, a.starlisting_id, a.trading_pair, a.interval,
                c.symbol as coin, q.symbol as quote,
                e.slug as exchange, mt.type as market_type
            FROM arenas a
            JOIN coins c ON a.coin_id = c.id
            JOIN coins q ON a.quote_id = q.id
            JOIN exchanges e ON a.exchange_id = e.id
            JOIN market_types mt ON a.market_type_id = mt.id
            WHERE a.id = :arena_id
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"arena_id": arena_id})
            row = result.mappings().fetchone()

        if not row:
            return None

        return ArenaSummary(
            id=row["id"],
            starlisting_id=row["starlisting_id"],
            trading_pair=row["trading_pair"],
            interval=row["interval"],
            coin=row["coin"],
            quote=row["quote"],
            exchange=row["exchange"],
            market_type=row["market_type"],
        )

    # =========================================================================
    # Kirby API Sync Methods
    # =========================================================================

    async def sync_from_kirby(self, starlisting_id: int) -> Tuple[ArenaResponse, bool]:
        """Fetch metadata from Kirby API and create/update local arena.

        Args:
            starlisting_id: Kirby starlisting ID

        Returns:
            Tuple of (arena response, was_created)

        Raises:
            HTTPException: If Kirby API request fails
        """
        from fastapi import HTTPException

        if not self.config.kirby_ws_url or not self.config.kirby_api_key:
            raise HTTPException(
                status_code=503,
                detail="Kirby API not configured - cannot sync arena metadata",
            )

        # Build Kirby API URL
        kirby_base_url = self.config.kirby_ws_url.replace("wss://", "https://").replace("ws://", "http://")
        if kirby_base_url.endswith("/ws"):
            kirby_base_url = kirby_base_url[:-3]

        kirby_url = f"{kirby_base_url}/starlistings/{starlisting_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    kirby_url,
                    headers={"Authorization": f"Bearer {self.config.kirby_api_key}"},
                )

                if response.status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Starlisting {starlisting_id} not found in Kirby",
                    )

                if response.status_code != 200:
                    logger.error(f"Kirby API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=502,
                        detail="Failed to fetch starlisting from Kirby API",
                    )

                data = response.json()
                return self._upsert_from_kirby_data(starlisting_id, data)

        except httpx.TimeoutException:
            logger.error("Timeout fetching starlisting from Kirby")
            raise HTTPException(
                status_code=504,
                detail="Timeout fetching starlisting from Kirby API",
            )
        except httpx.RequestError as e:
            logger.error(f"Error fetching starlisting from Kirby: {e}")
            raise HTTPException(
                status_code=502,
                detail="Failed to connect to Kirby API",
            )

    def _upsert_from_kirby_data(self, starlisting_id: int, data: dict) -> Tuple[ArenaResponse, bool]:
        """Upsert arena and lookup records from Kirby API data.

        Args:
            starlisting_id: Kirby starlisting ID
            data: Kirby API response data

        Returns:
            Tuple of (arena response, was_created)
        """
        from datetime import datetime

        now = int(datetime.now().timestamp() * 1000)
        engine = self.state_manager._get_engine()

        with engine.begin() as conn:
            # Upsert exchange
            exchange_id = self._get_or_create_lookup(
                conn, "exchanges", "slug",
                data.get("exchange", ""),
                {"display_name": data.get("exchange_display", data.get("exchange", ""))},
                now,
            )

            # Upsert coin (base asset)
            coin_id = self._get_or_create_lookup(
                conn, "coins", "symbol",
                data.get("coin", ""),
                {"name": data.get("coin_name", data.get("coin", ""))},
                now,
            )

            # Upsert quote asset
            quote_id = self._get_or_create_lookup(
                conn, "coins", "symbol",
                data.get("quote", ""),
                {"name": data.get("quote_name", data.get("quote", ""))},
                now,
            )

            # Upsert market type
            market_type_id = self._get_or_create_lookup(
                conn, "market_types", "type",
                data.get("market_type", ""),
                {"display": data.get("market_type_display", data.get("market_type", ""))},
                now,
            )

            # Check if arena exists
            result = conn.execute(
                text("SELECT id FROM arenas WHERE starlisting_id = :starlisting_id"),
                {"starlisting_id": starlisting_id},
            )
            existing = result.fetchone()

            if existing:
                # Update existing arena
                conn.execute(
                    text("""
                        UPDATE arenas SET
                            trading_pair = :trading_pair,
                            trading_pair_id = :trading_pair_id,
                            coin_id = :coin_id,
                            quote_id = :quote_id,
                            exchange_id = :exchange_id,
                            market_type_id = :market_type_id,
                            interval = :interval,
                            interval_seconds = :interval_seconds,
                            is_active = :is_active,
                            last_synced_at = :now,
                            updated_at = :now
                        WHERE starlisting_id = :starlisting_id
                    """),
                    {
                        "starlisting_id": starlisting_id,
                        "trading_pair": data.get("trading_pair", ""),
                        "trading_pair_id": data.get("trading_pair_id"),
                        "coin_id": coin_id,
                        "quote_id": quote_id,
                        "exchange_id": exchange_id,
                        "market_type_id": market_type_id,
                        "interval": data.get("interval", ""),
                        "interval_seconds": data.get("interval_seconds", 0),
                        "is_active": data.get("active", True),
                        "now": now,
                    },
                )
                arena_id = existing[0]
                was_created = False
                logger.info(f"Updated arena {arena_id} from Kirby starlisting {starlisting_id}")
            else:
                # Insert new arena
                result = conn.execute(
                    text("""
                        INSERT INTO arenas (
                            starlisting_id, trading_pair, trading_pair_id,
                            coin_id, quote_id, exchange_id, market_type_id,
                            interval, interval_seconds, is_active,
                            last_synced_at, created_at, updated_at
                        ) VALUES (
                            :starlisting_id, :trading_pair, :trading_pair_id,
                            :coin_id, :quote_id, :exchange_id, :market_type_id,
                            :interval, :interval_seconds, :is_active,
                            :now, :now, :now
                        )
                        RETURNING id
                    """),
                    {
                        "starlisting_id": starlisting_id,
                        "trading_pair": data.get("trading_pair", ""),
                        "trading_pair_id": data.get("trading_pair_id"),
                        "coin_id": coin_id,
                        "quote_id": quote_id,
                        "exchange_id": exchange_id,
                        "market_type_id": market_type_id,
                        "interval": data.get("interval", ""),
                        "interval_seconds": data.get("interval_seconds", 0),
                        "is_active": data.get("active", True),
                        "now": now,
                    },
                )
                arena_id = result.fetchone()[0]
                was_created = True
                logger.info(f"Created arena {arena_id} from Kirby starlisting {starlisting_id}")

        # Fetch and return the full arena response
        arena = self.get_arena_by_id(arena_id)
        return arena, was_created

    def _get_or_create_lookup(
        self,
        conn,
        table: str,
        key_column: str,
        key_value: str,
        extra_data: dict,
        now: int,
    ) -> int:
        """Get or create a lookup table record.

        Args:
            conn: Database connection
            table: Table name
            key_column: Unique key column name
            key_value: Value for the key column
            extra_data: Additional columns to set
            now: Current timestamp

        Returns:
            ID of the record
        """
        # Check if exists
        result = conn.execute(
            text(f"SELECT id FROM {table} WHERE {key_column} = :key_value"),
            {"key_value": key_value},
        )
        existing = result.fetchone()

        if existing:
            return existing[0]

        # Determine columns and values based on table
        if table == "exchanges":
            result = conn.execute(
                text("""
                    INSERT INTO exchanges (slug, display_name, created_at, updated_at)
                    VALUES (:key_value, :display_name, :now, :now)
                    RETURNING id
                """),
                {"key_value": key_value, "display_name": extra_data.get("display_name", key_value), "now": now},
            )
        elif table == "coins":
            result = conn.execute(
                text("""
                    INSERT INTO coins (symbol, name, created_at, updated_at)
                    VALUES (:key_value, :name, :now, :now)
                    RETURNING id
                """),
                {"key_value": key_value, "name": extra_data.get("name", key_value), "now": now},
            )
        elif table == "market_types":
            result = conn.execute(
                text("""
                    INSERT INTO market_types (type, display, created_at, updated_at)
                    VALUES (:key_value, :display, :now, :now)
                    RETURNING id
                """),
                {"key_value": key_value, "display": extra_data.get("display", key_value), "now": now},
            )
        else:
            raise ValueError(f"Unknown lookup table: {table}")

        return result.fetchone()[0]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_arena_response(self, row) -> ArenaResponse:
        """Convert database row to ArenaResponse."""
        return ArenaResponse(
            id=row["id"],
            starlisting_id=row["starlisting_id"],
            trading_pair=row["trading_pair"],
            trading_pair_id=row["trading_pair_id"],
            coin=CoinResponse(
                id=row["coin_id"],
                symbol=row["coin_symbol"],
                name=row["coin_name"],
            ),
            quote=CoinResponse(
                id=row["quote_id"],
                symbol=row["quote_symbol"],
                name=row["quote_name"],
            ),
            exchange=ExchangeResponse(
                id=row["exchange_id"],
                slug=row["exchange_slug"],
                name=row["exchange_display"],
            ),
            market_type=MarketTypeResponse(
                id=row["market_type_id"],
                type=row["market_type_type"],
                name=row["market_type_display"],
            ),
            interval=row["interval"],
            interval_seconds=row["interval_seconds"],
            is_active=bool(row["is_active"]),
            last_synced_at=row["last_synced_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
