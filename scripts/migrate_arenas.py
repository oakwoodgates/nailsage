#!/usr/bin/env python
"""
Migration script to populate arena tables from existing strategy starlisting IDs.

This script:
1. Queries all unique starlisting_ids from the strategies table
2. For each starlisting_id, fetches metadata from Kirby API
3. Creates lookup records (exchanges, coins, market_types)
4. Creates arena records with FK references
5. Updates strategies.arena_id with the new arena IDs

Usage:
    python scripts/migrate_arenas.py

Environment variables required:
    DATABASE_URL: PostgreSQL connection string
    KIRBY_WS_URL: Kirby WebSocket URL (will be converted to REST)
    KIRBY_API_KEY: Kirby API key
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

import httpx
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_or_create_lookup(conn, table: str, key_column: str, key_value: str, extra_data: dict, now: int) -> int:
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

    # Insert based on table type
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

    new_id = result.fetchone()[0]
    logger.info(f"Created {table} record: {key_value} (id={new_id})")
    return new_id


async def migrate_arena(
    engine,
    client: httpx.AsyncClient,
    kirby_base_url: str,
    kirby_api_key: str,
    starlisting_id: int,
) -> int | None:
    """Migrate a single starlisting to an arena.

    Args:
        engine: SQLAlchemy engine
        client: HTTP client
        kirby_base_url: Kirby API base URL
        kirby_api_key: Kirby API key
        starlisting_id: Kirby starlisting ID

    Returns:
        Arena ID if successful, None otherwise
    """
    now = int(datetime.now().timestamp() * 1000)

    # Fetch from Kirby
    try:
        response = await client.get(
            f"{kirby_base_url}/starlistings/{starlisting_id}",
            headers={"Authorization": f"Bearer {kirby_api_key}"},
        )

        if response.status_code == 404:
            logger.warning(f"Starlisting {starlisting_id} not found in Kirby - skipping")
            return None

        if response.status_code != 200:
            logger.error(f"Kirby API error for starlisting {starlisting_id}: {response.status_code}")
            return None

        data = response.json()

    except Exception as e:
        logger.error(f"Error fetching starlisting {starlisting_id}: {e}")
        return None

    # Insert arena and lookup records
    with engine.begin() as conn:
        # Upsert exchange
        exchange_id = get_or_create_lookup(
            conn, "exchanges", "slug",
            data.get("exchange", ""),
            {"display_name": data.get("exchange_display", data.get("exchange", ""))},
            now,
        )

        # Upsert coin (base asset)
        coin_id = get_or_create_lookup(
            conn, "coins", "symbol",
            data.get("coin", ""),
            {"name": data.get("coin_name", data.get("coin", ""))},
            now,
        )

        # Upsert quote asset
        quote_id = get_or_create_lookup(
            conn, "coins", "symbol",
            data.get("quote", ""),
            {"name": data.get("quote_name", data.get("quote", ""))},
            now,
        )

        # Upsert market type
        market_type_id = get_or_create_lookup(
            conn, "market_types", "type",
            data.get("market_type", ""),
            {"display": data.get("market_type_display", data.get("market_type", ""))},
            now,
        )

        # Check if arena already exists
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
            logger.info(f"Updated arena {arena_id} for starlisting {starlisting_id}")
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
            logger.info(f"Created arena {arena_id} for starlisting {starlisting_id}: {data.get('trading_pair')} on {data.get('exchange')}")

        # Update strategies with this arena_id
        result = conn.execute(
            text("""
                UPDATE strategies
                SET arena_id = :arena_id
                WHERE starlisting_id = :starlisting_id AND arena_id IS NULL
            """),
            {"arena_id": arena_id, "starlisting_id": starlisting_id},
        )
        if result.rowcount > 0:
            logger.info(f"Updated {result.rowcount} strategies with arena_id={arena_id}")

    return arena_id


async def main():
    """Run the arena migration."""
    database_url = os.getenv("DATABASE_URL")
    kirby_ws_url = os.getenv("KIRBY_WS_URL") or os.getenv("KIRBY_WS_URL_PRO")
    kirby_api_key = os.getenv("KIRBY_API_KEY") or os.getenv("KIRBY_API_KEY_PRO")

    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    if not kirby_ws_url or not kirby_api_key:
        logger.error("KIRBY_WS_URL and KIRBY_API_KEY environment variables required")
        sys.exit(1)

    # Build Kirby base URL
    kirby_base_url = kirby_ws_url.replace("wss://", "https://").replace("ws://", "http://")
    if kirby_base_url.endswith("/ws"):
        kirby_base_url = kirby_base_url[:-3]

    logger.info(f"Kirby API base URL: {kirby_base_url}")

    engine = create_engine(database_url)

    # Get all unique starlisting_ids from strategies
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT starlisting_id FROM strategies"))
        starlisting_ids = [row[0] for row in result.fetchall()]

    logger.info(f"Found {len(starlisting_ids)} unique starlisting IDs to migrate")

    if not starlisting_ids:
        logger.info("No starlisting IDs found - nothing to migrate")
        return

    # Migrate each starlisting
    success_count = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for starlisting_id in starlisting_ids:
            arena_id = await migrate_arena(
                engine, client, kirby_base_url, kirby_api_key, starlisting_id
            )
            if arena_id:
                success_count += 1

    logger.info(f"Migration complete: {success_count}/{len(starlisting_ids)} arenas created/updated")

    # Print summary
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM arenas"))
        arena_count = result.scalar()

        result = conn.execute(text("SELECT COUNT(*) FROM exchanges"))
        exchange_count = result.scalar()

        result = conn.execute(text("SELECT COUNT(*) FROM coins"))
        coin_count = result.scalar()

        result = conn.execute(text("SELECT COUNT(*) FROM market_types"))
        market_type_count = result.scalar()

    logger.info(f"Database summary:")
    logger.info(f"  - Arenas: {arena_count}")
    logger.info(f"  - Exchanges: {exchange_count}")
    logger.info(f"  - Coins: {coin_count}")
    logger.info(f"  - Market Types: {market_type_count}")


if __name__ == "__main__":
    asyncio.run(main())
