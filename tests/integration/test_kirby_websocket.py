"""Quick test to verify historical candle messages from Kirby."""

import asyncio
import json
import logging

import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


import pytest


@pytest.mark.skip("Requires live Kirby websocket; skipped in CI.")
async def test_historical_candles():
    """Test connection and log all incoming messages."""
    url = "ws://localhost:8000/ws?api_key=kb_abc9fca6194efa30f1c1b36b875fd6e43b897ec6"

    logger.info(f"Connecting to {url}")

    async with websockets.connect(url) as ws:
        logger.info("Connected!")

        # Subscribe to starlisting 2 with 10 historical candles
        subscribe_msg = {
            "action": "subscribe",
            "starlisting_ids": [2],
            "history": 10
        }

        logger.info(f"Sending: {subscribe_msg}")
        await ws.send(json.dumps(subscribe_msg))

        # Receive messages for 30 seconds
        message_count = 0
        historical_count = 0
        candle_count = 0

        try:
            for _ in range(100):  # Read up to 100 messages
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                message_count += 1

                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "historical":
                    historical_count += 1
                    # data might be a dict or list
                    if isinstance(data.get("data"), dict):
                        candle_time = data.get("data", {}).get("time", "?")
                        logger.info(f"[{message_count}] HISTORICAL candle: {candle_time}")
                    else:
                        logger.info(f"[{message_count}] HISTORICAL: {json.dumps(data)[:300]}")

                elif msg_type == "candle":
                    candle_count += 1
                    if isinstance(data.get("data"), dict):
                        candle_time = data.get("data", {}).get("time", "?")
                        logger.info(f"[{message_count}] LIVE candle: {candle_time}")
                    else:
                        logger.info(f"[{message_count}] LIVE: {json.dumps(data)[:300]}")

                else:
                    logger.info(f"[{message_count}] {msg_type}: {json.dumps(data)[:200]}")

        except asyncio.TimeoutError:
            logger.info("No more messages (timeout)")

        logger.info("=" * 70)
        logger.info(f"Summary:")
        logger.info(f"  Total messages: {message_count}")
        logger.info(f"  Historical candles: {historical_count}")
        logger.info(f"  Live candles: {candle_count}")
        logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_historical_candles())
