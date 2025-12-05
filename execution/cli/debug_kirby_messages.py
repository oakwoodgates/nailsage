"""Debug script to see raw Kirby messages."""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import websockets
from config.paper_trading import load_paper_trading_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Connect and print first few messages."""
    config = load_paper_trading_config()
    url = f"{config.websocket.url}?api_key={config.websocket.api_key}"

    async with websockets.connect(url) as ws:
        logger.info("Connected!")

        # Subscribe
        subscribe_msg = {
            "action": "subscribe",
            "starlisting_ids": [1],
            "historical_candles": 5
        }
        await ws.send(json.dumps(subscribe_msg))
        logger.info(f"Sent: {subscribe_msg}")

        # Receive first 10 messages
        for i in range(10):
            msg = await ws.recv()
            data = json.loads(msg)
            logger.info(f"\nMessage {i+1}:")
            logger.info(json.dumps(data, indent=2))

        logger.info("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
