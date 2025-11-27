"""Ring buffer for efficient candle storage and retrieval.

This module provides a fixed-size ring buffer for storing recent candles.
It's optimized for:
- Fast insertion of new candles
- Efficient retrieval of recent N candles
- Conversion to pandas DataFrame for feature engineering
- Thread-safe operations (for use with asyncio)

The buffer automatically evicts old candles when full, maintaining
a sliding window of recent market data.
"""

import logging
from collections import deque
from datetime import datetime
from threading import Lock
from typing import List, Optional

import pandas as pd

from execution.websocket.models import Candle

logger = logging.getLogger(__name__)


class CandleBuffer:
    """
    Thread-safe ring buffer for candle storage.

    This buffer maintains a fixed-size sliding window of recent candles,
    automatically evicting old candles when the buffer is full.

    Attributes:
        maxlen: Maximum number of candles to store
        starlisting_id: Kirby starlisting ID this buffer is for
        interval: Timeframe (e.g., "15m", "4h")
        _buffer: Internal deque for O(1) append/pop operations
        _lock: Thread lock for concurrent access safety
    """

    def __init__(self, maxlen: int, starlisting_id: int, interval: str):
        """
        Initialize candle buffer.

        Args:
            maxlen: Maximum number of candles to store
            starlisting_id: Kirby starlisting ID
            interval: Timeframe (e.g., "15m", "4h")
        """
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")

        self.maxlen = maxlen
        self.starlisting_id = starlisting_id
        self.interval = interval
        self._buffer: deque[Candle] = deque(maxlen=maxlen)
        self._lock = Lock()

        logger.info(
            f"Initialized candle buffer for starlisting {starlisting_id} "
            f"({interval}) with maxlen={maxlen}"
        )

    def add(self, candle: Candle) -> None:
        """
        Add a candle to the buffer.

        If the buffer is full, the oldest candle is automatically evicted.

        Args:
            candle: Candle to add
        """
        with self._lock:
            # Check for duplicate timestamps
            if len(self._buffer) > 0 and candle.timestamp == self._buffer[-1].timestamp:
                # Update existing candle (in case of price updates on same timestamp)
                logger.debug(
                    f"Updating candle at {candle.time} "
                    f"(close: {self._buffer[-1].close:.2f} → {candle.close:.2f})"
                )
                self._buffer[-1] = candle
            else:
                # Add new candle
                self._buffer.append(candle)
                logger.debug(
                    f"Added candle: {candle.time} "
                    f"(buffer: {len(self._buffer)}/{self.maxlen})"
                )

    def add_batch(self, candles: List[Candle]) -> None:
        """
        Add multiple candles to the buffer (e.g., historical data).

        Candles should be in chronological order (oldest first).

        Args:
            candles: List of candles to add

        Raises:
            ValueError: If any candle doesn't match buffer configuration
        """
        logger.info(f"Adding batch of {len(candles)} candles to buffer")

        for candle in candles:
            self.add(candle)

        logger.info(
            f"Batch added. Buffer now contains {len(self._buffer)} candles "
            f"({self.get_oldest().datetime.isoformat()} to "
            f"{self.get_latest().datetime.isoformat()})"
        )

    def get_latest(self, n: Optional[int] = None) -> List[Candle] | Candle:
        """
        Get the most recent candle(s).

        Args:
            n: Number of recent candles to retrieve (None = single latest candle)

        Returns:
            List of candles if n is specified, single candle otherwise

        Raises:
            ValueError: If buffer is empty or n is invalid
        """
        with self._lock:
            if len(self._buffer) == 0:
                raise ValueError("Buffer is empty")

            if n is None:
                return self._buffer[-1]

            if n <= 0:
                raise ValueError("n must be positive")

            if n > len(self._buffer):
                logger.warning(
                    f"Requested {n} candles but buffer only contains {len(self._buffer)}"
                )
                n = len(self._buffer)

            # Return n most recent candles (in chronological order)
            return list(self._buffer)[-n:]

    def get_oldest(self) -> Candle:
        """
        Get the oldest candle in the buffer.

        Returns:
            Oldest candle

        Raises:
            ValueError: If buffer is empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                raise ValueError("Buffer is empty")
            return self._buffer[0]

    def get_all(self) -> List[Candle]:
        """
        Get all candles in the buffer (in chronological order).

        Returns:
            List of all candles
        """
        with self._lock:
            return list(self._buffer)

    def to_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Convert buffer to pandas DataFrame for feature engineering.

        Args:
            n: Number of recent candles to include (None = all)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, num_trades
            Index is datetime (from timestamp)
            Data is ALWAYS sorted by timestamp to ensure chronological order

        Raises:
            ValueError: If buffer is empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                raise ValueError("Buffer is empty")

            # Get candles to include
            if n is None:
                candles = list(self._buffer)
            else:
                candles = list(self._buffer)[-n:]

            # CRITICAL: Sort by timestamp to ensure chronological order
            # This is essential for correct technical indicator calculation
            # even if historical candles arrived out-of-order
            candles_sorted = sorted(candles, key=lambda c: c.timestamp)

            # Convert to DataFrame
            data = {
                "timestamp": [c.timestamp for c in candles_sorted],
                "open": [c.open for c in candles_sorted],
                "high": [c.high for c in candles_sorted],
                "low": [c.low for c in candles_sorted],
                "close": [c.close for c in candles_sorted],
                "volume": [c.volume for c in candles_sorted],
                "num_trades": [c.num_trades if c.num_trades is not None else 0 for c in candles_sorted],
            }
            df = pd.DataFrame(data)

            # Set datetime index
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")

            return df

    def clear(self) -> None:
        """Clear all candles from the buffer."""
        with self._lock:
            self._buffer.clear()
            logger.info("Buffer cleared")

    def __len__(self) -> int:
        """Get number of candles in buffer."""
        with self._lock:
            return len(self._buffer)

    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        with self._lock:
            return len(self._buffer) == self.maxlen

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    def get_time_range(self) -> tuple[datetime, datetime]:
        """
        Get time range of candles in buffer.

        Returns:
            Tuple of (oldest_datetime, latest_datetime)

        Raises:
            ValueError: If buffer is empty
        """
        with self._lock:
            if len(self._buffer) == 0:
                raise ValueError("Buffer is empty")
            return (self._buffer[0].datetime, self._buffer[-1].datetime)

    def __repr__(self) -> str:
        """Human-readable representation."""
        with self._lock:
            if len(self._buffer) == 0:
                return (
                    f"CandleBuffer(starlisting_id={self.starlisting_id}, "
                    f"interval={self.interval}, size=0/{self.maxlen}, empty)"
                )
            else:
                oldest, latest = self.get_time_range()
                return (
                    f"CandleBuffer(starlisting_id={self.starlisting_id}, "
                    f"interval={self.interval}, size={len(self._buffer)}/{self.maxlen}, "
                    f"range={oldest.isoformat()} to {latest.isoformat()})"
                )


class MultiCandleBuffer:
    """
    Manager for multiple candle buffers (one per starlisting).

    This class provides a convenient interface for managing candle buffers
    for multiple starlistings simultaneously.

    Attributes:
        maxlen: Default maximum number of candles per buffer
        _buffers: Dictionary mapping starlisting_id to CandleBuffer
    """

    def __init__(self, maxlen: int = 500):
        """
        Initialize multi-buffer manager.

        Args:
            maxlen: Default maximum number of candles per buffer
        """
        self.maxlen = maxlen
        self._buffers: dict[int, CandleBuffer] = {}
        self._lock = Lock()

    def get_or_create(self, starlisting_id: int, interval: str) -> CandleBuffer:
        """
        Get existing buffer or create new one for a starlisting.

        Args:
            starlisting_id: Kirby starlisting ID
            interval: Timeframe (e.g., "15m", "4h")

        Returns:
            CandleBuffer for the starlisting
        """
        with self._lock:
            if starlisting_id not in self._buffers:
                self._buffers[starlisting_id] = CandleBuffer(
                    maxlen=self.maxlen, starlisting_id=starlisting_id, interval=interval
                )
                logger.info(f"Created new buffer for starlisting {starlisting_id} ({interval})")

            return self._buffers[starlisting_id]

    def get(self, starlisting_id: int) -> Optional[CandleBuffer]:
        """
        Get existing buffer for a starlisting.

        Args:
            starlisting_id: Kirby starlisting ID

        Returns:
            CandleBuffer if exists, None otherwise
        """
        with self._lock:
            return self._buffers.get(starlisting_id)

    def add_candle(self, candle: Candle, starlisting_id: int, interval: str) -> None:
        """
        Add a candle to the appropriate buffer (auto-creates buffer if needed).

        Args:
            candle: Candle to add
            starlisting_id: Kirby starlisting ID
            interval: Timeframe (e.g., "15m", "4h")
        """
        buffer = self.get_or_create(starlisting_id, interval)
        buffer.add(candle)

    def __len__(self) -> int:
        """Get number of buffers."""
        with self._lock:
            return len(self._buffers)

    def __repr__(self) -> str:
        """Human-readable representation."""
        with self._lock:
            return (
                f"MultiCandleBuffer(num_buffers={len(self._buffers)}, "
                f"starlistings={list(self._buffers.keys())})"
            )
