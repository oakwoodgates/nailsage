"""Candle close detection for live trading.

Detects when a new candle starts (previous candle closed) by tracking timestamps.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CandleCloseEvent:
    """Event indicating a candle has closed.

    Attributes:
        previous_timestamp: Timestamp of candle that just closed
        current_timestamp: Timestamp of new candle
        is_first_candle: True if this is the first candle seen
    """

    previous_timestamp: Optional[int]
    current_timestamp: int
    is_first_candle: bool


class CandleCloseDetector:
    """
    Detects candle closes by tracking timestamp changes.

    In live trading, candles update every second while forming, but we only want
    to trade when a candle CLOSES (i.e., when a new candle starts with a different
    timestamp).

    Example:
        detector = CandleCloseDetector()

        # First candle
        event = detector.process_candle(timestamp=1000)
        # event.is_first_candle = True

        # Same candle updating
        event = detector.process_candle(timestamp=1000)
        # event = None (still forming)

        # New candle (previous one closed!)
        event = detector.process_candle(timestamp=2000)
        # event.previous_timestamp = 1000, current_timestamp = 2000
    """

    def __init__(self):
        """Initialize candle close detector."""
        self._last_timestamp: Optional[int] = None
        self._candles_seen = 0

    def process_candle(self, timestamp: int) -> Optional[CandleCloseEvent]:
        """
        Process a candle timestamp and detect if previous candle closed.

        Args:
            timestamp: Current candle timestamp (Unix milliseconds)

        Returns:
            CandleCloseEvent if candle closed (or first candle), None if still forming

        Raises:
            ValueError: If timestamp goes backward (data issue)
        """
        # Validate timestamp doesn't go backward
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                f"Timestamp went backward: {self._last_timestamp} -> {timestamp}. "
                f"This indicates a data issue or WebSocket reconnection."
            )

        # First candle ever seen
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._candles_seen += 1
            logger.debug(f"First candle seen: timestamp={timestamp}")

            return CandleCloseEvent(
                previous_timestamp=None,
                current_timestamp=timestamp,
                is_first_candle=True,
            )

        # Same candle (still forming) - no close event
        if timestamp == self._last_timestamp:
            logger.debug(f"Candle still forming: timestamp={timestamp}")
            return None

        # Different timestamp = new candle started = previous candle closed!
        logger.info(
            f"🕯️  Candle closed! Previous: {self._last_timestamp}, New: {timestamp}"
        )

        event = CandleCloseEvent(
            previous_timestamp=self._last_timestamp,
            current_timestamp=timestamp,
            is_first_candle=False,
        )

        # Update state
        self._last_timestamp = timestamp
        self._candles_seen += 1

        return event

    def reset(self) -> None:
        """Reset detector state (useful for testing or reconnection)."""
        self._last_timestamp = None
        self._candles_seen = 0
        logger.info("Candle close detector reset")

    def get_last_timestamp(self) -> Optional[int]:
        """Get last seen candle timestamp."""
        return self._last_timestamp

    def get_candles_seen(self) -> int:
        """Get total number of candles seen."""
        return self._candles_seen

    def __repr__(self) -> str:
        return (
            f"CandleCloseDetector("
            f"last_timestamp={self._last_timestamp}, "
            f"candles_seen={self._candles_seen})"
        )
