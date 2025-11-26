"""Unit tests for CandleCloseDetector."""

import pytest

from execution.runner.candle_close_detector import CandleCloseDetector, CandleCloseEvent


class TestCandleCloseDetector:
    """Tests for CandleCloseDetector."""

    def test_init(self):
        """Test initialization."""
        detector = CandleCloseDetector()

        assert detector.get_last_timestamp() is None
        assert detector.get_candles_seen() == 0

    def test_first_candle(self):
        """Test processing first candle."""
        detector = CandleCloseDetector()

        event = detector.process_candle(timestamp=1000)

        assert event is not None
        assert event.is_first_candle is True
        assert event.previous_timestamp is None
        assert event.current_timestamp == 1000
        assert detector.get_candles_seen() == 1

    def test_same_candle_forming(self):
        """Test same candle updating (no close event)."""
        detector = CandleCloseDetector()

        # First candle
        event1 = detector.process_candle(timestamp=1000)
        assert event1 is not None

        # Same candle (still forming)
        event2 = detector.process_candle(timestamp=1000)
        assert event2 is None  # No close event

        # Should not increment candles_seen
        assert detector.get_candles_seen() == 1

    def test_candle_close_detected(self):
        """Test candle close detection when timestamp changes."""
        detector = CandleCloseDetector()

        # First candle
        event1 = detector.process_candle(timestamp=1000)
        assert event1.is_first_candle is True

        # Same candle updating
        event2 = detector.process_candle(timestamp=1000)
        assert event2 is None

        # New candle (previous one closed!)
        event3 = detector.process_candle(timestamp=2000)

        assert event3 is not None
        assert event3.is_first_candle is False
        assert event3.previous_timestamp == 1000
        assert event3.current_timestamp == 2000
        assert detector.get_candles_seen() == 2

    def test_multiple_candle_closes(self):
        """Test multiple candle closes in sequence."""
        detector = CandleCloseDetector()

        timestamps = [1000, 1000, 1000, 2000, 2000, 3000, 4000]
        events = [detector.process_candle(ts) for ts in timestamps]

        # First candle (1000)
        assert events[0] is not None
        assert events[0].is_first_candle is True

        # Same candle updates
        assert events[1] is None
        assert events[2] is None

        # Candle close (1000 -> 2000)
        assert events[3] is not None
        assert events[3].previous_timestamp == 1000
        assert events[3].current_timestamp == 2000

        # Same candle update
        assert events[4] is None

        # Candle close (2000 -> 3000)
        assert events[5] is not None
        assert events[5].previous_timestamp == 2000
        assert events[5].current_timestamp == 3000

        # Candle close (3000 -> 4000)
        assert events[6] is not None
        assert events[6].previous_timestamp == 3000
        assert events[6].current_timestamp == 4000

        assert detector.get_candles_seen() == 4

    def test_timestamp_backward_raises_error(self):
        """Test that timestamp going backward raises error."""
        detector = CandleCloseDetector()

        # Process normal sequence
        detector.process_candle(timestamp=3000)
        detector.process_candle(timestamp=4000)

        # Timestamp goes backward (data issue)
        with pytest.raises(ValueError, match="Timestamp went backward"):
            detector.process_candle(timestamp=2000)

    def test_reset(self):
        """Test resetting detector."""
        detector = CandleCloseDetector()

        # Process some candles
        detector.process_candle(timestamp=1000)
        detector.process_candle(timestamp=2000)
        detector.process_candle(timestamp=3000)

        assert detector.get_candles_seen() == 3
        assert detector.get_last_timestamp() == 3000

        # Reset
        detector.reset()

        assert detector.get_candles_seen() == 0
        assert detector.get_last_timestamp() is None

        # Should work like new after reset
        event = detector.process_candle(timestamp=5000)
        assert event.is_first_candle is True

    def test_large_timestamp_gap(self):
        """Test handling large gaps in timestamps (missed candles)."""
        detector = CandleCloseDetector()

        detector.process_candle(timestamp=1000)

        # Large gap (e.g., missed many candles due to reconnection)
        event = detector.process_candle(timestamp=10000)

        assert event is not None
        assert event.previous_timestamp == 1000
        assert event.current_timestamp == 10000
        # Should still detect as candle close
        assert not event.is_first_candle

    def test_repr(self):
        """Test string representation."""
        detector = CandleCloseDetector()

        detector.process_candle(timestamp=1000)
        detector.process_candle(timestamp=2000)

        repr_str = repr(detector)

        assert "CandleCloseDetector" in repr_str
        assert "last_timestamp=2000" in repr_str
        assert "candles_seen=2" in repr_str

    def test_realistic_trading_scenario(self):
        """Test realistic trading scenario with candle updates."""
        detector = CandleCloseDetector()

        # Simulate 15-minute candle updates
        # Candle at 09:00 updates multiple times
        events = []
        for i in range(5):
            event = detector.process_candle(timestamp=900_000)
            events.append(event)

        # First update creates event, rest are None
        assert events[0] is not None
        assert all(e is None for e in events[1:])

        # New candle at 09:15 (900 seconds = 900000 ms)
        event = detector.process_candle(timestamp=1_800_000)

        assert event is not None
        assert event.previous_timestamp == 900_000
        assert event.current_timestamp == 1_800_000
        assert detector.get_candles_seen() == 2
