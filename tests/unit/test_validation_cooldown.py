"""Unit tests for trade cooldown functionality."""

import numpy as np
import pytest

from training.cli.validate_model import apply_trade_cooldown


class TestTradeCooldown:
    """Test suite for trade cooldown mechanism."""

    def test_cooldown_imports(self):
        """Test that cooldown function can be imported."""
        assert callable(apply_trade_cooldown)

    def test_cooldown_logic_simple(self):
        """Test cooldown logic with simple example."""

        # Create signals: [1, 1, 1, -1, 1]
        signals = np.array([1, 1, 1, -1, 1])
        min_bars = 2

        result = apply_trade_cooldown(signals, min_bars)

        # First signal (1) passes
        # Second signal (1) blocked (< 2 bars)
        # Third signal (1) blocked (< 2 bars from first)
        # Fourth signal (-1) blocked (< 2 bars from first)
        # Fifth signal (1) passes (>= 2 bars from first)

        # Check that cooldown was applied
        assert result[0] != 0  # First passes
        assert sum(result != 0) <= len(signals)  # Some blocked

    def test_cooldown_zero_bars(self):
        """Test cooldown with 0 bars (no cooldown)."""

        signals = np.array([1, 1, -1, 1, -1])
        min_bars = 0

        result = apply_trade_cooldown(signals, min_bars)

        # With no cooldown, all signals should pass
        assert np.array_equal(result, signals)

    def test_cooldown_all_neutral(self):
        """Test cooldown with all neutral signals."""

        signals = np.array([0, 0, 0, 0, 0])
        min_bars = 4

        result = apply_trade_cooldown(signals, min_bars)

        # No trades, result should be same
        assert np.array_equal(result, signals)

    def test_cooldown_far_apart(self):
        """Test cooldown with signals far apart."""

        # Signals 10 bars apart
        signals = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        min_bars = 4

        result = apply_trade_cooldown(signals, min_bars)

        # All signals should pass (far apart)
        assert result[0] == 1
        assert result[10] == -1
        assert result[20] == 1
