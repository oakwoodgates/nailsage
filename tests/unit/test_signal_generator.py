"""Unit tests for SignalGenerator."""

import pytest
from datetime import datetime

from execution.inference.signal_generator import SignalGenerator, SignalGeneratorConfig
from execution.inference.predictor import Prediction
from execution.portfolio.signal import StrategySignal


class TestSignalGeneratorConfig:
    """Test suite for SignalGeneratorConfig validation."""

    def test_config_valid(self):
        """Test config initialization with valid parameters."""
        config = SignalGeneratorConfig(
            strategy_name="test_strategy",
            asset="BTC/USDT",
            confidence_threshold=0.6,
            position_size_pct=10.0,  # 10% of bankroll
            cooldown_bars=4,
        )
        assert config.strategy_name == "test_strategy"
        assert config.asset == "BTC/USDT"
        assert config.confidence_threshold == 0.6
        assert config.position_size_pct == 10.0
        assert config.cooldown_bars == 4

    def test_config_default_position_size_pct(self):
        """Test default position size is 10%."""
        config = SignalGeneratorConfig(
            strategy_name="test",
            asset="BTC/USDT",
        )
        assert config.position_size_pct == 10.0

    def test_config_invalid_confidence_too_high(self):
        """Test config validation with confidence > 1.0."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            SignalGeneratorConfig(
                strategy_name="test",
                asset="BTC/USDT",
                confidence_threshold=1.5,
            )

    def test_config_invalid_confidence_negative(self):
        """Test config validation with negative confidence."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
            SignalGeneratorConfig(
                strategy_name="test",
                asset="BTC/USDT",
                confidence_threshold=-0.1,
            )

    def test_config_invalid_position_size_pct_zero(self):
        """Test config validation with zero position size percentage."""
        with pytest.raises(ValueError, match="position_size_pct must be between 0 and 100"):
            SignalGeneratorConfig(
                strategy_name="test",
                asset="BTC/USDT",
                position_size_pct=0.0,
            )

    def test_config_invalid_position_size_pct_negative(self):
        """Test config validation with negative position size percentage."""
        with pytest.raises(ValueError, match="position_size_pct must be between 0 and 100"):
            SignalGeneratorConfig(
                strategy_name="test",
                asset="BTC/USDT",
                position_size_pct=-10.0,
            )

    def test_config_invalid_position_size_pct_over_100(self):
        """Test config validation with position size percentage over 100."""
        with pytest.raises(ValueError, match="position_size_pct must be between 0 and 100"):
            SignalGeneratorConfig(
                strategy_name="test",
                asset="BTC/USDT",
                position_size_pct=150.0,
            )

    def test_config_invalid_cooldown_negative(self):
        """Test config validation with negative cooldown."""
        with pytest.raises(ValueError, match="cooldown_bars must be non-negative"):
            SignalGeneratorConfig(
                strategy_name="test",
                asset="BTC/USDT",
                cooldown_bars=-1,
            )


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""

    @pytest.fixture
    def config(self):
        """Create default config for testing."""
        return SignalGeneratorConfig(
            strategy_name="test_strategy",
            asset="BTC/USDT",
            confidence_threshold=0.6,
            position_size_pct=10.0,  # 10% of bankroll
            cooldown_bars=4,
        )

    @pytest.fixture
    def generator(self, config):
        """Create signal generator for testing."""
        return SignalGenerator(config)

    @pytest.fixture
    def prediction_long(self):
        """Create a 3-class long prediction."""
        return Prediction(
            timestamp=1700000000000,  # Unix ms
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test_model_123",
            prediction=2,  # Long (3-class)
            confidence=0.75,
            probabilities={"short": 0.10, "neutral": 0.15, "long": 0.75},
        )

    @pytest.fixture
    def prediction_short(self):
        """Create a 3-class short prediction."""
        return Prediction(
            timestamp=1700000900000,  # 15 minutes later
            datetime=datetime(2023, 11, 14, 22, 28, 20),
            model_id="test_model_123",
            prediction=0,  # Short (3-class)
            confidence=0.80,
            probabilities={"short": 0.80, "neutral": 0.10, "long": 0.10},
        )

    @pytest.fixture
    def prediction_neutral(self):
        """Create a 3-class neutral prediction."""
        return Prediction(
            timestamp=1700001800000,  # 30 minutes after first
            datetime=datetime(2023, 11, 14, 22, 43, 20),
            model_id="test_model_123",
            prediction=1,  # Neutral (3-class)
            confidence=0.70,
            probabilities={"short": 0.15, "neutral": 0.70, "long": 0.15},
        )

    @pytest.fixture
    def prediction_low_confidence(self):
        """Create a low confidence prediction."""
        return Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test_model_123",
            prediction=2,  # Long
            confidence=0.45,  # Below 0.6 threshold
            probabilities={"short": 0.30, "neutral": 0.25, "long": 0.45},
        )

    # Binary (2-class) prediction fixtures
    @pytest.fixture
    def prediction_binary_long(self):
        """Create a binary (2-class) long prediction."""
        return Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="binary_model",
            prediction=1,  # Long (binary: 0=short, 1=long)
            confidence=0.80,
            probabilities={"short": 0.20, "neutral": 0.0, "long": 0.80},  # neutral=0 indicates binary
        )

    @pytest.fixture
    def prediction_binary_short(self):
        """Create a binary (2-class) short prediction."""
        return Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="binary_model",
            prediction=0,  # Short (binary: 0=short, 1=long)
            confidence=0.75,
            probabilities={"short": 0.75, "neutral": 0.0, "long": 0.25},  # neutral=0 indicates binary
        )

    # 5-class prediction fixtures
    @pytest.fixture
    def prediction_5class_strong_long(self):
        """Create a 5-class strong long prediction."""
        return Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="5class_model",
            prediction=4,  # Strong Long (5-class)
            confidence=0.70,
            probabilities={"strong_short": 0.05, "weak_short": 0.05, "neutral": 0.10, "weak_long": 0.10, "strong_long": 0.70},
        )

    @pytest.fixture
    def prediction_5class_weak_short(self):
        """Create a 5-class weak short prediction."""
        return Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="5class_model",
            prediction=1,  # Weak Short (5-class)
            confidence=0.65,
            probabilities={"strong_short": 0.10, "weak_short": 0.65, "neutral": 0.15, "weak_long": 0.05, "strong_long": 0.05},
        )

    @pytest.fixture
    def prediction_5class_neutral(self):
        """Create a 5-class neutral prediction."""
        return Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="5class_model",
            prediction=2,  # Neutral (5-class)
            confidence=0.60,
            probabilities={"strong_short": 0.10, "weak_short": 0.10, "neutral": 0.60, "weak_long": 0.10, "strong_long": 0.10},
        )

    # Test: Confidence Filtering
    def test_generate_signal_high_confidence(self, generator, prediction_long):
        """Test that high confidence prediction generates signal."""
        signal = generator.generate_signal(prediction_long, current_bankroll=10000.0)

        assert signal is not None
        assert isinstance(signal, StrategySignal)
        assert signal.signal == 1  # Long
        assert signal.confidence == 0.75
        assert signal.strategy_name == "test_strategy"
        assert signal.asset == "BTC/USDT"
        # Position size = 10% of $10,000 bankroll = $1,000
        assert signal.position_size_usd == 1000.0

    def test_generate_signal_position_size_scales_with_bankroll(self, generator, prediction_long):
        """Test that position size scales with current bankroll."""
        # With $5,000 bankroll, 10% = $500
        signal_small = generator.generate_signal(prediction_long, current_bankroll=5000.0)
        assert signal_small.position_size_usd == 500.0

        generator.reset()  # Reset to avoid deduplication

        # With $20,000 bankroll, 10% = $2,000
        signal_large = generator.generate_signal(prediction_long, current_bankroll=20000.0)
        assert signal_large.position_size_usd == 2000.0

    def test_generate_signal_low_confidence(self, generator, prediction_low_confidence):
        """Test that low confidence prediction is filtered out."""
        signal = generator.generate_signal(prediction_low_confidence)
        assert signal is None

    def test_generate_signal_exactly_at_threshold(self, generator):
        """Test prediction exactly at confidence threshold."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test_model_123",
            prediction=2,
            confidence=0.6,  # Exactly at threshold
            probabilities={"short": 0.2, "neutral": 0.2, "long": 0.6},
        )
        signal = generator.generate_signal(prediction)
        assert signal is not None  # Should pass (>=)

    # Test: Signal Deduplication
    def test_deduplication_same_signal(self, generator, prediction_long):
        """Test that duplicate signals are suppressed."""
        # First signal should pass
        signal1 = generator.generate_signal(prediction_long)
        assert signal1 is not None
        assert signal1.signal == 1  # Long

        # Second identical signal should be suppressed
        prediction_long2 = Prediction(
            timestamp=1700000900000,  # 15 min later
            datetime=datetime(2023, 11, 14, 22, 28, 20),
            model_id="test_model_123",
            prediction=2,  # Long (same)
            confidence=0.80,
            probabilities={"short": 0.10, "neutral": 0.10, "long": 0.80},
        )
        signal2 = generator.generate_signal(prediction_long2, candle_interval_ms=900000)
        assert signal2 is None  # Duplicate

    def test_deduplication_different_signal(self, generator, prediction_long, prediction_short):
        """Test that different signals are not deduplicated."""
        # First signal: Long
        signal1 = generator.generate_signal(prediction_long, candle_interval_ms=900000)
        assert signal1 is not None
        assert signal1.signal == 1

        # Second signal: Short (different, but within cooldown)
        # Move timestamp to be outside cooldown (4 bars = 60 minutes)
        prediction_short_later = Prediction(
            timestamp=1700003600000,  # 60 minutes later (4 bars)
            datetime=datetime(2023, 11, 14, 23, 13, 20),
            model_id="test_model_123",
            prediction=0,  # Short
            confidence=0.80,
            probabilities={"short": 0.80, "neutral": 0.10, "long": 0.10},
        )
        signal2 = generator.generate_signal(prediction_short_later, candle_interval_ms=900000)
        assert signal2 is not None
        assert signal2.signal == -1  # Short

    # Test: Cooldown Mechanism
    def test_cooldown_blocks_signal(self, generator, prediction_long):
        """Test that cooldown period blocks new signals."""
        # First signal
        signal1 = generator.generate_signal(prediction_long, candle_interval_ms=900000)
        assert signal1 is not None

        # Second signal 2 bars later (within cooldown of 4 bars)
        prediction_2bars = Prediction(
            timestamp=1700001800000,  # 30 minutes = 2 bars
            datetime=datetime(2023, 11, 14, 22, 43, 20),
            model_id="test_model_123",
            prediction=0,  # Different direction
            confidence=0.80,
            probabilities={"short": 0.80, "neutral": 0.10, "long": 0.10},
        )
        signal2 = generator.generate_signal(prediction_2bars, candle_interval_ms=900000)
        assert signal2 is None  # Blocked by cooldown

    def test_cooldown_allows_after_period(self, generator, prediction_long):
        """Test that signals are allowed after cooldown period."""
        # First signal
        signal1 = generator.generate_signal(prediction_long, candle_interval_ms=900000)
        assert signal1 is not None

        # Second signal 4 bars later (exactly at cooldown threshold)
        prediction_4bars = Prediction(
            timestamp=1700003600000,  # 60 minutes = 4 bars
            datetime=datetime(2023, 11, 14, 23, 13, 20),
            model_id="test_model_123",
            prediction=0,  # Different direction
            confidence=0.80,
            probabilities={"short": 0.80, "neutral": 0.10, "long": 0.10},
        )
        signal2 = generator.generate_signal(prediction_4bars, candle_interval_ms=900000)
        assert signal2 is not None  # Should pass (>=)

    def test_cooldown_zero_bars(self):
        """Test that cooldown_bars=0 allows immediate signals."""
        config = SignalGeneratorConfig(
            strategy_name="test",
            asset="BTC/USDT",
            cooldown_bars=0,  # No cooldown
        )
        generator = SignalGenerator(config)

        # First signal
        prediction1 = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=2,
            confidence=0.75,
            probabilities={"short": 0.1, "neutral": 0.15, "long": 0.75},
        )
        signal1 = generator.generate_signal(prediction1)
        assert signal1 is not None

        # Immediate second signal (different direction)
        prediction2 = Prediction(
            timestamp=1700000000000,  # Same timestamp
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=0,
            confidence=0.80,
            probabilities={"short": 0.8, "neutral": 0.1, "long": 0.1},
        )
        signal2 = generator.generate_signal(prediction2)
        assert signal2 is not None  # No cooldown

    # Test: Neutral Signal Handling
    def test_neutral_signal_allowed_by_default(self, generator, prediction_neutral):
        """Test that neutral signals are allowed by default."""
        signal = generator.generate_signal(prediction_neutral)
        assert signal is not None
        assert signal.signal == 0  # Neutral

    def test_neutral_signal_suppressed_when_disabled(self, prediction_neutral):
        """Test that neutral signals can be suppressed."""
        config = SignalGeneratorConfig(
            strategy_name="test",
            asset="BTC/USDT",
            allow_neutral_signals=False,
        )
        generator = SignalGenerator(config)

        signal = generator.generate_signal(prediction_neutral)
        assert signal is None  # Suppressed

    # Test: Prediction to Signal Mapping
    def test_prediction_to_signal_short(self, generator):
        """Test prediction class 0 maps to signal -1 (short)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=0,  # Short
            confidence=0.75,
            probabilities={"short": 0.75, "neutral": 0.15, "long": 0.10},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == -1

    def test_prediction_to_signal_neutral(self, generator):
        """Test prediction class 1 maps to signal 0 (neutral)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=1,  # Neutral
            confidence=0.75,
            probabilities={"short": 0.15, "neutral": 0.75, "long": 0.10},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == 0

    def test_prediction_to_signal_long(self, generator):
        """Test prediction class 2 maps to signal 1 (long)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="test",
            prediction=2,  # Long
            confidence=0.75,
            probabilities={"short": 0.10, "neutral": 0.15, "long": 0.75},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == 1

    # Test: Stats and State Management
    def test_stats_initial(self, generator):
        """Test initial stats are correct."""
        stats = generator.get_stats()
        assert stats["signals_generated"] == 0
        assert stats["last_signal"] is None
        assert stats["last_signal_timestamp"] is None
        assert stats["strategy_name"] == "test_strategy"
        assert stats["asset"] == "BTC/USDT"
        assert stats["confidence_threshold"] == 0.6
        assert stats["position_size_pct"] == 10.0

    def test_stats_after_signal(self, generator, prediction_long):
        """Test stats are updated after generating signal."""
        generator.generate_signal(prediction_long)

        stats = generator.get_stats()
        assert stats["signals_generated"] == 1
        assert stats["last_signal"] == 1  # Long
        assert stats["last_signal_timestamp"] == 1700000000000

    def test_stats_counts_multiple_signals(self, generator, prediction_long):
        """Test stats count multiple signals correctly."""
        # Generate first signal
        generator.generate_signal(prediction_long, candle_interval_ms=900000)

        # Generate second signal (after cooldown)
        prediction2 = Prediction(
            timestamp=1700003600000,  # 60 min later
            datetime=datetime(2023, 11, 14, 23, 13, 20),
            model_id="test",
            prediction=0,  # Different
            confidence=0.80,
            probabilities={"short": 0.80, "neutral": 0.10, "long": 0.10},
        )
        generator.generate_signal(prediction2, candle_interval_ms=900000)

        stats = generator.get_stats()
        assert stats["signals_generated"] == 2

    def test_reset_clears_state(self, generator, prediction_long):
        """Test reset clears generator state."""
        # Generate signal
        generator.generate_signal(prediction_long)

        # Verify state is set
        stats = generator.get_stats()
        assert stats["signals_generated"] == 1
        assert stats["last_signal"] == 1

        # Reset
        generator.reset()

        # Verify state is cleared (but count remains)
        stats = generator.get_stats()
        assert stats["signals_generated"] == 1  # Count not reset
        assert stats["last_signal"] is None
        assert stats["last_signal_timestamp"] is None

    # Test: Edge Cases
    def test_first_signal_no_deduplication(self, generator, prediction_long):
        """Test first signal is not deduplicated (no previous signal)."""
        signal = generator.generate_signal(prediction_long)
        assert signal is not None

    def test_first_signal_no_cooldown(self, generator, prediction_long):
        """Test first signal is not blocked by cooldown (no previous signal)."""
        signal = generator.generate_signal(prediction_long)
        assert signal is not None

    def test_bars_since_last_signal_no_previous(self, generator):
        """Test bars calculation with no previous signal."""
        bars = generator._bars_since_last_signal(
            current_timestamp=1700000000000,
            candle_interval_ms=900000
        )
        assert bars == float('inf')

    def test_bars_since_last_signal_calculation(self, generator, prediction_long):
        """Test bars calculation is correct."""
        # Generate first signal
        generator.generate_signal(prediction_long, candle_interval_ms=900000)

        # Calculate bars 30 minutes later (2 bars)
        bars = generator._bars_since_last_signal(
            current_timestamp=1700001800000,  # +30 min
            candle_interval_ms=900000  # 15 min per bar
        )
        assert bars == 2

    def test_different_candle_intervals(self, generator):
        """Test cooldown works with different candle intervals."""
        config = SignalGeneratorConfig(
            strategy_name="test",
            asset="SOL/USDT",
            cooldown_bars=4,
        )
        generator = SignalGenerator(config)

        # First signal
        prediction1 = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 0, 0),
            model_id="test",
            prediction=2,
            confidence=0.75,
            probabilities={"short": 0.1, "neutral": 0.15, "long": 0.75},
        )
        signal1 = generator.generate_signal(prediction1, candle_interval_ms=14400000)  # 4H
        assert signal1 is not None

        # Second signal 3 bars later (should be blocked)
        prediction2 = Prediction(
            timestamp=1700043200000,  # +12 hours = 3 bars
            datetime=datetime(2023, 11, 15, 10, 0, 0),
            model_id="test",
            prediction=0,
            confidence=0.80,
            probabilities={"short": 0.8, "neutral": 0.1, "long": 0.1},
        )
        signal2 = generator.generate_signal(prediction2, candle_interval_ms=14400000)
        assert signal2 is None  # Blocked (3 < 4)

        # Third signal 4 bars later (should pass)
        prediction3 = Prediction(
            timestamp=1700057600000,  # +16 hours = 4 bars
            datetime=datetime(2023, 11, 15, 14, 0, 0),
            model_id="test",
            prediction=0,
            confidence=0.80,
            probabilities={"short": 0.8, "neutral": 0.1, "long": 0.1},
        )
        signal3 = generator.generate_signal(prediction3, candle_interval_ms=14400000)
        assert signal3 is not None  # Allowed (4 >= 4)

    # ===========================================================================
    # Multi-Mode Classification Tests
    # ===========================================================================

    # Test: Binary (2-class) Classification
    def test_binary_mode_detection(self, generator, prediction_binary_long):
        """Test binary mode is detected from neutral=0.0."""
        num_classes = generator._detect_num_classes(prediction_binary_long)
        assert num_classes == 2

    def test_binary_long_signal(self, generator, prediction_binary_long):
        """Test binary prediction=1 maps to long signal (+1)."""
        signal = generator.generate_signal(prediction_binary_long)
        assert signal is not None
        assert signal.signal == 1  # Long

    def test_binary_short_signal(self, generator, prediction_binary_short):
        """Test binary prediction=0 maps to short signal (-1)."""
        signal = generator.generate_signal(prediction_binary_short)
        assert signal is not None
        assert signal.signal == -1  # Short

    def test_binary_no_neutral_possible(self, generator):
        """Test binary mode never produces neutral signal."""
        # In binary mode, there's no class that maps to neutral
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="binary_model",
            prediction=0,
            confidence=0.70,
            probabilities={"short": 0.70, "neutral": 0.0, "long": 0.30},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal != 0  # Never neutral in binary mode

    # Test: 3-class Classification (default)
    def test_3class_mode_detection(self, generator, prediction_long):
        """Test 3-class mode is detected (default)."""
        num_classes = generator._detect_num_classes(prediction_long)
        assert num_classes == 3

    def test_3class_short_signal(self, generator):
        """Test 3-class prediction=0 maps to short signal (-1)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="3class_model",
            prediction=0,  # Short
            confidence=0.75,
            probabilities={"short": 0.75, "neutral": 0.15, "long": 0.10},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == -1

    def test_3class_neutral_signal(self, generator):
        """Test 3-class prediction=1 maps to neutral signal (0)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="3class_model",
            prediction=1,  # Neutral
            confidence=0.75,
            probabilities={"short": 0.10, "neutral": 0.75, "long": 0.15},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == 0

    def test_3class_long_signal(self, generator):
        """Test 3-class prediction=2 maps to long signal (+1)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="3class_model",
            prediction=2,  # Long
            confidence=0.75,
            probabilities={"short": 0.10, "neutral": 0.15, "long": 0.75},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == 1

    # Test: 5-class Classification
    def test_5class_mode_detection(self, generator, prediction_5class_strong_long):
        """Test 5-class mode is detected from probability keys."""
        num_classes = generator._detect_num_classes(prediction_5class_strong_long)
        assert num_classes == 5

    def test_5class_strong_short_signal(self, generator):
        """Test 5-class prediction=0 (strong short) maps to short signal (-1)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="5class_model",
            prediction=0,  # Strong short
            confidence=0.70,
            probabilities={"strong_short": 0.70, "weak_short": 0.10, "neutral": 0.10, "weak_long": 0.05, "strong_long": 0.05},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == -1

    def test_5class_weak_short_signal(self, generator, prediction_5class_weak_short):
        """Test 5-class prediction=1 (weak short) maps to short signal (-1)."""
        signal = generator.generate_signal(prediction_5class_weak_short)
        assert signal is not None
        assert signal.signal == -1  # Weak short still maps to short

    def test_5class_neutral_signal(self, generator, prediction_5class_neutral):
        """Test 5-class prediction=2 (neutral) maps to neutral signal (0)."""
        signal = generator.generate_signal(prediction_5class_neutral)
        assert signal is not None
        assert signal.signal == 0

    def test_5class_weak_long_signal(self, generator):
        """Test 5-class prediction=3 (weak long) maps to long signal (+1)."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="5class_model",
            prediction=3,  # Weak long
            confidence=0.65,
            probabilities={"strong_short": 0.05, "weak_short": 0.05, "neutral": 0.15, "weak_long": 0.65, "strong_long": 0.10},
        )
        signal = generator.generate_signal(prediction)
        assert signal.signal == 1  # Weak long maps to long

    def test_5class_strong_long_signal(self, generator, prediction_5class_strong_long):
        """Test 5-class prediction=4 (strong long) maps to long signal (+1)."""
        signal = generator.generate_signal(prediction_5class_strong_long)
        assert signal is not None
        assert signal.signal == 1  # Strong long maps to long

    def test_5class_detection_by_prediction_value(self, generator):
        """Test 5-class is detected when prediction >= 4 even without special keys."""
        prediction = Prediction(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 22, 13, 20),
            model_id="5class_model",
            prediction=4,  # Class 4 indicates 5-class
            confidence=0.70,
            probabilities={"short": 0.05, "neutral": 0.25, "long": 0.70},  # Generic keys
        )
        num_classes = generator._detect_num_classes(prediction)
        assert num_classes == 5
