"""Trading signal from a strategy."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class StrategySignal:
    """
    Trading signal emitted by a strategy.

    Represents a strategy's desired position change, including
    direction, confidence, and requested position size.

    Attributes:
        strategy_name: Name of the strategy emitting the signal
        asset: Trading pair (e.g., 'BTC/USDT', 'SOL/USDT')
        signal: Direction of signal: -1 (short), 0 (neutral/close), 1 (long)
        confidence: Model confidence in the signal (0.0 to 1.0)
        timestamp: When the signal was generated
        position_size_usd: Requested position size in USD

    Example:
        >>> signal = StrategySignal(
        ...     strategy_name='momentum_classifier_v1',
        ...     asset='BTC/USDT',
        ...     signal=1,  # long
        ...     confidence=0.85,
        ...     timestamp=datetime.now(),
        ...     position_size_usd=10000.0
        ... )
    """

    strategy_name: str
    asset: str
    signal: int  # -1 (short), 0 (neutral), 1 (long)
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    position_size_usd: float  # Requested position size

    def __post_init__(self):
        """Validate signal fields."""
        if self.signal not in (-1, 0, 1):
            raise ValueError(f"Signal must be -1, 0, or 1, got {self.signal}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.position_size_usd < 0:
            raise ValueError(f"Position size must be positive, got {self.position_size_usd}")

    @property
    def direction_name(self) -> str:
        """Human-readable direction name."""
        return {-1: "short", 0: "neutral", 1: "long"}[self.signal]

    def __repr__(self) -> str:
        """String representation of signal."""
        return (
            f"StrategySignal({self.strategy_name}, {self.asset}, "
            f"{self.direction_name}, conf={self.confidence:.2f}, "
            f"size=${self.position_size_usd:,.0f})"
        )
