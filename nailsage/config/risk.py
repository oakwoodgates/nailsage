"""Risk management configuration."""

from typing import Optional

from pydantic import Field, field_validator

from nailsage.config.base import BaseConfig


class RiskConfig(BaseConfig):
    """
    Configuration for risk management.

    Defines position sizing, leverage limits, and risk controls.
    """

    # Position sizing
    position_size_method: str = Field(
        default="fixed_fractional",
        description="Position sizing method (fixed_fractional, kelly, confidence_weighted)",
    )
    position_size_pct: float = Field(
        default=0.1,  # 10% of capital per trade
        description="Position size as percentage of capital (for fixed_fractional)",
        gt=0.0,
        le=1.0,
    )

    # Leverage
    max_leverage: float = Field(
        default=1.0,
        description="Maximum leverage allowed (1.0 = no leverage)",
        ge=1.0,
        le=100.0,
    )
    leverage_per_strategy: Optional[dict[str, float]] = Field(
        default=None,
        description="Strategy-specific leverage limits",
    )

    # Risk limits
    max_positions: int = Field(
        default=5,
        description="Maximum number of concurrent positions",
        ge=1,
        le=100,
    )
    max_exposure: float = Field(
        default=1.0,
        description="Maximum total exposure as fraction of capital",
        gt=0.0,
        le=10.0,
    )
    max_correlation: float = Field(
        default=0.7,
        description="Maximum correlation between positions (0-1)",
        ge=0.0,
        le=1.0,
    )

    # Stop loss / Take profit
    use_stop_loss: bool = Field(
        default=False,
        description="Enable stop loss",
    )
    stop_loss_pct: float = Field(
        default=0.02,  # 2% stop loss
        description="Stop loss as percentage (e.g., 0.02 = 2%)",
        gt=0.0,
        le=0.5,
    )
    use_take_profit: bool = Field(
        default=False,
        description="Enable take profit",
    )
    take_profit_pct: float = Field(
        default=0.05,  # 5% take profit
        description="Take profit as percentage (e.g., 0.05 = 5%)",
        gt=0.0,
        le=2.0,
    )

    # Risk per trade
    max_risk_per_trade_pct: float = Field(
        default=0.02,  # 2% of capital
        description="Maximum risk per trade as percentage of capital",
        gt=0.0,
        le=0.2,  # Max 20%
    )

    # Portfolio-level risk
    max_drawdown_stop_pct: float = Field(
        default=0.20,  # 20% drawdown
        description="Stop trading if drawdown exceeds this percentage",
        gt=0.0,
        le=0.5,
    )

    @field_validator("position_size_method")
    @classmethod
    def validate_position_size_method(cls, v: str) -> str:
        """Validate position sizing method."""
        allowed = ["fixed_fractional", "kelly", "confidence_weighted"]
        if v not in allowed:
            raise ValueError(f"position_size_method must be one of {allowed}")
        return v

    @field_validator("take_profit_pct")
    @classmethod
    def validate_take_profit_vs_stop_loss(cls, v: float, info) -> float:
        """Ensure take profit is larger than stop loss."""
        stop_loss = info.data.get("stop_loss_pct")
        if stop_loss and v <= stop_loss:
            raise ValueError("take_profit_pct must be greater than stop_loss_pct")
        return v

    def get_strategy_leverage(self, strategy_name: str) -> float:
        """
        Get leverage limit for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Leverage limit (defaults to max_leverage if not specified)
        """
        if self.leverage_per_strategy and strategy_name in self.leverage_per_strategy:
            return self.leverage_per_strategy[strategy_name]
        return self.max_leverage

    def calculate_position_size(
        self,
        capital: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Calculate position size based on configuration.

        Args:
            capital: Available capital
            confidence: Model confidence (0-1, used for confidence_weighted)

        Returns:
            Position size in capital units
        """
        if self.position_size_method == "fixed_fractional":
            return capital * self.position_size_pct

        elif self.position_size_method == "confidence_weighted":
            return capital * self.position_size_pct * confidence

        elif self.position_size_method == "kelly":
            # Simplified Kelly criterion (would need win rate and payoff ratio)
            # For now, just use fixed fractional
            return capital * self.position_size_pct

        return capital * self.position_size_pct
