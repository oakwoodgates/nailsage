"""Backtesting configuration."""

from pydantic import Field, field_validator

from config.base import BaseConfig


class BacktestConfig(BaseConfig):
    """
    Configuration for backtesting engine.

    Defines transaction costs, slippage, and execution assumptions.
    """

    # Capital and position sizing
    initial_capital: float = Field(
        default=10000.0,
        description="Initial capital for backtesting (USD)",
        gt=0.0,
    )

    # Transaction costs
    maker_fee: float = Field(
        default=0.0002,  # 0.02% - typical maker fee
        description="Maker fee as decimal (e.g., 0.0002 = 0.02%)",
        ge=0.0,
        le=0.01,  # Max 1% to catch config errors
    )
    taker_fee: float = Field(
        default=0.0004,  # 0.04% - typical taker fee
        description="Taker fee as decimal (e.g., 0.0004 = 0.04%)",
        ge=0.0,
        le=0.01,
    )
    assume_maker: bool = Field(
        default=False,
        description="Assume maker orders (if False, uses taker fee)",
    )

    # Slippage
    slippage_bps: float = Field(
        default=2.0,
        description="Slippage in basis points (1 bps = 0.01%)",
        ge=0.0,
        le=100.0,  # Max 1% slippage
    )

    # Leverage and funding (for perps)
    enable_leverage: bool = Field(
        default=False,
        description="Enable leverage trading",
    )
    max_leverage: float = Field(
        default=1.0,
        description="Maximum leverage (1.0 = no leverage)",
        ge=1.0,
        le=100.0,
    )
    funding_rate_annual: float = Field(
        default=0.0,
        description="Annual funding rate for perps (e.g., 0.05 = 5%)",
        ge=-0.5,  # Allow negative funding
        le=0.5,
    )

    # Execution assumptions
    fill_assumption: str = Field(
        default="close",
        description="Fill price assumption (close, open, high, low, midpoint)",
    )

    # Risk controls
    max_position_size: float = Field(
        default=1.0,
        description="Maximum position size as fraction of capital",
        gt=0.0,
        le=10.0,  # Allow up to 10x with leverage
    )

    @field_validator("fill_assumption")
    @classmethod
    def validate_fill_assumption(cls, v: str) -> str:
        """Validate fill assumption is a recognized value."""
        allowed = ["close", "open", "high", "low", "midpoint"]
        if v not in allowed:
            raise ValueError(f"fill_assumption must be one of {allowed}")
        return v

    @field_validator("max_leverage")
    @classmethod
    def validate_leverage_enabled(cls, v: float, info) -> float:
        """Ensure leverage is enabled if max_leverage > 1."""
        if v > 1.0 and not info.data.get("enable_leverage", False):
            raise ValueError("enable_leverage must be True if max_leverage > 1")
        return v

    def get_transaction_cost(self) -> float:
        """
        Get the transaction cost per trade based on maker/taker assumption.

        Returns:
            Transaction cost as decimal
        """
        fee = self.maker_fee if self.assume_maker else self.taker_fee
        slippage = self.slippage_bps / 10000.0  # Convert bps to decimal
        return fee + slippage

    def get_daily_funding_rate(self) -> float:
        """
        Get daily funding rate from annual rate.

        Returns:
            Daily funding rate as decimal
        """
        return self.funding_rate_annual / 365.0
