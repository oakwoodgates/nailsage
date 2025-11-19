"""Paper trading configuration for live execution module.

This module provides configuration classes for:
- WebSocket connection to Kirby API
- Starlisting ID mappings
- Paper trading parameters
- State persistence settings

Configuration is loaded from environment variables using pydantic-settings.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.base import BaseConfig


class WebSocketConfig(BaseConfig):
    """
    WebSocket connection configuration for Kirby API.

    Attributes:
        url: WebSocket URL (ws:// or wss://)
        api_key: Kirby API key (starts with kb_)
        reconnect_enabled: Enable automatic reconnection
        reconnect_max_attempts: Maximum reconnection attempts (0 = infinite)
        reconnect_initial_delay: Initial delay before reconnection (seconds)
        reconnect_max_delay: Maximum delay between reconnections (seconds)
        reconnect_backoff_multiplier: Exponential backoff multiplier
        heartbeat_interval: Expected heartbeat interval (seconds)
        heartbeat_timeout: Timeout before reconnection (seconds)
        historical_candles: Number of historical candles to request on connect
    """

    url: str = Field(..., description="WebSocket URL (ws:// or wss://)")
    api_key: str = Field(..., description="Kirby API key (starts with kb_)")

    # Reconnection settings (industry standard exponential backoff)
    reconnect_enabled: bool = Field(default=True, description="Enable automatic reconnection")
    reconnect_max_attempts: int = Field(default=0, description="Max attempts (0 = infinite)")
    reconnect_initial_delay: float = Field(default=1.0, description="Initial delay (seconds)")
    reconnect_max_delay: float = Field(default=60.0, description="Max delay (seconds)")
    reconnect_backoff_multiplier: float = Field(default=2.0, description="Backoff multiplier")

    # Heartbeat settings
    heartbeat_interval: int = Field(default=30, description="Expected heartbeat interval (seconds)")
    heartbeat_timeout: int = Field(default=45, description="Timeout before reconnection (seconds)")

    # Historical data
    historical_candles: int = Field(default=200, description="Historical candles on connect")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate WebSocket URL format."""
        if not v.startswith(("ws://", "wss://")):
            raise ValueError("WebSocket URL must start with ws:// or wss://")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if not v.startswith("kb_"):
            raise ValueError("Kirby API key must start with kb_")
        if len(v) < 10:  # kb_ + at least 7 chars
            raise ValueError("API key appears invalid (too short)")
        return v


class StarlistingMapping(BaseConfig):
    """
    Mapping of (symbol, interval) to Kirby starlisting_id.

    This is a temporary hardcoded solution until Kirby provides
    an API endpoint to query starlisting IDs by symbol/interval.

    Attributes:
        btc_usdt_15m: BTC/USDT 15-minute starlisting ID
        sol_usdt_4h: SOL/USDT 4-hour starlisting ID
    """

    btc_usdt_15m: int = Field(..., description="BTC/USDT 15m starlisting ID")
    sol_usdt_4h: int = Field(..., description="SOL/USDT 4h starlisting ID")

    def get_starlisting_id(self, symbol: str, interval: str) -> Optional[int]:
        """
        Get starlisting ID for a given symbol and interval.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            interval: Timeframe (e.g., "15m", "4h")

        Returns:
            Starlisting ID if found, None otherwise
        """
        key = f"{symbol.lower().replace('/', '_')}_{interval.lower()}"
        mapping = {
            "btc_usdt_15m": self.btc_usdt_15m,
            "sol_usdt_4h": self.sol_usdt_4h,
        }
        return mapping.get(key)


class PaperTradingConfig(BaseSettings):
    """
    Main paper trading configuration loaded from environment variables.

    This class uses pydantic-settings to automatically load values from
    environment variables or .env file.

    Attributes:
        ws_url: WebSocket URL for Kirby API
        ws_api_key: API key for authentication
        starlisting_btc_usdt_15m: BTC/USDT 15m starlisting ID
        starlisting_sol_usdt_4h: SOL/USDT 4h starlisting ID
        initial_capital: Starting capital for paper trading (USDT)
        db_path: Path to SQLite database for state persistence
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        snapshot_interval: Interval for saving state snapshots (seconds)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kirby API settings
    kirby_ws_url: str = Field(..., alias="KIRBY_WS_URL")
    kirby_api_key: str = Field(..., alias="KIRBY_API_KEY")

    # Starlisting ID mappings
    starlisting_btc_usdt_15m: int = Field(..., alias="STARLISTING_BTC_USDT_15M")
    starlisting_sol_usdt_4h: int = Field(..., alias="STARLISTING_SOL_USDT_4H")

    # Paper trading settings
    paper_trading_initial_capital: float = Field(
        default=100000.0,
        alias="PAPER_TRADING_INITIAL_CAPITAL"
    )
    paper_trading_db_path: str = Field(
        default="execution/state/paper_trading.db",
        alias="PAPER_TRADING_DB_PATH"
    )
    paper_trading_log_level: str = Field(
        default="INFO",
        alias="PAPER_TRADING_LOG_LEVEL"
    )
    paper_trading_snapshot_interval: int = Field(
        default=60,
        alias="PAPER_TRADING_SNAPSHOT_INTERVAL"
    )

    @property
    def websocket(self) -> WebSocketConfig:
        """Get WebSocket configuration."""
        return WebSocketConfig(
            url=self.kirby_ws_url,
            api_key=self.kirby_api_key,
        )

    @property
    def starlistings(self) -> StarlistingMapping:
        """Get starlisting ID mappings."""
        return StarlistingMapping(
            btc_usdt_15m=self.starlisting_btc_usdt_15m,
            sol_usdt_4h=self.starlisting_sol_usdt_4h,
        )

    @property
    def db_path_obj(self) -> Path:
        """Get database path as Path object."""
        return Path(self.paper_trading_db_path)

    @property
    def log_level_obj(self) -> int:
        """Get logging level as integer constant."""
        return getattr(logging, self.paper_trading_log_level.upper())

    @field_validator("paper_trading_initial_capital")
    @classmethod
    def validate_capital(cls, v: float) -> float:
        """Validate initial capital is positive."""
        if v <= 0:
            raise ValueError("Initial capital must be positive")
        return v

    @field_validator("paper_trading_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("paper_trading_snapshot_interval")
    @classmethod
    def validate_snapshot_interval(cls, v: int) -> int:
        """Validate snapshot interval is reasonable."""
        if v < 1:
            raise ValueError("Snapshot interval must be at least 1 second")
        if v > 3600:
            raise ValueError("Snapshot interval should not exceed 1 hour")
        return v


def load_paper_trading_config(env_file: Optional[Path] = None) -> PaperTradingConfig:
    """
    Load paper trading configuration from environment.

    Args:
        env_file: Optional path to .env file (defaults to .env in project root)

    Returns:
        Validated paper trading configuration

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If .env file doesn't exist

    Example:
        >>> config = load_paper_trading_config()
        >>> print(config.websocket.url)
        ws://143.198.18.115:8000/ws
        >>> starlisting_id = config.starlistings.get_starlisting_id("BTC/USDT", "15m")
        >>> print(starlisting_id)
        1
    """
    if env_file:
        return PaperTradingConfig(_env_file=str(env_file))
    else:
        return PaperTradingConfig()
