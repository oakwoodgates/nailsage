"""API configuration using Pydantic settings.

Environment variables can be set directly or via .env file.
"""

import os
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """API server configuration.

    All settings can be overridden via environment variables with API_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Database
    database_url: Optional[str] = Field(
        default=None,
        description="Database URL (overrides DATABASE_URL env var)"
    )

    # WebSocket settings
    ws_heartbeat_interval: int = Field(
        default=30,
        description="WebSocket heartbeat interval in seconds"
    )
    ws_max_connections: int = Field(
        default=100,
        description="Maximum WebSocket connections"
    )

    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow CORS credentials"
    )

    # Portfolio settings
    initial_capital: float = Field(
        default=100000.0,
        description="Initial capital for P&L calculations"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("database_url", mode="before")
    @classmethod
    def get_database_url(cls, v: Optional[str]) -> str:
        """Fall back to DATABASE_URL env var if not set."""
        if v:
            return v
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL environment variable is required")
        return url


# Global config instance (lazy loaded)
_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """Get API configuration singleton."""
    global _config
    if _config is None:
        _config = APIConfig()
    return _config
