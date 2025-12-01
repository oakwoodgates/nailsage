"""Dependency injection for API endpoints."""

import logging
from functools import lru_cache
from typing import Generator

from execution.persistence.state_manager import StateManager

from api.config import get_config, APIConfig
from api.services.strategy_service import StrategyService
from api.services.trade_service import TradeService
from api.services.position_service import PositionService
from api.services.stats_service import StatsService
from api.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)

# Global state manager instance
_state_manager: StateManager | None = None


def get_state_manager() -> StateManager:
    """Get or create the global StateManager instance.

    Returns:
        StateManager instance
    """
    global _state_manager
    if _state_manager is None:
        config = get_config()
        _state_manager = StateManager(database_url=config.database_url)
        logger.info("StateManager initialized")
    return _state_manager


def close_state_manager() -> None:
    """Close the global StateManager instance."""
    global _state_manager
    if _state_manager is not None:
        _state_manager.close()
        _state_manager = None
        logger.info("StateManager closed")


# Service dependencies

def get_strategy_service() -> StrategyService:
    """Get StrategyService instance.

    Returns:
        StrategyService
    """
    return StrategyService(get_state_manager())


def get_trade_service() -> TradeService:
    """Get TradeService instance.

    Returns:
        TradeService
    """
    return TradeService(get_state_manager())


def get_position_service() -> PositionService:
    """Get PositionService instance.

    Returns:
        PositionService
    """
    return PositionService(get_state_manager())


def get_stats_service() -> StatsService:
    """Get StatsService instance.

    Returns:
        StatsService
    """
    config = get_config()
    return StatsService(
        get_state_manager(),
        initial_capital=config.initial_capital,
    )


def get_portfolio_service() -> PortfolioService:
    """Get PortfolioService instance.

    Returns:
        PortfolioService
    """
    config = get_config()
    return PortfolioService(
        get_state_manager(),
        initial_capital=config.initial_capital,
    )
