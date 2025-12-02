"""API routers package."""

from api.routers.health import router as health_router
from api.routers.strategies import router as strategies_router
from api.routers.trades import router as trades_router
from api.routers.positions import router as positions_router
from api.routers.portfolio import router as portfolio_router
from api.routers.stats import router as stats_router

__all__ = [
    "health_router",
    "strategies_router",
    "trades_router",
    "positions_router",
    "portfolio_router",
    "stats_router",
]
