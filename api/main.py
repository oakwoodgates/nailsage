"""FastAPI application factory for Nailsage Trading API.

This module creates and configures the FastAPI application with:
- REST API endpoints (versioned at /api/v1/)
- WebSocket endpoint for real-time updates
- CORS middleware
- Request logging
- Error handling
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_config
from api.dependencies import get_state_manager, close_state_manager
from api.routers.health import router as health_router
from api.routers.strategies import router as strategies_router
from api.routers.trades import router as trades_router
from api.routers.positions import router as positions_router
from api.routers.portfolio import router as portfolio_router
from api.routers.stats import router as stats_router
from api.routers.candles import router as candles_router
from api.routers.models import router as models_router
from api.routers.arenas import router as arenas_router
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.error_handler import ErrorHandlerMiddleware
from api.websocket.manager import get_connection_manager
from api.websocket.events import EventDispatcher
from api.websocket.handlers import WebSocketHandler
from api.websocket.poller import DatabasePoller
from api.websocket.kirby_bridge import KirbyBridge, set_kirby_bridge

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("Starting Nailsage API server...")

    # Initialize database connection
    get_state_manager()

    # Initialize WebSocket components
    connection_manager = get_connection_manager()
    event_dispatcher = EventDispatcher.get_instance()
    event_dispatcher.set_connection_manager(connection_manager)
    await event_dispatcher.start()

    # Start database poller for cross-container event emission
    state_manager = get_state_manager()
    poller = DatabasePoller(
        state_manager=state_manager,
        event_dispatcher=event_dispatcher,
        interval=3.0,
    )
    await poller.start()

    # Start KirbyBridge for price data proxy (if configured)
    config = get_config()
    kirby_bridge = None
    if config.kirby_ws_url and config.kirby_api_key:
        kirby_bridge = KirbyBridge(
            kirby_url=config.kirby_ws_url,
            kirby_api_key=config.kirby_api_key,
            connection_manager=connection_manager,
        )
        set_kirby_bridge(kirby_bridge)
        await kirby_bridge.start()
        logger.info("KirbyBridge started for price data proxy")
    else:
        logger.info("KirbyBridge not configured - price subscriptions disabled")

    logger.info("Nailsage API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Nailsage API server...")

    # Stop KirbyBridge
    if kirby_bridge:
        await kirby_bridge.stop()

    # Stop database poller
    await poller.stop()

    # Stop event dispatcher
    await event_dispatcher.stop()

    # Close database connection
    close_state_manager()

    logger.info("Nailsage API server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    config = get_config()

    app = FastAPI(
        title="Nailsage Trading API",
        description="""
## Overview

REST API and WebSocket for the Nailsage trading dashboard.

## REST Endpoints

All REST endpoints are versioned at `/api/v1/`:

- **Strategies**: `/api/v1/strategies` - Strategy management and stats
- **Models**: `/api/v1/models` - ML model metadata and results
- **Arenas**: `/api/v1/arenas` - Trading arenas (exchange + pair + interval)
- **Trades**: `/api/v1/trades` - Trade history
- **Positions**: `/api/v1/positions` - Position tracking
- **Portfolio**: `/api/v1/portfolio` - Portfolio overview
- **Stats**: `/api/v1/stats` - Financial statistics

## WebSocket

Connect to `/ws` for real-time updates. Subscribe to channels:
- `trades` - Live trade activity
- `positions` - Position changes
- `portfolio` - Portfolio metrics
- `prices` - Market data
- `signals` - Trading signals

### Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    channels: ['trades', 'positions']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.data);
};
```
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Include routers
    app.include_router(health_router)
    app.include_router(strategies_router)
    app.include_router(models_router)
    app.include_router(arenas_router)
    app.include_router(trades_router)
    app.include_router(positions_router)
    app.include_router(portfolio_router)
    app.include_router(stats_router)
    app.include_router(candles_router)

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates.

        Connect to receive live updates for:
        - Trade executions
        - Position changes
        - Portfolio metrics
        - Price updates
        - Trading signals

        Send subscription messages to control which channels you receive:
        ```json
        {"action": "subscribe", "channels": ["trades", "positions"]}
        {"action": "unsubscribe", "channels": ["prices"]}
        ```
        """
        handler = WebSocketHandler()
        await handler.handle_connection(websocket)

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )
