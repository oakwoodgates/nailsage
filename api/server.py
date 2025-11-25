"""FastAPI server for Nailsage dashboard backend.

Provides REST API and WebSocket endpoints for:
- Strategy management
- Trade history
- Position tracking
- Performance leaderboards
- Real-time updates

Environment Variables:
    DATABASE_URL: Database connection string
    POLL_INTERVAL: WebSocket polling interval in seconds (default: 2)
    LOG_LEVEL: Logging level (default: INFO)
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Nailsage Trading API",
    description="REST API and WebSocket for trading dashboard",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
logger.info(f"Connected to database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'SQLite'}")

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()


# Pydantic models for API responses
class StrategyResponse(BaseModel):
    id: int
    strategy_name: str
    version: str
    starlisting_id: int
    interval: str
    model_id: Optional[str]
    is_active: bool
    created_at: int
    updated_at: int


class TradeResponse(BaseModel):
    id: int
    position_id: int
    strategy_id: int
    starlisting_id: int
    trade_type: str
    size: float
    price: float
    fees: float
    slippage: float
    timestamp: int
    created_at: int


class PositionResponse(BaseModel):
    id: int
    strategy_id: int
    starlisting_id: int
    side: str
    size: float
    entry_price: float
    entry_timestamp: int
    exit_price: Optional[float]
    exit_timestamp: Optional[int]
    realized_pnl: Optional[float]
    unrealized_pnl: Optional[float]
    fees_paid: float
    status: str
    exit_reason: Optional[str]


class LeaderboardEntry(BaseModel):
    strategy_id: int
    strategy_name: str
    total_trades: int
    total_pnl: float
    win_rate: float
    avg_profit: float
    avg_loss: float


# ==============================================================================
# Health Check
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# ==============================================================================
# Strategy Endpoints
# ==============================================================================

@app.get("/strategies", response_model=List[StrategyResponse])
async def get_strategies(active_only: bool = Query(True, description="Filter to active strategies only")):
    """Get all strategies."""
    try:
        with engine.connect() as conn:
            if active_only:
                result = conn.execute(text("SELECT * FROM strategies WHERE is_active = TRUE ORDER BY created_at DESC"))
            else:
                result = conn.execute(text("SELECT * FROM strategies ORDER BY created_at DESC"))

            strategies = []
            for row in result:
                strategies.append(StrategyResponse(
                    id=row.id,
                    strategy_name=row.strategy_name,
                    version=row.version,
                    starlisting_id=row.starlisting_id,
                    interval=row.interval,
                    model_id=row.model_id,
                    is_active=bool(row.is_active),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                ))

            return strategies
    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: int):
    """Get strategy by ID."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM strategies WHERE id = :id"),
                {"id": strategy_id}
            )
            row = result.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Strategy not found")

            return StrategyResponse(
                id=row.id,
                strategy_name=row.strategy_name,
                version=row.version,
                starlisting_id=row.starlisting_id,
                interval=row.interval,
                model_id=row.model_id,
                is_active=bool(row.is_active),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Trade Endpoints
# ==============================================================================

@app.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    limit: int = Query(100, ge=1, le=1000, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Get trades with optional filtering."""
    try:
        with engine.connect() as conn:
            if strategy_id:
                result = conn.execute(
                    text("""
                        SELECT * FROM trades
                        WHERE strategy_id = :strategy_id
                        ORDER BY timestamp DESC
                        LIMIT :limit OFFSET :offset
                    """),
                    {"strategy_id": strategy_id, "limit": limit, "offset": offset}
                )
            else:
                result = conn.execute(
                    text("""
                        SELECT * FROM trades
                        ORDER BY timestamp DESC
                        LIMIT :limit OFFSET :offset
                    """),
                    {"limit": limit, "offset": offset}
                )

            trades = []
            for row in result:
                trades.append(TradeResponse(
                    id=row.id,
                    position_id=row.position_id,
                    strategy_id=row.strategy_id,
                    starlisting_id=row.starlisting_id,
                    trade_type=row.trade_type,
                    size=row.size,
                    price=row.price,
                    fees=row.fees,
                    slippage=row.slippage,
                    timestamp=row.timestamp,
                    created_at=row.created_at,
                ))

            return trades
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Position Endpoints
# ==============================================================================

@app.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    strategy_id: Optional[int] = Query(None, description="Filter by strategy ID"),
    status: Optional[str] = Query(None, description="Filter by status (open/closed)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of positions to return"),
):
    """Get positions with optional filtering."""
    try:
        with engine.connect() as conn:
            query = "SELECT * FROM positions WHERE 1=1"
            params = {"limit": limit}

            if strategy_id:
                query += " AND strategy_id = :strategy_id"
                params["strategy_id"] = strategy_id

            if status:
                query += " AND status = :status"
                params["status"] = status

            query += " ORDER BY entry_timestamp DESC LIMIT :limit"

            result = conn.execute(text(query), params)

            positions = []
            for row in result:
                positions.append(PositionResponse(
                    id=row.id,
                    strategy_id=row.strategy_id,
                    starlisting_id=row.starlisting_id,
                    side=row.side,
                    size=row.size,
                    entry_price=row.entry_price,
                    entry_timestamp=row.entry_timestamp,
                    exit_price=row.exit_price,
                    exit_timestamp=row.exit_timestamp,
                    realized_pnl=row.realized_pnl,
                    unrealized_pnl=row.unrealized_pnl,
                    fees_paid=row.fees_paid,
                    status=row.status,
                    exit_reason=row.exit_reason,
                ))

            return positions
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Leaderboard Endpoint
# ==============================================================================

@app.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(metric: str = Query("total_pnl", description="Metric to rank by (total_pnl, win_rate)")):
    """Get strategy leaderboard ranked by performance metric."""
    try:
        with engine.connect() as conn:
            # Calculate strategy statistics
            result = conn.execute(text("""
                SELECT
                    s.id as strategy_id,
                    s.strategy_name,
                    COUNT(t.id) as total_trades,
                    COALESCE(SUM(p.realized_pnl), 0) as total_pnl,
                    CAST(SUM(CASE WHEN p.realized_pnl > 0 THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(p.id), 0) as win_rate,
                    AVG(CASE WHEN p.realized_pnl > 0 THEN p.realized_pnl ELSE NULL END) as avg_profit,
                    AVG(CASE WHEN p.realized_pnl < 0 THEN p.realized_pnl ELSE NULL END) as avg_loss
                FROM strategies s
                LEFT JOIN trades t ON s.id = t.strategy_id
                LEFT JOIN positions p ON s.id = p.strategy_id AND p.status = 'closed'
                WHERE s.is_active = TRUE
                GROUP BY s.id, s.strategy_name
                ORDER BY {} DESC
            """.format("total_pnl" if metric == "total_pnl" else "win_rate")))

            leaderboard = []
            for row in result:
                leaderboard.append(LeaderboardEntry(
                    strategy_id=row.strategy_id,
                    strategy_name=row.strategy_name,
                    total_trades=row.total_trades or 0,
                    total_pnl=row.total_pnl or 0.0,
                    win_rate=row.win_rate or 0.0,
                    avg_profit=row.avg_profit or 0.0,
                    avg_loss=row.avg_loss or 0.0,
                ))

            return leaderboard
    except Exception as e:
        logger.error(f"Error generating leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Search Endpoint
# ==============================================================================

@app.get("/search")
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=100),
):
    """Search strategies, trades, and positions by metadata."""
    try:
        with engine.connect() as conn:
            # Search strategies
            strategies = conn.execute(
                text("""
                    SELECT 'strategy' as type, id, strategy_name as name, NULL as details
                    FROM strategies
                    WHERE strategy_name LIKE :query OR model_id LIKE :query
                    LIMIT :limit
                """),
                {"query": f"%{query}%", "limit": limit}
            ).fetchall()

            # Search trades
            trades = conn.execute(
                text("""
                    SELECT 'trade' as type, id, trade_type as name, strategy_id as details
                    FROM trades
                    WHERE trade_type LIKE :query
                    LIMIT :limit
                """),
                {"query": f"%{query}%", "limit": limit}
            ).fetchall()

            results = []
            for row in strategies:
                results.append({"type": row.type, "id": row.id, "name": row.name})
            for row in trades:
                results.append({"type": row.type, "id": row.id, "name": row.name, "strategy_id": row.details})

            return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# WebSocket Endpoint
# ==============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live updates."""
    await manager.connect(websocket)

    try:
        # Send initial data
        await websocket.send_json({"type": "connected", "message": "Connected to Nailsage API"})

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for client messages (e.g., subscription requests)
            data = await websocket.receive_text()
            logger.debug(f"Received from client: {data}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket")


# ==============================================================================
# Background Tasks
# ==============================================================================

async def poll_database():
    """Poll database for new data and broadcast to WebSocket clients."""
    last_trade_id = 0
    last_position_id = 0
    poll_interval = int(os.getenv('POLL_INTERVAL', '2'))

    while True:
        try:
            await asyncio.sleep(poll_interval)

            if not manager.active_connections:
                continue

            with engine.connect() as conn:
                # Check for new trades
                result = conn.execute(
                    text("SELECT * FROM trades WHERE id > :last_id ORDER BY id"),
                    {"last_id": last_trade_id}
                )
                new_trades = result.fetchall()

                if new_trades:
                    trades_data = []
                    for row in new_trades:
                        trades_data.append({
                            "id": row.id,
                            "strategy_id": row.strategy_id,
                            "trade_type": row.trade_type,
                            "size": row.size,
                            "price": row.price,
                            "timestamp": row.timestamp,
                        })
                        last_trade_id = row.id

                    await manager.broadcast({
                        "type": "new_trades",
                        "data": trades_data,
                    })

                # Check for new/updated positions
                result = conn.execute(
                    text("SELECT * FROM positions WHERE id > :last_id OR updated_at > :last_check ORDER BY id"),
                    {"last_id": last_position_id, "last_check": int((datetime.now().timestamp() - poll_interval) * 1000)}
                )
                updated_positions = result.fetchall()

                if updated_positions:
                    positions_data = []
                    for row in updated_positions:
                        positions_data.append({
                            "id": row.id,
                            "strategy_id": row.strategy_id,
                            "status": row.status,
                            "unrealized_pnl": row.unrealized_pnl,
                            "realized_pnl": row.realized_pnl,
                        })
                        if row.id > last_position_id:
                            last_position_id = row.id

                    await manager.broadcast({
                        "type": "position_updates",
                        "data": positions_data,
                    })

        except Exception as e:
            logger.error(f"Error in database polling: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    logger.info("Starting Nailsage API server...")
    asyncio.create_task(poll_database())
    logger.info("Background database polling started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Nailsage API server...")
    engine.dispose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
