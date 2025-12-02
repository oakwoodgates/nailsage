"""Health check endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_state_manager
from execution.persistence.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    checks: dict
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint.

    Returns service health status.
    """
    try:
        state_manager = get_state_manager()
        engine = state_manager._get_engine()

        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return HealthResponse(
            status="healthy",
            database="connected",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database=f"error: {str(e)}",
            timestamp=datetime.now().isoformat(),
        )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Readiness check endpoint.

    Verifies all required services are ready.
    """
    checks = {
        "database": False,
        "tables": False,
    }

    try:
        state_manager = get_state_manager()
        engine = state_manager._get_engine()

        from sqlalchemy import text, inspect

        # Check database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            checks["database"] = True

        # Check tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        required_tables = ["strategies", "positions", "trades", "signals"]
        checks["tables"] = all(t in tables for t in required_tables)

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")

    ready = all(checks.values())

    return ReadyResponse(
        ready=ready,
        checks=checks,
        timestamp=datetime.now().isoformat(),
    )
