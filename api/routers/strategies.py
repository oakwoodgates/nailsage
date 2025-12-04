"""Strategy endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_strategy_service, get_trade_service, get_position_service, get_model_service
from api.services.strategy_service import StrategyService
from api.services.trade_service import TradeService
from api.services.position_service import PositionService
from api.services.model_service import ModelService
from api.schemas.strategies import (
    StrategyResponse,
    StrategyWithStats,
    StrategyListResponse,
    BankrollResponse,
    BankrollUpdateRequest,
)
from api.schemas.trades import TradeListResponse
from api.schemas.positions import PositionListResponse
from api.schemas.models import ModelSummary, ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/strategies", tags=["Strategies"])


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    active_only: bool = Query(True, description="Filter to active strategies only"),
    strategy_service: StrategyService = Depends(get_strategy_service),
):
    """List all strategies.

    Args:
        active_only: If True, return only active strategies

    Returns:
        List of strategies
    """
    try:
        strategies = strategy_service.get_all_strategies(active_only=active_only)
        return StrategyListResponse(
            strategies=strategies,
            total=len(strategies),
        )
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: int,
    strategy_service: StrategyService = Depends(get_strategy_service),
):
    """Get strategy by ID.

    Args:
        strategy_id: Strategy ID

    Returns:
        Strategy details
    """
    try:
        strategy = strategy_service.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/stats", response_model=StrategyWithStats)
async def get_strategy_stats(
    strategy_id: int,
    strategy_service: StrategyService = Depends(get_strategy_service),
):
    """Get strategy with performance statistics.

    Args:
        strategy_id: Strategy ID

    Returns:
        Strategy with computed statistics
    """
    try:
        strategy = strategy_service.get_strategy_with_stats(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy stats {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/trades", response_model=TradeListResponse)
async def get_strategy_trades(
    strategy_id: int,
    limit: int = Query(100, ge=1, le=1000, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    trade_service: TradeService = Depends(get_trade_service),
):
    """Get trades for a specific strategy.

    Args:
        strategy_id: Strategy ID
        limit: Maximum number of trades
        offset: Pagination offset

    Returns:
        List of trades for the strategy
    """
    try:
        return trade_service.get_trades(
            strategy_id=strategy_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error getting trades for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/positions", response_model=PositionListResponse)
async def get_strategy_positions(
    strategy_id: int,
    status: Optional[str] = Query(None, description="Filter by status (open/closed)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of positions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    position_service: PositionService = Depends(get_position_service),
):
    """Get positions for a specific strategy.

    Args:
        strategy_id: Strategy ID
        status: Filter by status
        limit: Maximum number of positions
        offset: Pagination offset

    Returns:
        List of positions for the strategy
    """
    try:
        return position_service.get_positions(
            strategy_id=strategy_id,
            status=status,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error getting positions for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/models", response_model=ModelListResponse)
async def get_strategy_models(
    strategy_id: int,
    strategy_service: StrategyService = Depends(get_strategy_service),
    model_service: ModelService = Depends(get_model_service),
):
    """Get all models trained for a specific strategy.

    Returns the training history for this strategy - all models that have
    been trained, sorted by training date (newest first).

    Args:
        strategy_id: Strategy ID

    Returns:
        List of models trained for this strategy
    """
    try:
        strategy = strategy_service.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Get models for this strategy
        # Extract the base strategy name from strategy_name (e.g., "sol_swing_momentum_v1" -> "sol_swing_momentum")
        base_name = strategy.strategy_name
        if base_name.endswith(f"_{strategy.version}"):
            base_name = base_name[:-len(f"_{strategy.version}")]

        models = model_service.get_models_for_strategy(base_name)

        return ModelListResponse(
            models=[
                ModelSummary(
                    model_id=m.model_id,
                    strategy_name=m.strategy_name,
                    model_type=m.model_type,
                    version=m.version,
                    trained_at=m.trained_at,
                    validation_metrics=m.validation_metrics,
                )
                for m in models
            ],
            total=len(models),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}/bankroll", response_model=BankrollResponse)
async def get_strategy_bankroll(
    strategy_id: int,
    strategy_service: StrategyService = Depends(get_strategy_service),
):
    """Get current bankroll for a strategy.

    Args:
        strategy_id: Strategy ID

    Returns:
        Bankroll details including current balance and P&L
    """
    try:
        strategy = strategy_service.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        pnl = strategy.current_bankroll - strategy.initial_bankroll
        pnl_pct = (pnl / strategy.initial_bankroll * 100) if strategy.initial_bankroll > 0 else 0

        return BankrollResponse(
            strategy_id=strategy.id,
            strategy_name=strategy.strategy_name,
            initial_bankroll=strategy.initial_bankroll,
            current_bankroll=strategy.current_bankroll,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_active=strategy.current_bankroll > 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bankroll for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{strategy_id}/bankroll", response_model=BankrollResponse)
async def update_strategy_bankroll(
    strategy_id: int,
    request: BankrollUpdateRequest,
    strategy_service: StrategyService = Depends(get_strategy_service),
):
    """Update/replenish strategy bankroll.

    Use this endpoint to manually adjust a strategy's bankroll,
    such as replenishing a depleted strategy.

    Args:
        strategy_id: Strategy ID
        request: New bankroll value

    Returns:
        Updated bankroll details
    """
    try:
        strategy = strategy_service.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Update bankroll
        strategy_service.update_strategy_bankroll(strategy_id, request.bankroll)

        # Return updated values
        pnl = request.bankroll - strategy.initial_bankroll
        pnl_pct = (pnl / strategy.initial_bankroll * 100) if strategy.initial_bankroll > 0 else 0

        return BankrollResponse(
            strategy_id=strategy_id,
            strategy_name=strategy.strategy_name,
            initial_bankroll=strategy.initial_bankroll,
            current_bankroll=request.bankroll,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_active=request.bankroll > 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating bankroll for strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
