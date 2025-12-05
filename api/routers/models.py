"""Model metadata endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_model_service
from api.services.model_service import ModelService
from api.schemas.models import (
    ModelSummary,
    ModelMetadataResponse,
    ModelListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


def _to_summary(model) -> ModelSummary:
    """Convert ModelMetadata to ModelSummary."""
    return ModelSummary(
        model_id=model.model_id,
        strategy_name=model.strategy_name,
        model_type=model.model_type,
        version=model.version,
        trained_at=model.trained_at,
        validation_metrics=model.validation_metrics,
    )


def _to_response(model) -> ModelMetadataResponse:
    """Convert ModelMetadata to ModelMetadataResponse."""
    return ModelMetadataResponse(
        model_id=model.model_id,
        strategy_name=model.strategy_name,
        strategy_timeframe=model.strategy_timeframe,
        version=model.version,
        model_type=model.model_type,
        trained_at=model.trained_at,
        training_dataset_path=model.training_dataset_path,
        training_date_range=list(model.training_date_range),
        validation_date_range=list(model.validation_date_range),
        feature_config=model.feature_config,
        hyperparameters=model.model_config,
        target_config=model.target_config,
        validation_metrics=model.validation_metrics,
        notes=model.notes,
        tags=model.tags,
    )


@router.get("", response_model=ModelListResponse)
async def list_models(
    model_service: ModelService = Depends(get_model_service),
):
    """List all trained models.

    Returns:
        List of all models with summary info
    """
    try:
        models = model_service.list_all_models()
        return ModelListResponse(
            models=[_to_summary(m) for m in models],
            total=len(models),
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelMetadataResponse)
async def get_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Get full model metadata including results.

    Args:
        model_id: Model identifier

    Returns:
        Complete model metadata with configuration and results
    """
    try:
        model = model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return _to_response(model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
