"""Utility functions for model metadata generation and management."""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from models.metadata import ModelMetadata


def generate_model_id(
    strategy_name: str,
    training_dataset_path: str,
    training_date_range: tuple[str, str],
    model_config: dict,
    use_hash: bool = True,
) -> str:
    """
    Generate a unique model ID.

    Args:
        strategy_name: Name of the strategy
        training_dataset_path: Path to training dataset
        training_date_range: Training date range
        model_config: Model hyperparameters
        use_hash: If True, use hash-based ID; if False, use UUID

    Returns:
        Unique model identifier
    """
    if use_hash:
        # Create deterministic hash based on key components
        components = [
            strategy_name,
            training_dataset_path,
            training_date_range[0],
            training_date_range[1],
            str(sorted(model_config.items())),
        ]
        hash_input = "|".join(components).encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()[:16]
    else:
        # Use random UUID
        return str(uuid.uuid4())


def create_model_metadata(
    strategy_name: str,
    strategy_timeframe: str,
    version: str,
    training_dataset_path: str,
    training_date_range: tuple[str, str],
    validation_date_range: tuple[str, str],
    model_type: str,
    feature_config: dict[str, Any],
    model_config: dict[str, Any],
    target_config: dict[str, Any],
    validation_metrics: dict[str, float],
    model_artifact_path: str,
    model_id: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> ModelMetadata:
    """
    Create ModelMetadata with automatic ID generation if not provided.

    Args:
        strategy_name: Strategy name (e.g., "ema_cross_rsi")
        strategy_timeframe: Timeframe (e.g., "short_term")
        version: Strategy version (e.g., "v1")
        training_dataset_path: Path to training data
        training_date_range: (start, end) for training
        validation_date_range: (start, end) for validation
        model_type: Model type (e.g., "xgboost")
        feature_config: Feature configuration dict
        model_config: Model hyperparameters
        target_config: Target variable configuration
        validation_metrics: Performance metrics
        model_artifact_path: Path to saved model file
        model_id: Optional model ID (generated if None)
        notes: Optional notes
        tags: Optional tags

    Returns:
        ModelMetadata instance
    """
    if model_id is None:
        model_id = generate_model_id(
            strategy_name=strategy_name,
            training_dataset_path=training_dataset_path,
            training_date_range=training_date_range,
            model_config=model_config,
            use_hash=True,
        )

    return ModelMetadata(
        model_id=model_id,
        strategy_name=strategy_name,
        strategy_timeframe=strategy_timeframe,
        version=version,
        training_dataset_path=training_dataset_path,
        training_date_range=training_date_range,
        validation_date_range=validation_date_range,
        model_type=model_type,
        feature_config=feature_config,
        model_config=model_config,
        target_config=target_config,
        validation_metrics=validation_metrics,
        model_artifact_path=model_artifact_path,
        trained_at=datetime.utcnow().isoformat() + "Z",
        notes=notes,
        tags=tags or [],
    )


def extract_feature_config(feature_engine_config: dict) -> dict[str, Any]:
    """
    Extract feature configuration from FeatureEngine or config.

    Args:
        feature_engine_config: Config dict or FeatureEngine instance

    Returns:
        Standardized feature config dict
    """
    # This is a helper to standardize feature config format
    # Will be more useful once we have actual FeatureEngine instances
    return {
        "indicators": feature_engine_config.get("indicators", []),
        "parameters": feature_engine_config.get("parameters", {}),
        "resampling": feature_engine_config.get("resampling", None),
    }


def compare_models(
    model1: ModelMetadata,
    model2: ModelMetadata,
    metrics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compare two models on specified metrics.

    Args:
        model1: First model
        model2: Second model
        metrics: List of metric names to compare (if None, compare all common metrics)

    Returns:
        Dict with comparison results
    """
    if metrics is None:
        # Find common metrics
        metrics = list(
            set(model1.validation_metrics.keys()) & set(model2.validation_metrics.keys())
        )

    comparison = {
        "model1_id": model1.model_id,
        "model2_id": model2.model_id,
        "metrics": {},
    }

    for metric in metrics:
        if metric in model1.validation_metrics and metric in model2.validation_metrics:
            val1 = model1.validation_metrics[metric]
            val2 = model2.validation_metrics[metric]
            comparison["metrics"][metric] = {
                "model1": val1,
                "model2": val2,
                "difference": val2 - val1,
                "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else None,
            }

    return comparison


def get_model_lineage(
    model: ModelMetadata,
    include_dataset: bool = True,
) -> dict[str, Any]:
    """
    Get complete lineage information for a model.

    Args:
        model: ModelMetadata to trace
        include_dataset: If True, include dataset metadata

    Returns:
        Dict with lineage information
    """
    lineage = {
        "model_id": model.model_id,
        "strategy": {
            "name": model.strategy_name,
            "timeframe": model.strategy_timeframe,
            "version": model.version,
        },
        "training": {
            "date_range": model.training_date_range,
            "dataset_path": model.training_dataset_path,
        },
        "validation": {
            "date_range": model.validation_date_range,
        },
        "configuration": {
            "model_type": model.model_type,
            "features": model.feature_config,
            "hyperparameters": model.model_config,
            "target": model.target_config,
        },
        "performance": model.validation_metrics,
        "trained_at": model.trained_at,
    }

    if include_dataset:
        dataset_meta = model.get_dataset_metadata()
        if dataset_meta:
            lineage["dataset"] = {
                "asset": dataset_meta.asset,
                "quote": dataset_meta.quote,
                "exchange": dataset_meta.exchange,
                "market_type": dataset_meta.market_type,
                "interval": dataset_meta.interval,
                "total_bars": dataset_meta.num_bars,
                "quality_score": dataset_meta.data_quality_score,
            }

    return lineage
