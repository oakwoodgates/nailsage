"""Model service for accessing model registry."""

import logging
from typing import List, Optional

from models.registry import ModelRegistry
from models.metadata import ModelMetadata

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model-related operations.

    Wraps ModelRegistry to provide model metadata access for the API.
    Reads from file-based JSON metadata in models/metadata/.
    """

    def __init__(self, models_dir: str = "models/trained", metadata_dir: str = "models/metadata"):
        """Initialize model service.

        Args:
            models_dir: Directory for model artifacts
            metadata_dir: Directory for model metadata JSON files
        """
        self.registry = ModelRegistry(models_dir=models_dir, metadata_dir=metadata_dir)
        logger.info("ModelService initialized")

    def list_all_models(self) -> List[ModelMetadata]:
        """List all models in registry.

        Returns:
            List of all model metadata
        """
        return self.registry.list_models()

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get specific model by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata if found, None otherwise
        """
        return self.registry.get_model(model_id)

    def get_models_for_strategy(self, strategy_name: str) -> List[ModelMetadata]:
        """Get all models trained for a strategy.

        Args:
            strategy_name: Strategy name to filter by

        Returns:
            List of models for the strategy, sorted by trained_at descending
        """
        models = self.registry.find_models(strategy_name=strategy_name)
        # Sort by trained_at descending (newest first)
        models.sort(key=lambda m: m.trained_at, reverse=True)
        return models

    def get_active_model(self, strategy_name: str, strategy_timeframe: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get the latest/active model for a strategy.

        Args:
            strategy_name: Strategy name
            strategy_timeframe: Optional timeframe filter

        Returns:
            Latest model for the strategy, or None if not found
        """
        return self.registry.get_latest_model(
            strategy_name=strategy_name,
            strategy_timeframe=strategy_timeframe,
        )

    def get_model_count(self) -> int:
        """Get total number of registered models.

        Returns:
            Count of models
        """
        return self.registry.get_model_count()

    def get_strategies_summary(self) -> dict:
        """Get summary of strategies and their models.

        Returns:
            Dict with strategy names as keys and model info as values
        """
        return self.registry.get_strategies_summary()
