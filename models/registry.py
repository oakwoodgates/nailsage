"""Model registry for managing trained models and their metadata."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from models.metadata import ModelMetadata
from utils.logger import get_logger

logger = get_logger("models")


class ModelRegistry:
    """
    Registry for managing trained models.

    Provides:
    - Save/load models with metadata
    - Query models by various criteria
    - Version management
    - Centralized model storage
    """

    def __init__(
        self,
        models_dir: Path | str = "models/trained",
        metadata_dir: Path | str = "models/metadata",
    ):
        """
        Initialize ModelRegistry.

        Args:
            models_dir: Directory for model artifacts
            metadata_dir: Directory for model metadata JSON files
        """
        self.models_dir = Path(models_dir)
        self.metadata_dir = Path(metadata_dir)

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized ModelRegistry",
            extra_data={
                "models_dir": str(self.models_dir),
                "metadata_dir": str(self.metadata_dir),
            },
        )

    def register_model(
        self,
        model_artifact_path: Path | str,
        metadata: ModelMetadata,
        copy_artifact: bool = True,
    ) -> ModelMetadata:
        """
        Register a trained model with the registry.

        Args:
            model_artifact_path: Path to the trained model file (.pkl, .joblib, etc.)
            metadata: ModelMetadata for the model
            copy_artifact: If True, copy the artifact to registry (default: True)

        Returns:
            ModelMetadata with updated artifact path
        """
        model_artifact_path = Path(model_artifact_path)

        if not model_artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_artifact_path}")

        # Determine target path in registry
        artifact_filename = f"{metadata.model_id}{model_artifact_path.suffix}"
        target_artifact_path = self.models_dir / artifact_filename

        # Copy or move artifact to registry
        if copy_artifact:
            logger.info(f"Copying model artifact to registry: {artifact_filename}")
            shutil.copy2(model_artifact_path, target_artifact_path)
        else:
            logger.info(f"Moving model artifact to registry: {artifact_filename}")
            shutil.move(str(model_artifact_path), target_artifact_path)

        # Update metadata with registry path
        metadata.model_artifact_path = str(target_artifact_path)

        # Save metadata
        metadata_path = self.metadata_dir / f"{metadata.model_id}.json"
        metadata.save(metadata_path)

        logger.info(
            f"Registered model: {metadata.model_id}",
            extra_data={
                "strategy": metadata.strategy_name,
                "timeframe": metadata.strategy_timeframe,
                "model_type": metadata.model_type,
            },
        )

        return metadata

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata if found, None otherwise
        """
        metadata_path = self.metadata_dir / f"{model_id}.json"

        if not metadata_path.exists():
            logger.warning(f"Model not found: {model_id}")
            return None

        return ModelMetadata.load(metadata_path)

    def list_models(self) -> list[ModelMetadata]:
        """
        List all registered models.

        Returns:
            List of ModelMetadata for all registered models
        """
        models = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                metadata = ModelMetadata.load(metadata_file)
                models.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
                continue

        return models

    def find_models(
        self,
        strategy_name: Optional[str] = None,
        strategy_timeframe: Optional[str] = None,
        model_type: Optional[str] = None,
        version: Optional[str] = None,
        trained_after: Optional[str] = None,
        trained_before: Optional[str] = None,
        tags: Optional[list[str]] = None,
        asset: Optional[str] = None,
        market_type: Optional[str] = None,
    ) -> list[ModelMetadata]:
        """
        Find models matching criteria.

        Args:
            strategy_name: Filter by strategy name
            strategy_timeframe: Filter by timeframe (short_term, long_term)
            model_type: Filter by model type (xgboost, lightgbm, etc.)
            version: Filter by version
            trained_after: ISO timestamp - models trained after this date
            trained_before: ISO timestamp - models trained before this date
            tags: Filter by tags (model must have all specified tags)
            asset: Filter by asset (requires dataset metadata)
            market_type: Filter by market type (requires dataset metadata)

        Returns:
            List of matching ModelMetadata
        """
        all_models = self.list_models()
        results = []

        for model in all_models:
            # Check basic filters
            if strategy_name and model.strategy_name != strategy_name:
                continue
            if strategy_timeframe and model.strategy_timeframe != strategy_timeframe:
                continue
            if model_type and model.model_type != model_type:
                continue
            if version and model.version != version:
                continue

            # Check date filters
            if trained_after and model.trained_at < trained_after:
                continue
            if trained_before and model.trained_at > trained_before:
                continue

            # Check tags
            if tags and not all(tag in model.tags for tag in tags):
                continue

            # Check asset/market filters (requires dataset metadata)
            if asset or market_type:
                asset_info = model.get_asset_info()
                if not asset_info:
                    continue
                if asset and asset_info["asset"] != asset:
                    continue
                if market_type and asset_info["market_type"] != market_type:
                    continue

            results.append(model)

        logger.info(f"Found {len(results)} models matching criteria")
        return results

    def get_latest_model(
        self,
        strategy_name: str,
        strategy_timeframe: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[ModelMetadata]:
        """
        Get the most recently trained model for a strategy.

        Args:
            strategy_name: Strategy name to search for
            strategy_timeframe: Optional timeframe filter
            version: Optional version filter

        Returns:
            Most recent ModelMetadata if found, None otherwise
        """
        models = self.find_models(
            strategy_name=strategy_name,
            strategy_timeframe=strategy_timeframe,
            version=version,
        )

        if not models:
            return None

        # Sort by trained_at descending
        models.sort(key=lambda m: m.trained_at, reverse=True)
        return models[0]

    def get_best_model(
        self,
        metric: str,
        strategy_name: Optional[str] = None,
        strategy_timeframe: Optional[str] = None,
        higher_is_better: bool = True,
    ) -> Optional[ModelMetadata]:
        """
        Get the best performing model by a specific metric.

        Args:
            metric: Metric name (e.g., "sharpe_ratio", "win_rate")
            strategy_name: Optional strategy filter
            strategy_timeframe: Optional timeframe filter
            higher_is_better: If True, maximize metric; if False, minimize

        Returns:
            Best performing ModelMetadata if found, None otherwise
        """
        models = self.find_models(
            strategy_name=strategy_name,
            strategy_timeframe=strategy_timeframe,
        )

        if not models:
            return None

        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.validation_metrics]

        if not models_with_metric:
            logger.warning(f"No models found with metric: {metric}")
            return None

        # Sort by metric
        models_with_metric.sort(
            key=lambda m: m.validation_metrics[metric],
            reverse=higher_is_better,
        )

        return models_with_metric[0]

    def delete_model(self, model_id: str, delete_artifact: bool = True) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model identifier
            delete_artifact: If True, also delete the model artifact file

        Returns:
            True if deleted, False if not found
        """
        metadata_path = self.metadata_dir / f"{model_id}.json"

        if not metadata_path.exists():
            logger.warning(f"Model not found: {model_id}")
            return False

        # Load metadata to get artifact path
        metadata = ModelMetadata.load(metadata_path)

        # Delete artifact if requested
        if delete_artifact:
            artifact_path = Path(metadata.model_artifact_path)
            if artifact_path.exists():
                artifact_path.unlink()
                logger.info(f"Deleted model artifact: {artifact_path}")

        # Delete metadata
        metadata_path.unlink()
        logger.info(f"Deleted model metadata: {model_id}")

        return True

    def get_model_count(self) -> int:
        """
        Get total number of registered models.

        Returns:
            Count of registered models
        """
        return len(list(self.metadata_dir.glob("*.json")))

    def get_strategies_summary(self) -> dict[str, Any]:
        """
        Get summary statistics by strategy.

        Returns:
            Dict with strategy names as keys and count/timeframe info as values
        """
        all_models = self.list_models()
        summary: dict[str, Any] = {}

        for model in all_models:
            strategy_key = model.strategy_name
            if strategy_key not in summary:
                summary[strategy_key] = {
                    "count": 0,
                    "timeframes": set(),
                    "model_types": set(),
                    "versions": set(),
                }

            summary[strategy_key]["count"] += 1
            summary[strategy_key]["timeframes"].add(model.strategy_timeframe)
            summary[strategy_key]["model_types"].add(model.model_type)
            summary[strategy_key]["versions"].add(model.version)

        # Convert sets to lists for JSON serialization
        for strategy in summary.values():
            strategy["timeframes"] = sorted(list(strategy["timeframes"]))
            strategy["model_types"] = sorted(list(strategy["model_types"]))
            strategy["versions"] = sorted(list(strategy["versions"]))

        return summary

    def find_models_by_config(self, config_hash: str) -> list[ModelMetadata]:
        """
        Find all models with a specific configuration hash.

        This finds all training runs of the same configuration (intent),
        regardless of when they were trained.

        Args:
            config_hash: Configuration hash to search for (16 hex chars)

        Returns:
            List of models with matching config hash, sorted chronologically
        """
        all_models = self.list_models()

        # Filter by config hash
        matching = [m for m in all_models if m.get_config_hash() == config_hash]

        # Sort chronologically by model_id (timestamp is embedded)
        matching.sort(key=lambda m: m.model_id)

        logger.info(f"Found {len(matching)} models with config hash: {config_hash}")
        return matching

    def get_latest_by_config(self, config_hash: str) -> Optional[ModelMetadata]:
        """
        Get the most recently trained model for a specific configuration.

        Args:
            config_hash: Configuration hash to search for

        Returns:
            Most recent model with this config, or None if not found
        """
        models = self.find_models_by_config(config_hash)
        if not models:
            return None

        # Last in chronologically sorted list
        return models[-1]

    def list_config_families(self) -> dict[str, list[ModelMetadata]]:
        """
        Group all models by their configuration hash.

        Returns models organized into "families" - all training runs
        of the same configuration grouped together.

        Returns:
            Dict mapping config_hash -> list of models with that hash
        """
        all_models = self.list_models()
        families: dict[str, list[ModelMetadata]] = {}

        for model in all_models:
            config_hash = model.get_config_hash()
            if config_hash:  # Hybrid IDs only
                if config_hash not in families:
                    families[config_hash] = []
                families[config_hash].append(model)

        # Sort each family chronologically
        for models in families.values():
            models.sort(key=lambda m: m.model_id)

        logger.info(f"Found {len(families)} unique configurations")
        return families
