"""Model metadata tracking for reproducibility and provenance."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from data.metadata import DatasetMetadata


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_paths_to_strings(item) for item in obj]
    else:
        return obj


@dataclass
class ModelMetadata:
    """
    Complete metadata for a trained model.

    Tracks everything needed for reproducibility:
    - What strategy/algorithm was used
    - What data was used for training
    - What features and hyperparameters were used
    - How well the model performed
    - Where the model artifact is stored
    """

    # Identity
    model_id: str  # UUID or hash-based identifier
    strategy_name: str  # e.g., "ema_cross_rsi", "bollinger_breakout"
    strategy_timeframe: str  # e.g., "short_term", "long_term"
    version: str  # e.g., "v1", "v2" for strategy evolution

    # Training Data Linkage
    training_dataset_path: str  # Path to the source .parquet/.csv file
    training_date_range: tuple[str, str]  # (start, end) ISO format
    validation_date_range: tuple[str, str]  # (start, end) ISO format

    # Model Type
    model_type: str  # "xgboost", "lightgbm", "random_forest", etc.

    # Configuration (stored as dicts for flexibility)
    feature_config: dict[str, Any]  # Which indicators, parameters, resampling
    model_config: dict[str, Any]  # Hyperparameters for the ML model
    target_config: dict[str, Any]  # How target variable was defined

    # Performance Metrics (from validation)
    validation_metrics: dict[str, float]  # Sharpe, win_rate, etc.

    # Storage
    model_artifact_path: str  # Path to saved model (.pkl, .joblib, etc.)

    # Timestamps
    trained_at: str  # ISO format timestamp
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Optional
    notes: Optional[str] = None
    tags: list[str] = field(default_factory=list)  # For custom categorization

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "model_id": self.model_id,
            "strategy_name": self.strategy_name,
            "strategy_timeframe": self.strategy_timeframe,
            "version": self.version,
            "training_dataset_path": str(self.training_dataset_path),
            "training_date_range": list(self.training_date_range),
            "validation_date_range": list(self.validation_date_range),
            "model_type": self.model_type,
            "feature_config": self.feature_config,
            "model_config": self.model_config,
            "target_config": self.target_config,
            "validation_metrics": self.validation_metrics,
            "model_artifact_path": str(self.model_artifact_path),
            "trained_at": self.trained_at,
            "created_at": self.created_at,
            "notes": self.notes,
            "tags": self.tags,
        }
        # Recursively convert any Path objects to strings
        return _convert_paths_to_strings(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create ModelMetadata from dictionary."""
        # Convert lists back to tuples for date ranges
        data = data.copy()
        if isinstance(data.get("training_date_range"), list):
            data["training_date_range"] = tuple(data["training_date_range"])
        if isinstance(data.get("validation_date_range"), list):
            data["validation_date_range"] = tuple(data["validation_date_range"])

        return cls(**data)

    def save(self, path: Path | str) -> Path:
        """
        Save metadata to JSON file.

        Args:
            path: Path to save metadata (typically {model_id}.json)

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path | str) -> "ModelMetadata":
        """
        Load metadata from JSON file.

        Args:
            path: Path to metadata file

        Returns:
            ModelMetadata instance
        """
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def get_config_hash(self) -> Optional[str]:
        """
        Extract config hash from hybrid model ID.

        For hybrid IDs (format: {config_hash}_{timestamp}_{suffix}),
        returns the config hash portion. For non-hybrid IDs, returns None.

        Returns:
            Config hash if hybrid ID, None otherwise
        """
        parts = self.model_id.split("_")
        if len(parts) >= 3 and len(parts[0]) == 16:
            # Hybrid ID: first part is config hash (16 hex chars)
            return parts[0]
        return None

    def get_training_timestamp(self) -> Optional[str]:
        """
        Extract training timestamp from hybrid model ID.

        For hybrid IDs (format: {config_hash}_{timestamp}_{suffix}),
        returns the timestamp portion. For non-hybrid IDs, returns None.

        Returns:
            Timestamp string (YYYYMMDD_HHMMSS) if hybrid ID, None otherwise
        """
        parts = self.model_id.split("_")
        if len(parts) >= 3 and len(parts[0]) == 16:
            # Hybrid ID: second and third parts form timestamp
            return f"{parts[1]}_{parts[2]}"
        return None

    def is_hybrid_id(self) -> bool:
        """
        Check if model ID is hybrid format.

        Returns:
            True if hybrid ID format, False otherwise
        """
        return self.get_config_hash() is not None

    def get_dataset_metadata(self) -> Optional[DatasetMetadata]:
        """
        Load the associated dataset metadata.

        Returns:
            DatasetMetadata if metadata file exists, None otherwise
        """
        dataset_path = Path(self.training_dataset_path)
        metadata_path = dataset_path.parent / f"{dataset_path.stem}.metadata.json"

        if metadata_path.exists():
            return DatasetMetadata.load(metadata_path)
        return None

    def get_asset_info(self) -> Optional[dict[str, str]]:
        """
        Get asset information from linked dataset metadata.

        Returns:
            Dict with asset, quote, exchange, market_type if available
        """
        dataset_meta = self.get_dataset_metadata()
        if dataset_meta:
            return {
                "asset": dataset_meta.asset,
                "quote": dataset_meta.quote,
                "exchange": dataset_meta.exchange,
                "market_type": dataset_meta.market_type,
            }
        return None

    def summary(self) -> str:
        """
        Get human-readable summary of model metadata.

        Returns:
            Summary string
        """
        asset_info = self.get_asset_info()
        asset_str = (
            f"{asset_info['asset']}/{asset_info['quote']} ({asset_info['exchange']} {asset_info['market_type']})"
            if asset_info
            else "Unknown"
        )

        summary = f"Model Metadata Summary\n"
        summary += f"{'=' * 50}\n\n"
        summary += f"Model ID:     {self.model_id}\n"
        summary += f"Strategy:     {self.strategy_name} ({self.strategy_timeframe})\n"
        summary += f"Version:      {self.version}\n"
        summary += f"Model Type:   {self.model_type}\n\n"
        summary += f"Asset:        {asset_str}\n"
        summary += f"Training:     {self.training_date_range[0]} to {self.training_date_range[1]}\n"
        summary += f"Validation:   {self.validation_date_range[0]} to {self.validation_date_range[1]}\n\n"
        summary += f"Performance Metrics:\n"
        for metric, value in self.validation_metrics.items():
            summary += f"  {metric:20s}: {value:>10.4f}\n"
        summary += f"\nTrained:      {self.trained_at}\n"
        summary += f"Artifact:     {self.model_artifact_path}\n"

        if self.tags:
            summary += f"Tags:         {', '.join(self.tags)}\n"

        if self.notes:
            summary += f"\nNotes:\n{self.notes}\n"

        return summary
