"""Model metadata schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    """Lightweight model info for list endpoints."""

    model_id: str = Field(description="Unique model identifier")
    strategy_name: str = Field(description="Strategy name (e.g., sol_swing_momentum)")
    model_type: str = Field(description="Model type (e.g., random_forest, xgboost)")
    version: str = Field(description="Model version")
    trained_at: str = Field(description="Training timestamp (ISO format)")
    validation_metrics: Dict[str, float] = Field(description="Validation performance metrics")


class ModelMetadataResponse(BaseModel):
    """Full model details including all results and configuration."""

    # Identity
    model_id: str = Field(description="Unique model identifier")
    strategy_name: str = Field(description="Strategy name")
    strategy_timeframe: str = Field(description="Strategy timeframe (short_term, long_term)")
    version: str = Field(description="Model version")
    model_type: str = Field(description="Model type (e.g., random_forest, xgboost)")
    trained_at: str = Field(description="Training timestamp (ISO format)")

    # Training data info
    training_dataset_path: str = Field(description="Path to training dataset")
    training_date_range: List[str] = Field(description="Training date range [start, end]")
    validation_date_range: List[str] = Field(description="Validation date range [start, end]")

    # Configuration
    feature_config: Dict[str, Any] = Field(description="Feature engineering configuration")
    hyperparameters: Dict[str, Any] = Field(description="Model hyperparameters")
    target_config: Dict[str, Any] = Field(description="Target variable configuration")

    # Results
    validation_metrics: Dict[str, float] = Field(description="Validation performance metrics")

    # Metadata
    notes: Optional[str] = Field(None, description="Notes about the model")
    tags: List[str] = Field(default_factory=list, description="Model tags")


class ModelListResponse(BaseModel):
    """Response for model list endpoint."""

    models: List[ModelSummary] = Field(description="List of models")
    total: int = Field(description="Total number of models")
