"""Unit tests for model registry."""

import tempfile
from pathlib import Path

import pytest

from models import (
    ModelMetadata,
    ModelRegistry,
    compare_models,
    create_model_metadata,
    generate_model_id,
)


@pytest.mark.unit
def test_generate_model_id_hash():
    """Test hash-based model ID generation."""
    model_id1 = generate_model_id(
        strategy_name="test_strategy",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        model_config={"max_depth": 5, "learning_rate": 0.01},
        use_hash=True,
    )

    model_id2 = generate_model_id(
        strategy_name="test_strategy",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        model_config={"max_depth": 5, "learning_rate": 0.01},
        use_hash=True,
    )

    # Same inputs should produce same hash
    assert model_id1 == model_id2
    assert len(model_id1) == 16  # Hash is truncated to 16 chars


@pytest.mark.unit
def test_generate_model_id_uuid():
    """Test UUID-based model ID generation."""
    model_id1 = generate_model_id(
        strategy_name="test_strategy",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        model_config={"max_depth": 5},
        use_hash=False,
    )

    model_id2 = generate_model_id(
        strategy_name="test_strategy",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        model_config={"max_depth": 5},
        use_hash=False,
    )

    # UUIDs should be different
    assert model_id1 != model_id2
    assert len(model_id1) == 36  # Standard UUID length


@pytest.mark.unit
def test_create_model_metadata():
    """Test model metadata creation."""
    metadata = create_model_metadata(
        strategy_name="test_strategy",
        strategy_timeframe="short_term",
        version="v1",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        validation_date_range=("2025-02-01", "2025-02-28"),
        model_type="xgboost",
        feature_config={"indicators": ["sma_10", "rsi_14"]},
        model_config={"max_depth": 5, "learning_rate": 0.01},
        target_config={"type": "classification", "classes": 3},
        validation_metrics={"sharpe_ratio": 1.5, "win_rate": 0.55},
        model_artifact_path="/path/to/model.joblib",
    )

    assert isinstance(metadata, ModelMetadata)
    assert metadata.strategy_name == "test_strategy"
    assert metadata.strategy_timeframe == "short_term"
    assert metadata.model_type == "xgboost"
    assert metadata.validation_metrics["sharpe_ratio"] == 1.5
    assert len(metadata.model_id) > 0


@pytest.mark.unit
def test_model_metadata_save_load(tmp_path):
    """Test saving and loading model metadata."""
    metadata = create_model_metadata(
        strategy_name="test_strategy",
        strategy_timeframe="long_term",
        version="v2",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-06-30"),
        validation_date_range=("2025-07-01", "2025-12-31"),
        model_type="lightgbm",
        feature_config={"indicators": ["ema_20"]},
        model_config={"num_leaves": 31},
        target_config={"type": "regression"},
        validation_metrics={"mse": 0.01},
        model_artifact_path="/path/to/model.pkl",
    )

    # Save
    save_path = tmp_path / f"{metadata.model_id}.json"
    metadata.save(save_path)

    assert save_path.exists()

    # Load
    loaded = ModelMetadata.load(save_path)

    assert loaded.model_id == metadata.model_id
    assert loaded.strategy_name == metadata.strategy_name
    assert loaded.model_type == metadata.model_type
    assert loaded.validation_metrics == metadata.validation_metrics


@pytest.mark.unit
def test_model_registry_initialization(temp_models_dir):
    """Test ModelRegistry initialization."""
    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    assert registry.models_dir.exists()
    assert registry.metadata_dir.exists()


@pytest.mark.unit
def test_model_registry_register_and_get(temp_models_dir):
    """Test registering and retrieving a model."""
    # Create temporary model file
    temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    temp_model.write(b"fake model data")
    temp_model.close()

    metadata = create_model_metadata(
        strategy_name="test_strategy",
        strategy_timeframe="short_term",
        version="v1",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        validation_date_range=("2025-02-01", "2025-02-28"),
        model_type="xgboost",
        feature_config={"indicators": ["sma"]},
        model_config={"max_depth": 3},
        target_config={"type": "classification"},
        validation_metrics={"sharpe_ratio": 1.2},
        model_artifact_path=temp_model.name,
    )

    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    # Register
    registered = registry.register_model(Path(temp_model.name), metadata)

    assert registered.model_id == metadata.model_id
    assert Path(registered.model_artifact_path).exists()

    # Get
    retrieved = registry.get_model(metadata.model_id)

    assert retrieved is not None
    assert retrieved.model_id == metadata.model_id
    assert retrieved.strategy_name == "test_strategy"

    # Cleanup
    Path(temp_model.name).unlink(missing_ok=True)


@pytest.mark.unit
def test_model_registry_find_models(temp_models_dir):
    """Test finding models by criteria."""
    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    # Create and register multiple models
    for i, (strategy, timeframe) in enumerate(
        [
            ("strategy_a", "short_term"),
            ("strategy_a", "long_term"),
            ("strategy_b", "short_term"),
        ]
    ):
        temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        temp_model.write(f"model {i}".encode())
        temp_model.close()

        metadata = create_model_metadata(
            strategy_name=strategy,
            strategy_timeframe=timeframe,
            version="v1",
            training_dataset_path=f"data/test{i}.parquet",
            training_date_range=("2025-01-01", "2025-01-31"),
            validation_date_range=("2025-02-01", "2025-02-28"),
            model_type="xgboost",
            feature_config={},
            model_config={},
            target_config={},
            validation_metrics={"sharpe_ratio": 1.0 + i * 0.1},
            model_artifact_path=temp_model.name,
        )

        registry.register_model(Path(temp_model.name), metadata)
        Path(temp_model.name).unlink(missing_ok=True)

    # Find by strategy
    strategy_a_models = registry.find_models(strategy_name="strategy_a")
    assert len(strategy_a_models) == 2

    # Find by timeframe
    short_term_models = registry.find_models(strategy_timeframe="short_term")
    assert len(short_term_models) == 2

    # Find by strategy and timeframe
    specific_models = registry.find_models(strategy_name="strategy_a", strategy_timeframe="short_term")
    assert len(specific_models) == 1


@pytest.mark.unit
def test_compare_models():
    """Test model comparison."""
    metadata1 = create_model_metadata(
        strategy_name="test",
        strategy_timeframe="short_term",
        version="v1",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        validation_date_range=("2025-02-01", "2025-02-28"),
        model_type="xgboost",
        feature_config={},
        model_config={},
        target_config={},
        validation_metrics={"sharpe_ratio": 1.5, "win_rate": 0.55},
        model_artifact_path="/path/to/model1.joblib",
    )

    metadata2 = create_model_metadata(
        strategy_name="test",
        strategy_timeframe="short_term",
        version="v2",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        validation_date_range=("2025-02-01", "2025-02-28"),
        model_type="xgboost",
        feature_config={},
        model_config={"max_depth": 7},
        target_config={},
        validation_metrics={"sharpe_ratio": 1.8, "win_rate": 0.60},
        model_artifact_path="/path/to/model2.joblib",
    )

    comparison = compare_models(metadata1, metadata2, metrics=["sharpe_ratio", "win_rate"])

    assert "metrics" in comparison
    assert "sharpe_ratio" in comparison["metrics"]
    assert comparison["metrics"]["sharpe_ratio"]["model1"] == 1.5
    assert comparison["metrics"]["sharpe_ratio"]["model2"] == 1.8
    assert abs(comparison["metrics"]["sharpe_ratio"]["difference"] - 0.3) < 0.001  # Float comparison


@pytest.mark.unit
def test_model_registry_get_best_model(temp_models_dir):
    """Test getting best model by metric."""
    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    # Register models with different performance
    for i, sharpe in enumerate([1.2, 1.8, 1.5]):
        temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        temp_model.write(f"model {i}".encode())
        temp_model.close()

        metadata = create_model_metadata(
            strategy_name="test_strategy",
            strategy_timeframe="short_term",
            version=f"v{i}",
            training_dataset_path=f"data/test{i}.parquet",
            training_date_range=("2025-01-01", "2025-01-31"),
            validation_date_range=("2025-02-01", "2025-02-28"),
            model_type="xgboost",
            feature_config={},
            model_config={},
            target_config={},
            validation_metrics={"sharpe_ratio": sharpe},
            model_artifact_path=temp_model.name,
        )

        registry.register_model(Path(temp_model.name), metadata)
        Path(temp_model.name).unlink(missing_ok=True)

    # Get best
    best = registry.get_best_model("sharpe_ratio", strategy_name="test_strategy")

    assert best is not None
    assert best.validation_metrics["sharpe_ratio"] == 1.8
    assert best.version == "v1"
