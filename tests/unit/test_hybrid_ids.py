"""Unit tests for hybrid ID functionality."""

import pytest

from models import ModelMetadata, ModelRegistry, create_model_metadata, generate_config_hash


@pytest.mark.unit
def test_model_metadata_parse_hybrid_id():
    """Test parsing hybrid ID components."""
    metadata = create_model_metadata(
        strategy_name="test_strategy",
        strategy_timeframe="short_term",
        version="v1",
        training_dataset_path="data/test.parquet",
        training_date_range=("2025-01-01", "2025-01-31"),
        validation_date_range=("2025-02-01", "2025-02-28"),
        model_type="xgboost",
        feature_config={"indicators": ["sma"]},
        model_config={"max_depth": 5},
        target_config={"type": "classification"},
        validation_metrics={"sharpe_ratio": 1.5},
        model_artifact_path="/path/to/model.joblib",
    )

    # Should have hybrid ID
    assert metadata.is_hybrid_id()

    # Should be able to extract config hash
    config_hash = metadata.get_config_hash()
    assert config_hash is not None
    assert len(config_hash) == 16

    # Should be able to extract timestamp
    timestamp = metadata.get_training_timestamp()
    assert timestamp is not None
    assert "_" in timestamp  # Format: YYYYMMDD_HHMMSS


@pytest.mark.unit
def test_config_hash_consistency():
    """Test that config hash is deterministic."""
    hash1 = generate_config_hash(
        strategy_name="ema_cross",
        training_dataset_path="data/btc.parquet",
        training_date_range=("2025-01-01", "2025-12-31"),
        model_config={"max_depth": 5, "lr": 0.01},
        feature_config={"indicators": ["ema_12", "ema_26"]},
        target_config={"type": "classification"},
    )

    hash2 = generate_config_hash(
        strategy_name="ema_cross",
        training_dataset_path="data/btc.parquet",
        training_date_range=("2025-01-01", "2025-12-31"),
        model_config={"max_depth": 5, "lr": 0.01},
        feature_config={"indicators": ["ema_12", "ema_26"]},
        target_config={"type": "classification"},
    )

    # Same inputs = same hash
    assert hash1 == hash2

    # Different config = different hash
    hash3 = generate_config_hash(
        strategy_name="ema_cross",
        training_dataset_path="data/btc.parquet",
        training_date_range=("2025-01-01", "2025-12-31"),
        model_config={"max_depth": 7, "lr": 0.01},  # Changed max_depth
        feature_config={"indicators": ["ema_12", "ema_26"]},
        target_config={"type": "classification"},
    )

    assert hash1 != hash3


@pytest.mark.unit
def test_registry_find_by_config(temp_models_dir):
    """Test finding models by configuration hash."""
    import tempfile
    from pathlib import Path

    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    # Create same config multiple times
    for i in range(3):
        temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        temp_model.write(f"model {i}".encode())
        temp_model.close()

        metadata = create_model_metadata(
            strategy_name="test_strategy",
            strategy_timeframe="short_term",
            version="v1",
            training_dataset_path="data/test.parquet",  # Same
            training_date_range=("2025-01-01", "2025-01-31"),  # Same
            validation_date_range=("2025-02-01", "2025-02-28"),
            model_type="xgboost",
            feature_config={"indicators": ["sma"]},  # Same
            model_config={"max_depth": 5},  # Same
            target_config={"type": "classification"},  # Same
            validation_metrics={"sharpe_ratio": 1.0 + i * 0.1},
            model_artifact_path=temp_model.name,
        )

        registry.register_model(Path(temp_model.name), metadata)
        Path(temp_model.name).unlink(missing_ok=True)

    # All should have same config hash
    all_models = registry.list_models()
    config_hashes = [m.get_config_hash() for m in all_models]
    assert len(set(config_hashes)) == 1  # All same hash

    # Find by config hash
    config_hash = config_hashes[0]
    found = registry.find_models_by_config(config_hash)
    assert len(found) == 3

    # Should be sorted chronologically
    for i in range(len(found) - 1):
        assert found[i].model_id < found[i + 1].model_id


@pytest.mark.unit
def test_registry_get_latest_by_config(temp_models_dir):
    """Test getting latest model for a configuration."""
    import tempfile
    from pathlib import Path
    import time

    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    # Train same config multiple times with small delay
    models_created = []
    for i in range(3):
        # Small delay to ensure different timestamps
        if i > 0:
            time.sleep(1)  # 1 second delay to ensure different timestamps

        temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        temp_model.write(f"model {i}".encode())
        temp_model.close()

        metadata = create_model_metadata(
            strategy_name="test_strategy",
            strategy_timeframe="short_term",
            version="v1",
            training_dataset_path="data/test.parquet",
            training_date_range=("2025-01-01", "2025-01-31"),
            validation_date_range=("2025-02-01", "2025-02-28"),
            model_type="xgboost",
            feature_config={},
            model_config={"max_depth": 5},
            target_config={},
            validation_metrics={"sharpe_ratio": 1.0 + i * 0.1},
            model_artifact_path=temp_model.name,
        )

        registry.register_model(Path(temp_model.name), metadata)
        models_created.append(metadata)
        Path(temp_model.name).unlink(missing_ok=True)

    # Get latest
    config_hash = models_created[0].get_config_hash()
    latest = registry.get_latest_by_config(config_hash)

    assert latest is not None
    assert latest.validation_metrics["sharpe_ratio"] == 1.2  # Last one (i=2)


@pytest.mark.unit
def test_registry_config_families(temp_models_dir):
    """Test grouping models into configuration families."""
    import tempfile
    from pathlib import Path

    registry = ModelRegistry(
        models_dir=temp_models_dir["trained"], metadata_dir=temp_models_dir["metadata"]
    )

    # Create 2 different configs, each trained twice
    for config_id in [1, 2]:
        for run in [1, 2]:
            temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
            temp_model.write(f"model config{config_id} run{run}".encode())
            temp_model.close()

            metadata = create_model_metadata(
                strategy_name="test_strategy",
                strategy_timeframe="short_term",
                version="v1",
                training_dataset_path="data/test.parquet",
                training_date_range=("2025-01-01", "2025-01-31"),
                validation_date_range=("2025-02-01", "2025-02-28"),
                model_type="xgboost",
                feature_config={},
                model_config={"max_depth": config_id * 5},  # Different configs
                target_config={},
                validation_metrics={"sharpe_ratio": 1.0},
                model_artifact_path=temp_model.name,
            )

            registry.register_model(Path(temp_model.name), metadata)
            Path(temp_model.name).unlink(missing_ok=True)

    # Get families
    families = registry.list_config_families()

    # Should have 2 families
    assert len(families) == 2

    # Each family should have 2 models
    for config_hash, models in families.items():
        assert len(models) == 2
        # Should be sorted chronologically
        assert models[0].model_id < models[1].model_id
