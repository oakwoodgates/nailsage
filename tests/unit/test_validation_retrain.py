"""Unit tests for walk-forward validation and retrain functionality."""

import pytest

from training.cli.validate_model import create_model


class TestWalkForwardRetrain:
    """Test suite for walk-forward validation with retrain."""

    def test_retrain_model_factory_imports(self):
        """Test that retrain model factory can be imported."""
        assert callable(create_model)

    def test_create_model_xgboost(self):
        """Test model creation for XGBoost."""

        params = {'max_depth': 5, 'n_estimators': 100}
        model = create_model('xgboost', params)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_create_model_lightgbm(self):
        """Test model creation for LightGBM."""

        params = {'max_depth': 5, 'n_estimators': 100}
        model = create_model('lightgbm', params)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_model_random_forest(self):
        """Test model creation for RandomForest."""

        params = {'max_depth': 5, 'n_estimators': 100}
        model = create_model('random_forest', params)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_create_model_unknown(self):
        """Test model creation with unknown type."""

        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model('unknown_model', {})
