"""Main training pipeline orchestrator for Nailsage ML models."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd

from config.strategy import StrategyConfig
from data.loader import DataLoader
from features.engine import FeatureEngine
from models import ModelRegistry, create_model_metadata
from models.feature_schema import FeatureSchema
from training.data_pipeline import DataPipeline
from training.signal_pipeline import SignalPipeline
from training.validator import Validator
from utils.logger import get_training_logger

logger = get_training_logger()


@dataclass
class TrainingResult:
    """Result of a training run."""

    model_id: str
    model_artifact_path: Path
    metadata: Dict[str, Any]
    training_metrics: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]] = None
    feature_schema: Optional[FeatureSchema] = None


class TrainingPipeline:
    """
    Orchestrates the complete ML model training pipeline.

    This class coordinates data loading, feature engineering, model training,
    validation, and model registration into a clean, testable pipeline.

    Attributes:
        config: Strategy configuration
        data_pipeline: Handles data loading and preparation
        signal_pipeline: Handles signal generation and filtering
        validator: Handles model validation
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize training pipeline.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.data_pipeline = DataPipeline(config)
        self.signal_pipeline = SignalPipeline(config)
        self.validator = Validator(config)

        logger.info(
            f"Initialized TrainingPipeline for {config.strategy_name} v{config.version}"
        )

    def run(
        self,
        train_only: bool = False,
        save_results: bool = True,
    ) -> TrainingResult:
        """
        Run the complete training pipeline.

        Args:
            train_only: If True, skip validation
            save_results: If True, save results to JSON

        Returns:
            TrainingResult with model details and metrics
        """
        logger.info("Starting training pipeline...")

        # Step 1: Load and prepare data
        df_clean, target_clean = self.data_pipeline.load_and_prepare_data()

        # Step 2: Filter training period
        train_mask = (
            (df_clean['timestamp'] >= self.config.train_start) &
            (df_clean['timestamp'] <= self.config.train_end)
        )
        df_train = df_clean[train_mask]
        target_train = target_clean[train_mask]

        logger.info(
            f"Training period: {self.config.train_start} to {self.config.train_end}"
        )
        logger.info(f"Training samples: {len(df_train):,}")

        # Step 3: Prepare features for training
        feature_cols = self.data_pipeline.get_feature_columns(df_train)
        X_train = df_train[feature_cols]
        y_train = target_train

        # Step 4: Calculate sample weights if specified
        sample_weights = self._calculate_sample_weights(y_train)

        # Step 5: Train model
        model = self._create_and_train_model(X_train, y_train, sample_weights)

        # Step 6: Evaluate on training set
        train_accuracy = self._evaluate_model(model, X_train, y_train)

        # Step 7: Run validation if requested
        validation_results = None
        if not train_only:
            validation_results = self.validator.run_validation(
                model, df_clean, target_clean, feature_cols
            )

        # Step 8: Create feature schema
        feature_schema = FeatureSchema(
            feature_names=feature_cols,
            ohlcv_columns=self.data_pipeline.ohlcv_columns
        )

        # Step 9: Save model temporarily
        model_filename = self._save_model_temporarily(model)

        # Step 10: Create and register model metadata
        metadata = self._create_model_metadata(
            model_filename, feature_cols, train_accuracy, validation_results
        )

        registry = ModelRegistry()
        registered_metadata = registry.register_model(
            model_artifact_path=Path(model_filename),
            metadata=metadata
        )

        # Step 11: Save results if requested
        if save_results and validation_results:
            self._save_training_results(registered_metadata, validation_results)

        # Step 12: Clean up temporary model file
        self._cleanup_temporary_model(model_filename)

        result = TrainingResult(
            model_id=registered_metadata.model_id,
            model_artifact_path=Path(registered_metadata.model_artifact_path),
            metadata=metadata.to_dict(),
            training_metrics={"train_accuracy": float(train_accuracy)},
            validation_results=validation_results,
            feature_schema=feature_schema,
        )

        logger.info("Training pipeline completed successfully")
        logger.info(f"Model ID: {result.model_id}")

        return result

    def _calculate_sample_weights(self, y_train: pd.Series) -> Optional[np.ndarray]:
        """Calculate sample weights for training if specified in config."""
        if not hasattr(self.config.target, 'class_weights') or not self.config.target.class_weights:
            return None

        class_weights = self.config.target.class_weights
        sample_weights = np.array([class_weights[label] for label in y_train])

        logger.info(f"Using class weights: {class_weights}")

        # Log class distribution with weights
        for class_label in sorted(class_weights.keys()):
            count = (y_train == class_label).sum()
            weight = class_weights[class_label]
            logger.info(
                f"  Class {class_label}: {count:,} samples, weight={weight:.1f}, "
                f"effective={count*weight:.1f}"
            )

        return sample_weights

    def _create_and_train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: Optional[np.ndarray]
    ):
        """Create and train the ML model."""
        model_type = self.config.model_type_str
        logger.info(f"Training {model_type} model...")

        # Import the model creation function from the original script
        model = self._create_model(model_type, self.config.model_params)

        # Prepare fit arguments
        fit_kwargs = {'X': X_train, 'y': y_train}
        if sample_weights is not None:
            fit_kwargs['sample_weight'] = sample_weights

        # Train
        model.fit(**fit_kwargs)

        return model

    def _create_model(self, model_type: str, params: dict):
        """
        Factory function to create ML model based on type.

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'extra_trees')
            params: Model hyperparameters

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**params)

        elif model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)

        elif model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(**params)

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Supported types: xgboost, lightgbm, random_forest, extra_trees"
            )

    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model on given data."""
        predictions = model.predict(X)
        accuracy = float((predictions == y).mean())
        logger.info(f"Training accuracy: {accuracy:.4f}")
        return accuracy

    def _save_model_temporarily(self, model) -> str:
        """Save model to temporary file for registration."""
        model_filename = (
            f"models/trained/{self.config.strategy_name}_{self.config.version}_temp.joblib"
        )
        Path(model_filename).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_filename)
        logger.info(f"Saved model to {model_filename}")
        return model_filename

    def _create_model_metadata(
        self,
        model_filename: str,
        feature_cols: list,
        train_accuracy: float,
        validation_results: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create model metadata for registration."""
        logger.info("Creating model metadata...")

        metadata = create_model_metadata(
            strategy_name=self.config.strategy_name,
            strategy_timeframe=self.config.strategy_timeframe,
            version=self.config.version,
            training_dataset_path=str(
                Path("data/raw") / self.config.data_source
            ),
            training_date_range=(self.config.train_start, self.config.train_end),
            validation_date_range=(self.config.validation_start, self.config.validation_end),
            model_type=self.config.model_type_str,
            feature_config=self.config.feature_config.model_dump(),
            model_config=self.config.model_params,
            target_config={
                "type": "classification",
                "classes": self.config.target.classes or 3,
                "lookahead_bars": self.config.target_lookahead_bars,
                "threshold_pct": self.config.target.threshold_pct,
                "class_weights": getattr(self.config.target, 'class_weights', None),
                "confidence_threshold": getattr(self.config.target, 'confidence_threshold', 0.0),
            },
            validation_metrics={
                "train_accuracy": float(train_accuracy),
                "avg_val_accuracy": float(validation_results['aggregate']['avg_val_accuracy']) if validation_results else 0.0,
            },
            model_artifact_path=model_filename,
            feature_schema=FeatureSchema(
                feature_names=feature_cols,
                ohlcv_columns=self.data_pipeline.ohlcv_columns
            ).to_dict(),
            notes=f"TrainingPipeline - {self.config.description or 'No description'}"
        )

        return metadata

    def _save_training_results(self, metadata, validation_results: Dict) -> None:
        """Save training results to JSON file."""
        results_dir = Path("results/training")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{self.config.strategy_name}_{self.config.version}_{timestamp}.json"

        output = {
            'model_id': metadata.model_id,
            'config': {
                'strategy_name': self.config.strategy_name,
                'version': self.config.version,
                'training_period': {
                    'start': self.config.train_start,
                    'end': self.config.train_end
                },
                'validation_period': {
                    'start': self.config.validation_start,
                    'end': self.config.validation_end
                }
            },
            'training_metrics': {
                'train_accuracy': float(metadata.validation_metrics['train_accuracy']),
            },
            'validation_results': validation_results
        }

        import json
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_file}")

    def _cleanup_temporary_model(self, model_filename: str) -> None:
        """Clean up temporary model file."""
        try:
            Path(model_filename).unlink()
            logger.debug(f"Cleaned up temporary model file: {model_filename}")
        except FileNotFoundError:
            pass  # Already cleaned up
