"""Walk-forward validation for time series strategies."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np

from config.backtest import BacktestConfig
from training.validation.time_series_split import TimeSeriesSplitter, TimeSeriesSplit
from training.validation.backtest import BacktestEngine
from training.validation.metrics import PerformanceMetrics, AccuracyMetrics, MetricsCalculator
from utils.logger import get_validation_logger

logger = get_validation_logger()


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward validation split."""

    split_index: int
    split_info: TimeSeriesSplit
    train_metrics: Optional[PerformanceMetrics]
    val_metrics: PerformanceMetrics
    train_accuracy: Optional[AccuracyMetrics]  # Training accuracy metrics
    val_accuracy: Optional[AccuracyMetrics]    # Validation accuracy metrics
    model: Any  # Trained model object
    feature_importance: Optional[Dict[str, float]] = None


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.

    Performs time-series cross-validation with proper train/validation splits,
    trains models on each split, and evaluates performance.
    """

    def __init__(
        self,
        splitter: TimeSeriesSplitter,
        backtest_config: BacktestConfig,
    ):
        """
        Initialize walk-forward validator.

        Args:
            splitter: TimeSeriesSplitter for generating train/val splits
            backtest_config: Configuration for backtesting
        """
        self.splitter = splitter
        self.backtest_config = backtest_config
        self.results: List[WalkForwardResult] = []

        logger.info("Initialized WalkForwardValidator")

    def validate(
        self,
        df: pd.DataFrame,
        train_func: Callable,
        predict_func: Callable,
        timestamp_column: str = "timestamp",
        target_column: str = "target",
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward validation.

        Args:
            df: DataFrame with features and target
            train_func: Function to train model: train_func(X_train, y_train) -> model
            predict_func: Function to predict: predict_func(model, X) -> predictions
            timestamp_column: Name of timestamp column
            target_column: Name of target column

        Returns:
            List of WalkForwardResult objects
        """
        logger.info("Starting walk-forward validation")

        # Generate splits
        splits = self.splitter.split(df, timestamp_column)

        logger.info(f"Generated {len(splits)} splits for validation")

        # Validate each split
        for split in splits:
            logger.info(f"Processing {split}")

            # Get train and validation indices
            train_idx, val_idx = self.splitter.get_train_val_indices(
                df, split, timestamp_column
            )

            # Get feature columns (everything except timestamp and target)
            feature_cols = [
                col for col in df.columns
                if col not in [timestamp_column, target_column]
            ]

            # Prepare data
            X_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx][target_column]
            X_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx][target_column]

            # Handle NaN values
            # For training, drop rows with NaN
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train = X_train[train_valid]
            y_train = y_train[train_valid]

            # For validation, we need to keep the time alignment
            # So we'll fill NaN with a strategy or skip those predictions
            val_valid = ~(X_val.isna().any(axis=1) | y_val.isna())

            logger.info(
                f"Split {split.split_index}: "
                f"Train samples={len(X_train)}, Val samples={len(X_val)}, "
                f"Val valid={val_valid.sum()}"
            )

            # Train model
            try:
                model = train_func(X_train, y_train)
            except Exception as e:
                logger.error(f"Training failed for split {split.split_index}: {e}")
                raise

            # Generate predictions
            try:
                train_preds = predict_func(model, X_train)
                val_preds = predict_func(model, X_val[val_valid])
            except Exception as e:
                logger.error(f"Prediction failed for split {split.split_index}: {e}")
                raise

            # Calculate accuracy metrics
            train_accuracy = MetricsCalculator.calculate_accuracy_metrics(y_train, train_preds)
            val_accuracy = MetricsCalculator.calculate_accuracy_metrics(y_val[val_valid], val_preds)

            # Convert predictions to trading signals
            # Assuming predictions are class labels: -1 (short), 0 (neutral), 1 (long)
            # Or probabilities that need to be thresholded
            train_signals = self._convert_to_signals(train_preds)
            val_signals_valid = self._convert_to_signals(val_preds)

            # Create full validation signals (with NaN for invalid rows)
            val_signals = pd.Series(0, index=X_val.index)
            val_signals[val_valid] = val_signals_valid

            # Backtest on training data (optional - to detect overfitting)
            train_metrics = None
            if len(train_signals) > 0:
                try:
                    backtest_train = BacktestEngine(self.backtest_config)
                    train_df = df.iloc[train_idx]
                    train_metrics = backtest_train.run(
                        train_df,
                        pd.Series(train_signals, index=train_df.index),
                    )
                except Exception as e:
                    logger.warning(f"Train backtest failed: {e}")

            # Backtest on validation data
            try:
                backtest_val = BacktestEngine(self.backtest_config)
                val_df = df.iloc[val_idx]
                val_metrics = backtest_val.run(val_df, val_signals)
            except Exception as e:
                logger.error(f"Validation backtest failed for split {split.split_index}: {e}")
                raise

            # Extract feature importance if available
            feature_importance = None
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(
                    zip(feature_cols, model.feature_importances_)
                )
            elif hasattr(model, "coef_"):
                feature_importance = dict(
                    zip(feature_cols, np.abs(model.coef_).flatten())
                )

            # Store result
            result = WalkForwardResult(
                split_index=split.split_index,
                split_info=split,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                model=model,
                feature_importance=feature_importance,
            )
            self.results.append(result)

            logger.info(
                f"Split {split.split_index} complete: "
                f"Train Acc={train_accuracy.accuracy:.2%}, "
                f"Val Acc={val_accuracy.accuracy:.2%}, "
                f"Val Sharpe={val_metrics.sharpe_ratio:.2f}, "
                f"Val Return={val_metrics.total_return:.2%}"
            )

        logger.info(f"Walk-forward validation complete: {len(self.results)}/{len(splits)} splits processed")
        return self.results

    def _convert_to_signals(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert model predictions to trading signals.

        Args:
            predictions: Model predictions

        Returns:
            Trading signals (-1, 0, 1)
        """
        # If predictions are already class labels (-1, 0, 1), return as is
        if np.all(np.isin(predictions, [-1, 0, 1])):
            return predictions

        # If predictions are probabilities or continuous values,
        # we need to threshold them
        # For now, simple strategy: positive = long, negative = short, near zero = neutral
        signals = np.sign(predictions)

        # Optional: add neutral zone
        neutral_threshold = 0.1
        signals[np.abs(predictions) < neutral_threshold] = 0

        return signals

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics across all validation splits.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.results:
            raise ValueError("No validation results available")

        val_returns = [r.val_metrics.total_return for r in self.results]
        val_sharpes = [r.val_metrics.sharpe_ratio for r in self.results]
        val_drawdowns = [r.val_metrics.max_drawdown for r in self.results]
        val_win_rates = [r.val_metrics.win_rate for r in self.results]

        aggregate = {
            "n_splits": len(self.results),
            "mean_return": np.mean(val_returns),
            "std_return": np.std(val_returns),
            "mean_sharpe": np.mean(val_sharpes),
            "std_sharpe": np.std(val_sharpes),
            "mean_max_drawdown": np.mean(val_drawdowns),
            "worst_drawdown": min(val_drawdowns),
            "mean_win_rate": np.mean(val_win_rates),
            "consistency_score": self._calculate_consistency_score(),
        }

        # Check for overfitting (if train metrics available)
        train_metrics_available = all(r.train_metrics is not None for r in self.results)
        if train_metrics_available:
            train_returns = [r.train_metrics.total_return for r in self.results]
            train_sharpes = [r.train_metrics.sharpe_ratio for r in self.results]

            aggregate["train_mean_return"] = np.mean(train_returns)
            aggregate["train_mean_sharpe"] = np.mean(train_sharpes)
            aggregate["overfit_gap_return"] = np.mean(train_returns) - np.mean(val_returns)
            aggregate["overfit_gap_sharpe"] = np.mean(train_sharpes) - np.mean(val_sharpes)

        return aggregate

    def _calculate_consistency_score(self) -> float:
        """
        Calculate consistency score across validation splits.

        A strategy is consistent if it performs well across all splits,
        not just on average.

        Returns:
            Consistency score (0-1, higher is better)
        """
        if not self.results:
            return 0.0

        # Count how many splits have positive returns
        positive_returns = sum(1 for r in self.results if r.val_metrics.total_return > 0)
        consistency = positive_returns / len(self.results)

        return consistency

    def get_summary(self) -> str:
        """
        Get summary of walk-forward validation results.

        Returns:
            Formatted summary string
        """
        if not self.results:
            return "No validation results available"

        aggregate = self.get_aggregate_metrics()

        summary = "Walk-Forward Validation Summary\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Number of Splits: {aggregate['n_splits']}\n\n"
        summary += "Validation Performance (across splits):\n"
        summary += f"  Mean Return:         {aggregate['mean_return']:>8.2%} ± {aggregate['std_return']:.2%}\n"
        summary += f"  Mean Sharpe:         {aggregate['mean_sharpe']:>8.2f} ± {aggregate['std_sharpe']:.2f}\n"
        summary += f"  Mean Max Drawdown:   {aggregate['mean_max_drawdown']:>8.2%}\n"
        summary += f"  Worst Drawdown:      {aggregate['worst_drawdown']:>8.2%}\n"
        summary += f"  Mean Win Rate:       {aggregate['mean_win_rate']:>8.2%}\n"
        summary += f"  Consistency Score:   {aggregate['consistency_score']:>8.2%}\n"

        if "overfit_gap_return" in aggregate:
            summary += "\nOverfitting Analysis:\n"
            summary += f"  Train Mean Return:   {aggregate['train_mean_return']:>8.2%}\n"
            summary += f"  Gap (Train - Val):   {aggregate['overfit_gap_return']:>8.2%}\n"
            summary += f"  Train Mean Sharpe:   {aggregate['train_mean_sharpe']:>8.2f}\n"
            summary += f"  Gap (Train - Val):   {aggregate['overfit_gap_sharpe']:>8.2f}\n"

        summary += "\nIndividual Split Results:\n"
        for r in self.results:
            summary += (
                f"  Split {r.split_index}: "
                f"Return={r.val_metrics.total_return:>7.2%}, "
                f"Sharpe={r.val_metrics.sharpe_ratio:>6.2f}, "
                f"Drawdown={r.val_metrics.max_drawdown:>7.2%}\n"
            )

        return summary
