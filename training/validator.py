"""Validation orchestration for training pipelines."""

import logging
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from config.backtest import BacktestConfig
from config.strategy import StrategyConfig
from training.validation.time_series_split import TimeSeriesSplitter
from training.validation.backtest import BacktestEngine
from utils.determinism import set_random_seeds, validate_config_consistency
from utils.logger import get_training_logger

logger = get_training_logger()


class Validator:
    """
    Orchestrates model validation using walk-forward analysis.

    This class handles the complete validation pipeline:
    - Time series splitting
    - Walk-forward validation across splits
    - Backtesting on each split
    - Aggregate metrics calculation

    Attributes:
        config: Strategy configuration
        splitter: Time series splitter for validation
        backtest_config: Backtesting configuration
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize validator.

        Args:
            config: Strategy configuration
        """
        self.config = config
        validate_config_consistency(config)

        # Create validation configuration
        validation_config = config.validation
        if validation_config is None:
            method = 'walk_forward'
            n_splits = 4
            expanding_window = True
            gap_bars = 4
        else:
            method = getattr(validation_config, 'method', 'walk_forward')
            n_splits = getattr(validation_config, 'n_splits', 4)
            expanding_window = getattr(validation_config, 'expanding_window', True)
            gap_bars = getattr(validation_config, 'gap_bars', 4)

        # Create time series splitter
        self.splitter = TimeSeriesSplitter(
            n_splits=n_splits,
            expanding_window=expanding_window,
            gap_bars=gap_bars,
        )

        # Create backtest configuration
        self.backtest_config = self._create_backtest_config()

        logger.info(f"Initialized Validator with {n_splits} splits, {method} method")

    def _create_backtest_config(self) -> BacktestConfig:
        """Create backtest configuration from strategy config."""
        if not self.config.backtest:
            # Default backtest config
            return BacktestConfig()

        return BacktestConfig(
            initial_capital=self.config.backtest.capital,
            taker_fee=self.config.backtest.transaction_cost_pct / 100.0,
            slippage_bps=self.config.backtest.slippage_bps,
            enable_leverage=self.config.backtest.leverage > 1,
            max_leverage=float(self.config.backtest.leverage),
            max_position_size=getattr(self.config.backtest, 'max_position_size', 1.0),
            funding_rate_annual=getattr(self.config.backtest, 'funding_rate_annual', 0.0),
            fill_assumption=getattr(self.config.backtest, 'fill_assumption', 'close'),
            assume_maker=getattr(self.config.backtest, 'assume_maker', False),
            maker_fee=getattr(self.config.backtest, 'maker_fee', 0.0002),
        )

    def run_validation(
        self,
        model,
        df_clean: pd.DataFrame,
        target_clean: pd.Series,
        feature_cols: list
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward validation.

        Args:
            model: Trained model
            df_clean: Cleaned feature DataFrame
            target_clean: Cleaned target series
            feature_cols: Feature column names

        Returns:
            Validation results dictionary
        """
        logger.info("="*70)
        logger.info("RUNNING WALK-FORWARD VALIDATION")
        logger.info("="*70)
        set_random_seeds()
        t_start = time.perf_counter()

        # Generate splits
        splits = self.splitter.split(df_clean, timestamp_column='timestamp')
        logger.info(f"Generated {len(splits)} validation splits")

        # Run validation on each split
        split_results = []
        for split in splits:
            logger.info(f"\n{'='*70}")
            logger.info(f"Split {split.split_index + 1}/{len(splits)}")
            logger.info(f"{'='*70}")

            split_result = self._validate_single_split(
                df_clean, target_clean, feature_cols, split
            )
            split_results.append(split_result)

        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(split_results)

        validation_results = {
            'n_splits': len(split_results),
            'splits': split_results,
            'aggregate': aggregate_results,
        }

        # Log aggregate results
        self._log_aggregate_results(aggregate_results)
        t_end = time.perf_counter()
        logger.info(f"Validation runtime: {t_end - t_start:.2f}s")

        return validation_results

    def _validate_single_split(
        self,
        df_clean: pd.DataFrame,
        target_clean: pd.Series,
        feature_cols: list,
        split
    ) -> Dict[str, Any]:
        """Validate model on a single split."""
        # Get train/val indices for this split
        train_idx, val_idx = self.splitter.get_train_val_indices(
            df_clean, split, timestamp_column='timestamp'
        )

        # Prepare train/val sets
        X_split_train = df_clean.iloc[train_idx][feature_cols]
        y_split_train = target_clean.iloc[train_idx]
        X_split_val = df_clean.iloc[val_idx][feature_cols]
        y_split_val = target_clean.iloc[val_idx]

        logger.info(f"Split validation samples: {len(X_split_val):,}")

        # Train fresh model for this split
        model = self._create_model()
        sample_weights = self._calculate_sample_weights(y_split_train)
        fit_kwargs = {'X': X_split_train, 'y': y_split_train}
        if sample_weights is not None:
            fit_kwargs['sample_weight'] = sample_weights
        model.fit(**fit_kwargs)

        # Generate predictions
        predictions = model.predict(X_split_val)
        probabilities = model.predict_proba(X_split_val) if hasattr(model, 'predict_proba') else None

        # Calculate accuracy
        val_acc = float((predictions == y_split_val).mean())
        logger.info(f"Validation accuracy: {val_acc:.4f}")

        # Convert to signals and run backtest
        signals = self._convert_to_signals_with_config(predictions, probabilities)
        backtest_metrics = self._run_backtest_on_split(df_clean.iloc[val_idx], signals)

        # Log split results
        logger.info("Split Results:")
        logger.info(f"  Accuracy: {val_acc:.4f}")
        logger.info(f"  Total Return: {backtest_metrics.total_return * 100:.2f}%")
        logger.info(f"  Sharpe Ratio: {backtest_metrics.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {backtest_metrics.max_drawdown * 100:.2f}%")
        logger.info(f"  Win Rate: {backtest_metrics.win_rate * 100:.2f}%")
        logger.info(f"  Total Trades: {backtest_metrics.total_trades}")

        return {
            'split': split.split_index + 1,
            'period': {
                'start': str(split.val_start),
                'end': str(split.val_end)
            },
            'val_metrics': {
                'accuracy': val_acc,
                'precision': float(self._safe_metric(probabilities, y_split_val, metric="precision")),
                'recall': float(self._safe_metric(probabilities, y_split_val, metric="recall")),
                'f1': float(self._safe_metric(probabilities, y_split_val, metric="f1")),
            },
            'backtest_metrics': {
                'total_return': float(backtest_metrics.total_return),
                'annual_return': float(backtest_metrics.annual_return),
                'sharpe_ratio': float(backtest_metrics.sharpe_ratio),
                'sortino_ratio': float(backtest_metrics.sortino_ratio),
                'max_drawdown': float(backtest_metrics.max_drawdown),
                'win_rate': float(backtest_metrics.win_rate),
                'total_trades': int(backtest_metrics.total_trades),
                'profit_factor': float(backtest_metrics.profit_factor),
            }
        }

    def _convert_to_signals_with_config(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray]
    ) -> pd.Series:
        """Convert predictions to signals using configuration."""
        from training.signal_pipeline import SignalPipeline

        signal_pipeline = SignalPipeline(self.config)
        num_classes = self._resolve_num_classes()

        signals = signal_pipeline.process_signals_for_backtest(
            predictions, probabilities, num_classes
        )

        # Convert to pandas Series (expected by BacktestEngine)
        return pd.Series(signals)

    def _run_backtest_on_split(self, df_split: pd.DataFrame, signals: pd.Series):
        """Run backtest on a single split."""
        backtest_engine = BacktestEngine(self.backtest_config)
        metrics = backtest_engine.run(df_split, signals, price_column="close")
        return metrics

    def _calculate_aggregate_metrics(self, split_results: list) -> Dict[str, Any]:
        """Calculate aggregate metrics across all splits."""
        if not split_results:
            return {}

        # Extract metrics
        val_accuracies = [r['val_metrics']['accuracy'] for r in split_results]
        precisions = [r['val_metrics']['precision'] for r in split_results]
        recalls = [r['val_metrics']['recall'] for r in split_results]
        f1s = [r['val_metrics']['f1'] for r in split_results]
        total_returns = [r['backtest_metrics']['total_return'] for r in split_results]
        sharpe_ratios = [r['backtest_metrics']['sharpe_ratio'] for r in split_results]
        max_drawdowns = [r['backtest_metrics']['max_drawdown'] for r in split_results]
        win_rates = [r['backtest_metrics']['win_rate'] for r in split_results]
        total_trades = sum([r['backtest_metrics']['total_trades'] for r in split_results])

        # Calculate aggregates
        avg_val_acc = float(np.mean(val_accuracies))
        avg_return = float(np.mean(total_returns))
        avg_sharpe = float(np.mean(sharpe_ratios))
        avg_max_dd = float(np.mean(max_drawdowns))
        avg_win_rate = float(np.mean(win_rates))

        positive_splits = sum(1 for r in split_results if r['backtest_metrics']['total_return'] > 0)
        consistency_pct = positive_splits / len(split_results)

        return {
            'avg_val_accuracy': avg_val_acc,
            'avg_precision': float(np.mean(precisions)),
            'avg_recall': float(np.mean(recalls)),
            'avg_f1': float(np.mean(f1s)),
            'avg_total_return': avg_return,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_max_drawdown': avg_max_dd,
            'avg_win_rate': avg_win_rate,
            'total_trades': int(total_trades),
            'positive_splits': int(positive_splits),
            'consistency_pct': float(consistency_pct),
            'n_splits': len(split_results),
        }

    def _log_aggregate_results(self, aggregate: Dict[str, Any]) -> None:
        """Log aggregate validation results."""
        logger.info(f"\n{'='*70}")
        logger.info("AGGREGATE VALIDATION RESULTS")
        logger.info(f"{'='*70}")

        logger.info(f"Average Validation Accuracy: {aggregate['avg_val_accuracy']:.4f}")
        logger.info(f"Average Precision: {aggregate['avg_precision']:.4f}")
        logger.info(f"Average Recall: {aggregate['avg_recall']:.4f}")
        logger.info(f"Average F1: {aggregate['avg_f1']:.4f}")
        logger.info(f"Average Total Return: {aggregate['avg_total_return'] * 100:.2f}%")
        logger.info(f"Average Sharpe Ratio: {aggregate['avg_sharpe_ratio']:.2f}")
        logger.info(f"Average Max Drawdown: {aggregate['avg_max_drawdown'] * 100:.2f}%")
        logger.info(f"Average Win Rate: {aggregate['avg_win_rate'] * 100:.2f}%")
        logger.info(f"Total Trades (all splits): {aggregate['total_trades']}")
        n_splits = aggregate.get('n_splits', getattr(self.splitter, "n_splits", 0))
        logger.info(f"Consistency: {aggregate['consistency_pct'] * 100:.1f}% ({aggregate['positive_splits']}/{n_splits} splits profitable)")
        logger.info(f"{'='*70}")

    def _create_model(self):
        """Factory to create ML model based on type."""
        model_type = self.config.model.type
        params = self.config.model.params

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

    def _calculate_sample_weights(self, y_train: pd.Series):
        """Calculate sample weights if class_weights provided."""
        if not hasattr(self.config.target, 'class_weights') or not self.config.target.class_weights:
            return None

        class_weights = self.config.target.class_weights
        sample_weights = np.array([class_weights[label] for label in y_train])

        logger.info(f"Using class weights: {class_weights}")
        return sample_weights

    def _safe_metric(self, probabilities, y_true, metric: str):
        """Compute simple precision/recall/f1 for binary/3-class using argmax predictions."""
        from sklearn.metrics import precision_score, recall_score, f1_score

        if probabilities is None:
            return 0.0

        y_pred = probabilities.argmax(axis=1)
        average = "binary" if len(set(y_true)) <= 2 else "macro"

        try:
            if metric == "precision":
                return precision_score(y_true, y_pred, average=average, zero_division=0)
            if metric == "recall":
                return recall_score(y_true, y_pred, average=average, zero_division=0)
            if metric == "f1":
                return f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception:
            return 0.0

        return 0.0

    def _resolve_num_classes(self) -> int:
        """Determine number of classes from target configuration."""
        target_type = getattr(self.config.target, "type", "").lower()
        explicit_classes = getattr(self.config.target, "classes", None)

        if explicit_classes:
            return explicit_classes

        if target_type in ("binary", "2class", "classification_2class"):
            return 2
        if target_type in ("", "3class", "classification_3class", "classification"):
            return 3
        if target_type == "regression":
            raise ValueError("Regression targets are not supported in validation pipeline.")

        raise ValueError(f"Unsupported target type '{target_type}'")
