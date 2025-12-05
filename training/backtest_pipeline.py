"""Backtesting pipeline for trained models."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import pandas as pd

from config.backtest import BacktestConfig
from config.strategy import StrategyConfig
from models import ModelRegistry
from models.feature_schema import FeatureSchema
from training.data_pipeline import DataPipeline
from training.signal_pipeline import SignalPipeline
from training.validation.backtest import BacktestEngine
from utils.determinism import set_random_seeds, validate_config_consistency
from utils.logger import get_training_logger

logger = get_training_logger()


class BacktestPipeline:
    """
    Orchestrates backtesting of trained models.

    This class handles the complete backtesting pipeline:
    - Model loading from registry
    - Data preparation for backtesting period
    - Feature computation and prediction generation
    - Signal processing and filtering
    - Backtesting execution
    - Results aggregation and reporting

    Attributes:
        config: Strategy configuration
        data_pipeline: Handles data loading and preparation
        signal_pipeline: Handles signal generation and filtering
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize backtest pipeline.

        Args:
            config: Strategy configuration
        """
        validate_config_consistency(config)
        self.config = config
        self.data_pipeline = DataPipeline(config)
        self.signal_pipeline = SignalPipeline(config)

        logger.info("Initialized BacktestPipeline")

    def run_backtest(
        self,
        model_id: str,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete backtest on a trained model.

        Args:
            model_id: Model ID to backtest
            save_results: If True, save results to JSON file

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for model {model_id}")
        set_random_seeds(getattr(self.config, "random_seed", 42))

        # Load model and metadata
        model, metadata = self._load_model_and_metadata(model_id)

        # Load and prepare data
        df_clean, target_clean = self.data_pipeline.load_and_prepare_data()

        # Filter to validation period
        df_val, target_val = self.data_pipeline.prepare_validation_data(
            df_clean, target_clean
        )

        # Prepare features
        feature_cols = self.data_pipeline.get_feature_columns(df_val)
        X_val = df_val[feature_cols]

        # Generate predictions
        predictions, probabilities = self._generate_predictions(model, X_val)

        # Convert to signals
        signals = self._process_signals(predictions, probabilities)

        # Create backtest configuration
        backtest_config = self._create_backtest_config()

        # Run backtest
        metrics = self._run_backtest_execution(df_val, signals, backtest_config)

        # Prepare results
        results = self._prepare_results(
            model_id, metadata, df_val, metrics, backtest_config
        )

        # Save results if requested
        if save_results:
            self._save_backtest_results(results)

        logger.info("Backtest completed successfully")
        return results

    def _load_model_and_metadata(self, model_id: str):
        """Load model and metadata from registry."""
        logger.info(f"Loading model {model_id} from registry...")

        registry = ModelRegistry()
        metadata = registry.get_model(model_id)

        if metadata is None:
            raise ValueError(f"Model not found: {model_id}")

        # Load model artifact
        model_path = Path(metadata.model_artifact_path)
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load feature schema if available
        if metadata.feature_schema:
            feature_schema = FeatureSchema.from_dict(metadata.feature_schema)
            logger.info(f"Feature schema loaded: {len(feature_schema.feature_names)} features")

        return model, metadata

    def _generate_predictions(self, model, X_val: pd.DataFrame):
        """Generate predictions from model."""
        logger.info("Generating predictions...")

        predictions = model.predict(X_val)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_val)

        return predictions, probabilities

    def _process_signals(self, predictions, probabilities):
        """Process predictions into trading signals."""
        num_classes = self._resolve_num_classes()

        signals = self.signal_pipeline.process_signals_for_backtest(
            predictions, probabilities, num_classes
        )

        # Log signal statistics
        signal_stats = self.signal_pipeline.get_signal_statistics(signals)
        logger.info(
            f"Final signals for backtesting: {signal_stats['long_signals']} long, "
            f"{signal_stats['short_signals']} short, {signal_stats['neutral_signals']} neutral"
        )

        return signals

    def _create_backtest_config(self) -> BacktestConfig:
        """Create backtest configuration."""
        if not self.config.backtest:
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

    def _run_backtest_execution(
        self,
        df_val: pd.DataFrame,
        signals,
        backtest_config: BacktestConfig
    ):
        """Execute the backtest."""
        logger.info("Running backtest...")

        backtest_engine = BacktestEngine(backtest_config)

        # Convert signals to pandas Series if needed
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=df_val.index)

        metrics = backtest_engine.run(
            df=df_val,
            signals=signals,
            price_column="close",
        )

        return metrics

    def _prepare_results(
        self,
        model_id: str,
        metadata,
        df_val: pd.DataFrame,
        metrics,
        backtest_config: BacktestConfig
    ) -> Dict[str, Any]:
        """Prepare comprehensive backtest results."""
        results = {
            "model_id": model_id,
            "config": {
                "strategy_name": self.config.strategy_name,
                "version": self.config.version,
                "period": {
                    "start": str(self.config.validation_start),
                    "end": str(self.config.validation_end)
                },
                "initial_capital": backtest_config.initial_capital,
                "leverage": backtest_config.max_leverage,
                "transaction_cost_pct": backtest_config.taker_fee * 100,
                "slippage_bps": backtest_config.slippage_bps,
                "confidence_threshold": getattr(self.config.target, 'confidence_threshold', 0.0),
                "min_bars_between_trades": self.config.backtest.min_bars_between_trades if self.config.backtest else 0,
            },
            "metrics": {
                "total_return": float(metrics.total_return),
                "annual_return": float(metrics.annual_return),
                "sharpe_ratio": float(metrics.sharpe_ratio),
                "sortino_ratio": float(metrics.sortino_ratio),
                "max_drawdown": float(metrics.max_drawdown),
                "win_rate": float(metrics.win_rate),
                "total_trades": int(metrics.total_trades),
                "winning_trades": int(metrics.winning_trades),
                "losing_trades": int(metrics.losing_trades),
                "avg_win": float(metrics.avg_win),
                "avg_loss": float(metrics.avg_loss),
                "profit_factor": float(metrics.profit_factor),
                "volatility": float(metrics.volatility),
                "downside_volatility": float(metrics.downside_volatility),
                "calmar_ratio": float(metrics.calmar_ratio),
            }
        }

        return results

    def _save_backtest_results(self, results: Dict[str, Any]) -> None:
        """Save backtest results to JSON file."""
        results_dir = Path("results/backtest")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{self.config.strategy_name}_{self.config.version}_{timestamp}.json"

        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_file}")

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display formatted backtest results."""
        config = results["config"]
        metrics = results["metrics"]

        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"\nModel: {results['model_id']}")
        print(f"Strategy: {config['strategy_name']} v{config['version']}")
        print(f"Period: {config['period']['start']} to {config['period']['end']}")
        print(f"Initial Capital: ${config['initial_capital']:,.2f}")
        print(f"Leverage: {config['leverage']}x")
        print(f"\nTransaction Costs:")
        print(f"  Fee: {config['transaction_cost_pct']:.3f}%")
        print(f"  Slippage: {config['slippage_bps']} bps")
        print(f"\nSignal Filtering:")
        print(f"  Confidence Threshold: {config['confidence_threshold']:.2f}")
        print(f"  Trade Cooldown: {config['min_bars_between_trades']} bars")

        print(f"\n{'PERFORMANCE METRICS':-^70}")
        print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
        print(f"Annual Return: {metrics['annual_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate: {metrics['win_rate'] * 100:.2f}%")

        print(f"\n{'TRADING STATISTICS':-^70}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\n{'RISK METRICS':-^70}")
        print(f"Volatility: {metrics['volatility'] * 100:.2f}%")
        print(f"Downside Volatility: {metrics['downside_volatility'] * 100:.2f}%")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")

        print("="*70)

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
            raise ValueError("Regression targets are not supported in backtest pipeline.")

        raise ValueError(f"Unsupported target type '{target_type}'")
