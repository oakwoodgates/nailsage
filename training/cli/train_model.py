"""Streamlined model training script using the new TrainingPipeline.

Usage:
    # Train and validate (default - saves results to JSON)
    python training/cli/train_model.py --config strategy-configs/dev_scalper_1m_v1.yaml

    # Train only (no validation)
    python training/cli/train_model.py --config strategy-configs/dev_scalper_1m_v1.yaml --train-only

    # Train and validate without saving JSON
    python training/cli/train_model.py --config strategy-configs/dev_scalper_1m_v1.yaml --no-json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.strategy import StrategyConfig
from training import TrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train model with optional validation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config YAML file"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip validation, only train model"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving results to JSON file (default: save to results/training/)"
    )

    args = parser.parse_args()

    # Load config
    config = StrategyConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration for {config.strategy_name} v{config.version}")

    # Create and run training pipeline
    try:
        pipeline = TrainingPipeline(config)
        result = pipeline.run(
            train_only=args.train_only,
            save_results=not args.no_json
        )

        logger.info("\nTraining completed successfully")
        logger.info(f"Model ID: {result.model_id}")
        logger.info(f"Training accuracy: {result.training_metrics['train_accuracy']:.4f}")

        if result.validation_results:
            agg = result.validation_results['aggregate']
            logger.info(f"Validation accuracy: {agg['avg_val_accuracy']:.4f}")
            logger.info(f"Average return: {agg['avg_total_return']*100:.2f}%")
            logger.info(f"Consistency: {agg['consistency_pct']*100:.1f}%")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())