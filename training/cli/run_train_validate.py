"""
Unified train + validate CLI entrypoint.

Usage:
    python -m training.cli.run_train_validate --config strategies/dev_scalper_1m_v1.yaml --train-only
"""

import argparse
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.strategy import StrategyConfig  # noqa: E402
from training.pipeline import TrainingPipeline  # noqa: E402
from utils.logger import get_training_logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run training (and validation) for a strategy config.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML config.")
    parser.add_argument("--train-only", action="store_true", help="Skip validation.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving results to disk.")
    args = parser.parse_args()

    logger = get_training_logger()
    cfg = StrategyConfig.from_yaml(args.config)

    pipeline = TrainingPipeline(cfg)
    result = pipeline.run(train_only=args.train_only, save_results=not args.no_save)

    logger.info(f"Completed. Model ID: {result.model_id}")
    logger.info(f"Artifact: {result.model_artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

