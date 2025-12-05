"""Streamlined backtesting script using the new BacktestPipeline.

Usage:
    # Run backtest (default - saves results to JSON)
    python scripts/run_backtest.py --config configs/strategies/dev_scalper_1m_v1.yaml --model-id 33b9a1937aacaa4d_20251126_152519_f8835d

    # Run backtest without saving JSON
    python scripts/run_backtest.py --config configs/strategies/dev_scalper_1m_v1.yaml --model-id MODEL_ID --no-json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.strategy import StrategyConfig
from training import BacktestPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtest on trained model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy config YAML file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID to backtest"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving results to JSON file (default: save to results/backtest/)"
    )

    args = parser.parse_args()

    # Load config
    config = StrategyConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration for {config.strategy_name} v{config.version}")

    # Create and run backtest pipeline
    try:
        pipeline = BacktestPipeline(config)
        results = pipeline.run_backtest(
            model_id=args.model_id,
            save_results=not args.no_json
        )

        # Display results
        pipeline.display_results(results)

        logger.info("Backtest completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())