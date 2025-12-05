"""Determinism utilities for reproducible ML training."""

import os
import random
import numpy as np
from typing import Optional


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducible results across Python, NumPy, and common ML libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Try to set seeds for common ML libraries (optional)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def validate_config_consistency(config) -> None:
    """
    Validate configuration for consistency and required fields.

    Args:
        config: Configuration object to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check for required strategy config fields
    if hasattr(config, 'target'):
        target_config = config.target
        if hasattr(target_config, 'type'):
            valid_types = ['binary', '3class', 'regression', 'classification', 'classification_2class', 'classification_3class', '']
            if target_config.type not in valid_types:
                raise ValueError(f"Invalid target type '{target_config.type}'. Must be one of {valid_types}")

        # Validate class counts for classification targets
        if hasattr(target_config, 'type') and target_config.type in ['binary', '3class', 'classification', 'classification_2class', 'classification_3class', '']:
            if not hasattr(target_config, 'lookahead_bars') or target_config.lookahead_bars <= 0:
                raise ValueError("lookahead_bars must be positive for classification targets")

    # Check backtest config consistency
    if hasattr(config, 'backtest'):
        backtest_config = config.backtest
        if hasattr(backtest_config, 'leverage') and backtest_config.leverage > 1:
            if not hasattr(backtest_config, 'enable_leverage') or not backtest_config.enable_leverage:
                raise ValueError("enable_leverage must be True when leverage > 1")

    # Check model config
    if hasattr(config, 'model'):
        model_config = config.model
        if hasattr(model_config, 'type'):
            # Could add validation for supported model types
            pass
