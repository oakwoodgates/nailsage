"""Structured logging utilities for NailSage.

Provides JSON-formatted logging for production and human-readable
logging for development. Supports different log levels and categories
for different components (data, features, training, backtesting, execution).
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Log categories
CATEGORY_DATA = "data"
CATEGORY_FEATURES = "features"
CATEGORY_TRAINING = "training"
CATEGORY_BACKTEST = "backtest"
CATEGORY_EXECUTION = "execution"
CATEGORY_VALIDATION = "validation"
CATEGORY_SYSTEM = "system"


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as JSON objects with timestamp, level, category,
    message, and any additional context.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "category": getattr(record, "category", CATEGORY_SYSTEM),
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Outputs logs with colors (if supported) and clear formatting.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record for human reading.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with colors
        """
        # Get color for log level
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Get category
        category = getattr(record, "category", CATEGORY_SYSTEM)

        # Build formatted message
        msg = f"{color}[{timestamp}] {record.levelname:8s}{reset} [{category:10s}] {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        # Add extra data if present
        if hasattr(record, "extra_data"):
            msg += f"\n  Data: {record.extra_data}"

        return msg


class CategoryAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds category to log records.

    This allows us to categorize logs by component (data, features, etc.)
    without creating separate loggers.
    """

    def __init__(self, logger: logging.Logger, category: str):
        """
        Initialize adapter.

        Args:
            logger: Base logger
            category: Log category (data, features, training, etc.)
        """
        super().__init__(logger, {})
        self.category = category

    def process(self, msg: str, kwargs: Dict) -> tuple:
        """
        Process log message to add category.

        Args:
            msg: Log message
            kwargs: Additional keyword arguments

        Returns:
            Tuple of (message, kwargs)
        """
        # Add category to extra
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["category"] = self.category

        # Handle extra_data separately
        if "extra_data" in kwargs:
            kwargs["extra"]["extra_data"] = kwargs.pop("extra_data")

        return msg, kwargs


def setup_logger(
    name: str = "nailsage",
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Setup and configure logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files (if None, logs to stdout only)
        json_format: If True, use JSON format; otherwise human-readable

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = HumanReadableFormatter()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create dated log file
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # Always use JSON for file logs
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(
    category: str,
    name: str = "nailsage",
    level: Optional[int] = None,
) -> CategoryAdapter:
    """
    Get logger with specific category.

    Args:
        category: Log category (data, features, training, etc.)
        name: Base logger name
        level: Optional logging level override

    Returns:
        Logger adapter with category
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return CategoryAdapter(logger, category)


# Convenience functions for common categories


def get_data_logger(name: str = "nailsage") -> CategoryAdapter:
    """Get logger for data operations."""
    return get_logger(CATEGORY_DATA, name)


def get_features_logger(name: str = "nailsage") -> CategoryAdapter:
    """Get logger for feature engineering."""
    return get_logger(CATEGORY_FEATURES, name)


def get_training_logger(name: str = "nailsage") -> CategoryAdapter:
    """Get logger for model training."""
    return get_logger(CATEGORY_TRAINING, name)


def get_backtest_logger(name: str = "nailsage") -> CategoryAdapter:
    """Get logger for backtesting."""
    return get_logger(CATEGORY_BACKTEST, name)


def get_execution_logger(name: str = "nailsage") -> CategoryAdapter:
    """Get logger for trade execution."""
    return get_logger(CATEGORY_EXECUTION, name)


def get_validation_logger(name: str = "nailsage") -> CategoryAdapter:
    """Get logger for validation."""
    return get_logger(CATEGORY_VALIDATION, name)
