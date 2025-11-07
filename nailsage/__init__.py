"""
NailSage - ML Trading Research Platform

A modular platform for building, testing, and deploying machine learning
trading strategies with rigorous validation and production-ready code.
"""

__version__ = "0.1.0"
__author__ = "NailSage Team"

from nailsage import config, data, features, models, strategies, validation, execution, utils

__all__ = [
    "config",
    "data",
    "features",
    "models",
    "strategies",
    "validation",
    "execution",
    "utils",
]
