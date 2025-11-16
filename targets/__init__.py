"""
Target variable creation for trading strategies.

This module contains functions for creating target variables from price data,
supporting classification and regression tasks.
"""

from targets.classification import create_3class_target

__all__ = ["create_3class_target"]
