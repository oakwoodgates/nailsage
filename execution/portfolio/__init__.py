"""
Portfolio coordination and management.

This module handles coordinating signals from multiple trading strategies,
tracking positions, and enforcing risk limits.

Phase 1: Simple pass-through coordinator with basic safety checks
Phase 2: Advanced coordination with portfolio optimization
"""

from execution.portfolio.coordinator import PortfolioCoordinator
from execution.portfolio.position import Position
from execution.portfolio.signal import StrategySignal

__all__ = [
    "PortfolioCoordinator",
    "Position",
    "StrategySignal",
]
