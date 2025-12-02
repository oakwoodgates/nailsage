"""Risk management module for paper trading.

This module provides risk controls and validation for trading operations:
- Capital allocation and tracking
- Position size limits
- Exposure limits per asset/strategy
- Drawdown monitoring
- Pre-trade risk checks
"""

from execution.risk.capital_allocator import CapitalAllocator
from execution.risk.exposure_tracker import ExposureTracker
from execution.risk.risk_manager import RiskManager, RiskCheckResult

__all__ = [
    "CapitalAllocator",
    "ExposureTracker",
    "RiskManager",
    "RiskCheckResult",
]
