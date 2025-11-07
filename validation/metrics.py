"""Performance metrics for trading strategies."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """
    Container for trading strategy performance metrics.

    Includes risk-adjusted returns, drawdown analysis, and win/loss statistics.
    """

    # Returns
    total_return: float
    annual_return: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: Optional[int]  # in bars

    # Win/Loss statistics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Volatility
    volatility: float
    downside_volatility: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
        }

    def summary(self) -> str:
        """
        Get human-readable summary.

        Returns:
            Formatted summary string
        """
        summary = "Performance Metrics\n"
        summary += "==================\n\n"
        summary += f"Returns:\n"
        summary += f"  Total Return:        {self.total_return:>8.2%}\n"
        summary += f"  Annual Return:       {self.annual_return:>8.2%}\n"
        summary += f"\n"
        summary += f"Risk-Adjusted:\n"
        summary += f"  Sharpe Ratio:        {self.sharpe_ratio:>8.2f}\n"
        summary += f"  Sortino Ratio:       {self.sortino_ratio:>8.2f}\n"
        summary += f"  Calmar Ratio:        {self.calmar_ratio:>8.2f}\n"
        summary += f"\n"
        summary += f"Drawdown:\n"
        summary += f"  Max Drawdown:        {self.max_drawdown:>8.2%}\n"
        if self.max_drawdown_duration:
            summary += f"  Max DD Duration:     {self.max_drawdown_duration:>8} bars\n"
        summary += f"\n"
        summary += f"Win/Loss:\n"
        summary += f"  Win Rate:            {self.win_rate:>8.2%}\n"
        summary += f"  Profit Factor:       {self.profit_factor:>8.2f}\n"
        summary += f"  Avg Win:             {self.avg_win:>8.2%}\n"
        summary += f"  Avg Loss:            {self.avg_loss:>8.2%}\n"
        summary += f"\n"
        summary += f"Trades:\n"
        summary += f"  Total:               {self.total_trades:>8}\n"
        summary += f"  Winning:             {self.winning_trades:>8}\n"
        summary += f"  Losing:              {self.losing_trades:>8}\n"
        summary += f"\n"
        summary += f"Volatility:\n"
        summary += f"  Total:               {self.volatility:>8.2%}\n"
        summary += f"  Downside:            {self.downside_volatility:>8.2%}\n"

        return summary


class MetricsCalculator:
    """
    Calculate trading performance metrics from returns or equity curve.
    """

    @staticmethod
    def calculate_from_returns(
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """
        Calculate metrics from returns series.

        Args:
            returns: Series of period returns
            trades: Optional DataFrame with trade information (profit/loss per trade)
            risk_free_rate: Annual risk-free rate (default: 0%)

        Returns:
            PerformanceMetrics object
        """
        if len(returns) == 0:
            raise ValueError("Cannot calculate metrics from empty returns")

        # Returns
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        periods_per_year = 252  # Trading days (can be adjusted for different timeframes)
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)

        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / periods_per_year
        if returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio
        if downside_volatility > 0:
            sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(periods_per_year)
        else:
            sortino_ratio = 0.0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Max drawdown duration
        dd_duration = MetricsCalculator._calculate_drawdown_duration(drawdown)

        # Calmar Ratio
        if abs(max_drawdown) > 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Trade statistics
        if trades is not None and len(trades) > 0:
            trade_stats = MetricsCalculator._calculate_trade_stats(trades)
        else:
            # Estimate from returns if no trade data
            trade_stats = MetricsCalculator._estimate_trade_stats(returns)

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            volatility=volatility,
            downside_volatility=downside_volatility,
        )

    @staticmethod
    def _calculate_drawdown_duration(drawdown: pd.Series) -> Optional[int]:
        """Calculate maximum drawdown duration in bars."""
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return None

        # Find consecutive drawdown periods
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_durations = in_drawdown.groupby(drawdown_periods).sum()

        return int(drawdown_durations.max()) if len(drawdown_durations) > 0 else None

    @staticmethod
    def _calculate_trade_stats(trades: pd.DataFrame) -> dict:
        """Calculate trade statistics from trades DataFrame."""
        # Expect trades DataFrame to have 'pnl' or 'return' column
        if "pnl" in trades.columns:
            trade_returns = trades["pnl"]
        elif "return" in trades.columns:
            trade_returns = trades["return"]
        else:
            raise ValueError("Trades DataFrame must have 'pnl' or 'return' column")

        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]

        total_trades = len(trade_returns)
        n_winning = len(winning_trades)
        n_losing = len(losing_trades)

        win_rate = n_winning / total_trades if total_trades > 0 else 0.0

        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0

        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0.0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": total_trades,
            "winning_trades": n_winning,
            "losing_trades": n_losing,
        }

    @staticmethod
    def _estimate_trade_stats(returns: pd.Series) -> dict:
        """Estimate trade statistics from returns (when trade data not available)."""
        winning_periods = returns[returns > 0]
        losing_periods = returns[returns < 0]

        total_periods = len(returns[returns != 0])
        n_winning = len(winning_periods)
        n_losing = len(losing_periods)

        win_rate = n_winning / total_periods if total_periods > 0 else 0.0
        avg_win = winning_periods.mean() if len(winning_periods) > 0 else 0.0
        avg_loss = losing_periods.mean() if len(losing_periods) > 0 else 0.0

        total_wins = winning_periods.sum()
        total_losses = abs(losing_periods.sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": total_periods,
            "winning_trades": n_winning,
            "losing_trades": n_losing,
        }
