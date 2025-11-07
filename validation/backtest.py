"""Backtesting engine with realistic transaction costs and execution modeling."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from config.backtest import BacktestConfig
from validation.metrics import MetricsCalculator, PerformanceMetrics
from utils.logger import get_backtest_logger

logger = get_backtest_logger()


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float  # position size in base currency
    pnl: float  # profit/loss in quote currency
    return_pct: float  # return as percentage
    fees: float  # total fees paid
    slippage: float  # slippage cost


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Features:
    - Realistic transaction costs (fees + slippage)
    - Position sizing and leverage
    - Trade tracking and analysis
    - Equity curve generation
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None

        logger.info(
            "Initialized BacktestEngine",
            extra_data={
                "initial_capital": config.initial_capital,
                "transaction_cost": config.get_transaction_cost(),
            },
        )

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        price_column: str = "close",
    ) -> PerformanceMetrics:
        """
        Run backtest on signals.

        Args:
            df: DataFrame with OHLCV data
            signals: Series with signals (-1 = short, 0 = neutral, 1 = long)
            price_column: Column to use for execution prices

        Returns:
            PerformanceMetrics object
        """
        logger.info(f"Running backtest on {len(df)} bars with {signals.sum()} signals")

        # Validate inputs
        if len(df) != len(signals):
            raise ValueError("df and signals must have same length")

        # Initialize
        capital = self.config.initial_capital
        position = 0  # Current position: 0 = flat, >0 = long, <0 = short
        position_size = 0  # Size of position in base currency
        entry_price = 0
        entry_time = None

        equity = [capital]
        timestamps = [df.index[0]]

        # Simulate trading
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            current_price = self._get_execution_price(df.iloc[i], price_column)
            current_time = df.index[i]

            # Check for position exit
            if position != 0 and (
                current_signal != position or current_signal == 0
            ):
                # Exit position
                exit_price = current_price
                pnl, fees, slippage = self._calculate_pnl(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    direction="long" if position > 0 else "short",
                )

                # Record trade
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    direction="long" if position > 0 else "short",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    pnl=pnl,
                    return_pct=(exit_price - entry_price) / entry_price * (1 if position > 0 else -1),
                    fees=fees,
                    slippage=slippage,
                )
                self.trades.append(trade)

                # Update capital
                capital += pnl

                # Reset position
                position = 0
                position_size = 0

            # Check for position entry
            if position == 0 and current_signal != 0:
                # Enter position
                position = current_signal
                entry_price = current_price
                entry_time = current_time

                # Calculate position size
                position_size = self._calculate_position_size(capital, current_price)

            # Update equity
            if position != 0:
                # Mark-to-market current position
                unrealized_pnl = (current_price - entry_price) * position_size * (1 if position > 0 else -1)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity.append(current_equity)
            timestamps.append(current_time)

        # Close any open position at end
        if position != 0:
            exit_price = self._get_execution_price(df.iloc[-1], price_column)
            pnl, fees, slippage = self._calculate_pnl(
                entry_price=entry_price,
                exit_price=exit_price,
                size=position_size,
                direction="long" if position > 0 else "short",
            )

            trade = Trade(
                entry_time=entry_time,
                exit_time=df.index[-1],
                direction="long" if position > 0 else "short",
                entry_price=entry_price,
                exit_price=exit_price,
                size=position_size,
                pnl=pnl,
                return_pct=(exit_price - entry_price) / entry_price * (1 if position > 0 else -1),
                fees=fees,
                slippage=slippage,
            )
            self.trades.append(trade)

        # Create equity curve
        self.equity_curve = pd.Series(equity, index=timestamps)

        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()

        # Create trades DataFrame
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "pnl": t.pnl,
                    "return": t.return_pct,
                }
                for t in self.trades
            ])
        else:
            trades_df = None

        # Calculate metrics
        metrics = MetricsCalculator.calculate_from_returns(returns, trades_df)

        logger.info(
            "Backtest complete",
            extra_data={
                "total_trades": len(self.trades),
                "final_capital": equity[-1],
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
            },
        )

        return metrics

    def _get_execution_price(self, bar: pd.Series, price_column: str) -> float:
        """
        Get execution price for a bar based on fill assumption.

        Args:
            bar: OHLCV bar
            price_column: Column to use

        Returns:
            Execution price
        """
        if self.config.fill_assumption == "close":
            price = bar[price_column]
        elif self.config.fill_assumption == "open":
            price = bar["open"]
        elif self.config.fill_assumption == "midpoint":
            price = (bar["high"] + bar["low"]) / 2
        else:
            price = bar[price_column]

        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        price = price * (1 + slippage_pct)

        return price

    def _calculate_position_size(self, capital: float, price: float) -> float:
        """
        Calculate position size based on capital and configuration.

        Args:
            capital: Available capital
            price: Current price

        Returns:
            Position size in base currency
        """
        # Max position value considering leverage
        max_position_value = capital * self.config.max_position_size

        # Apply leverage if enabled
        if self.config.enable_leverage:
            max_position_value *= self.config.max_leverage

        # Convert to size in base currency
        size = max_position_value / price

        return size

    def _calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        direction: str,
    ) -> tuple[float, float, float]:
        """
        Calculate PnL including fees and slippage.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            direction: 'long' or 'short'

        Returns:
            Tuple of (pnl, fees, slippage)
        """
        # Calculate raw PnL
        if direction == "long":
            raw_pnl = (exit_price - entry_price) * size
        else:
            raw_pnl = (entry_price - exit_price) * size

        # Calculate fees
        entry_value = entry_price * size
        exit_value = exit_price * size
        fee_rate = self.config.get_transaction_cost()
        fees = (entry_value + exit_value) * fee_rate

        # Slippage already included in execution price
        slippage = 0

        # Net PnL
        net_pnl = raw_pnl - fees - slippage

        return net_pnl, fees, slippage

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve from last backtest."""
        if self.equity_curve is None:
            raise ValueError("No backtest has been run yet")
        return self.equity_curve

    def get_trades(self) -> pd.DataFrame:
        """Get trades DataFrame from last backtest."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "fees": t.fees,
                "slippage": t.slippage,
            }
            for t in self.trades
        ])
