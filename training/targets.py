"""Classification target variable creation."""

import pandas as pd


def create_binary_target(
    df: pd.DataFrame,
    lookahead_bars: int,
    threshold_pct: float = 0.0
) -> pd.Series:
    """
    Create binary target variable (Long/Short only, no neutral).

    Classifies future returns into two categories:
    - Short (0): Price drops (or drops more than threshold if threshold > 0)
    - Long (1): Price rises (or rises more than threshold if threshold > 0)

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        lookahead_bars: How many bars ahead to look for price movement
        threshold_pct: Optional minimum threshold. If 0, any positive = long, any negative = short.
                      If > 0, returns within threshold become NaN (filtered out).

    Returns:
        Series with target labels: 0 (short), 1 (long). NaN for filtered rows.
    """
    # Calculate future returns
    future_returns = df['close'].pct_change(periods=lookahead_bars).shift(-lookahead_bars)

    # Binary classification
    if threshold_pct > 0:
        # With threshold: filter out small moves
        threshold = threshold_pct / 100
        target = pd.Series(index=df.index, dtype=float)
        target[future_returns > threshold] = 1  # Long
        target[future_returns < -threshold] = 0  # Short
        # Moves within threshold stay NaN (will be dropped)
    else:
        # Simple: positive = long, negative = short
        target = (future_returns > 0).astype(int)
        target[future_returns == 0] = pd.NA  # Exactly zero is ambiguous

    return target


def create_3class_target(
    df: pd.DataFrame,
    lookahead_bars: int,
    threshold_pct: float
) -> pd.Series:
    """
    Create 3-class target variable based on future price movement.

    Classifies future returns into three categories:
    - Short (0): Price drops more than threshold_pct
    - Neutral (1): Price moves within ±threshold_pct
    - Long (2): Price rises more than threshold_pct

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        lookahead_bars: How many bars ahead to look for price movement
        threshold_pct: Percentage threshold for classification (e.g., 0.5 for 0.5%)

    Returns:
        Series with target labels: 0 (short), 1 (neutral), 2 (long)

    Example:
        >>> df = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
        >>> target = create_3class_target(df, lookahead_bars=1, threshold_pct=0.5)
        >>> # Returns series with 0/1/2 based on >0.5% future price changes
    """
    # Calculate future returns
    future_returns = df['close'].pct_change(periods=lookahead_bars).shift(-lookahead_bars)

    # Classify based on threshold
    # XGBoost requires class labels to be 0, 1, 2 for 3-class classification
    target = pd.Series(1, index=df.index)  # Default: neutral (1)
    target[future_returns > threshold_pct / 100] = 2  # Long (2)
    target[future_returns < -threshold_pct / 100] = 0  # Short (0)

    return target
