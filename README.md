# nailsage

## ML Trading System: Research & Development Platform
A high-frequency trading research platform for futures markets, built with Scikit-learn and compute-on-the-fly feature engineering. Nailsage is named after the Great Nailsage Sly, trainer of the Nailmasters, from Hollow Knight. 

## 🎯 Project Overview
This is a research and development platform for building, testing, and deploying machine learning trading strategies. The system is designed for rapid experimentation with different technical indicators, timeframes, and model architectures while maintaining rigorous validation practices.

### Core Philosophy
- Research First: Optimized for rapid iteration and hypothesis testing
- Compute-on-the-Fly: Dynamic feature computation eliminates data pipeline bottlenecks
- Multi-Strategy: Support for concurrent development of short-term and long-term strategies
- Validation-Rigorous: Built-in walk-forward validation and realistic backtesting

## 🏗️ System Architecture
### High-Level Data Flow

```
Raw OHLCV Data → Feature Engine → ML Models → Portfolio Manager → Execution
```

### Core Components
1. Feature Engineering Engine
- Dynamic computation of technical indicators during training/prediction
- Strategy-specific configurations for different time horizons
- Lookback-aware calculations to avoid future data leakage
- Cached computations for performance optimization

2. Multi-Strategy Support
- Short-term strategies: 5min-1hr horizons, high leverage, technical patterns
- Long-term strategies: 1day-1week horizons, fundamental + technical factors
- Ability to add any strategy - delta neutral, moon phase, type 5, 6, 7, etc.
- Portfolio-level coordination between strategies

3. Validation Framework
- Walk-forward validation for time-series data
- Realistic backtesting with transaction costs and slippage
- Regime analysis to test strategy robustness
- Parameter stability testing

## 📁 Project Structure

```
ml_trading_research/
├── 📁 data/
│   ├── raw/                 # Original OHLCV data (CSV/Parquet)
│   └── processed/           # (Optional) Precomputed datasets
├── 📁 features/
│   ├── feature_engine.py    # Core dynamic computation engine
│   ├── indicators/          # Technical indicator implementations
│   └── configs.py           # Strategy-specific feature configurations
├── 📁 strategies/
│   ├── short_term/          # High-frequency strategies (0.5-1% targets)
│   │   ├── train.py
│   │   ├── backtest.py
│   │   └── config.yaml
│   ├── long_term/           # Swing strategies (multi-day holds)
│   │   ├── train.py
│   │   ├── backtest.py
│   │   └── config.yaml
|   └── ...
├── 📁 validation/
│   ├── walk_forward.py      # Time-series validation
│   ├── backtest_engine.py   # Realistic backtesting with costs
│   └── regime_analysis.py   # Market condition testing
├── 📁 models/
│   ├── trained/             # Serialized model files (.pkl, .joblib)
│   └── registry.py          # Model versioning and management
├── 📁 execution/
│   ├── paper_trading.py     # Testnet integration
│   └── risk_manager.py      # Position sizing and risk controls
└── 📁 experiments/
    ├── strategy_comparison.ipynb
    └── feature_analysis.ipynb
```

## 🚀 Quick Start
### 1. Environment Setup

```
git clone <repository>
cd ml_trading_research
pip install -r requirements.txt
```

### 2. Data Preparation
```
# Place your OHLCV data in data/raw/
# Supported formats: CSV, Parquet with timestamp, open, high, low, close, volume
```

### 3. Run a Strategy Experiment

```
from features.feature_engine import FeatureEngine
from strategies.short_term.train import ShortTermTrainer

# Initialize with strategy configuration
feature_engine = FeatureEngine('short_term')
trainer = ShortTermTrainer(feature_engine)

# Train and validate
model, validation_results = trainer.run_experiment()
```

### 4. Backtest Your Strategy

```
from strategies.short_term.backtest import BacktestEngine

backtester = BacktestEngine(
    model=model,
    feature_engine=feature_engine,
    transaction_costs=0.0004,  # 0.04% taker fee
    slippage_bps=2             # 2 basis points
)

results = backtester.run()
print(f"Sharpe Ratio: {results['sharpe']}")
print(f"Max Drawdown: {results['max_drawdown']}")
```

## 🔧 Key Implementation Patterns

### Dynamic Feature Computation

```
# Features are computed on-demand, not precomputed
features = feature_engine.compute_features(
    data=ohlcv_df,
    timestamp='2024-01-01 12:00:00',
    lookback=1000
)
# Returns: {'sma_20': 42050, 'rsi_14': 62.5, 'macd': 15.2, ...}
```

### Strategy-Specific Configurations

```
# strategies/short_term/config.yaml
feature_config:
  sma_windows: [5, 10, 20]
  rsi_window: 14
  macd_fast: 8
  macd_slow: 21
  volume_ma_window: 10
  
target_config:
  horizon: 15      # minutes
  threshold: 0.005 # 0.5% move
  
risk_config:
  max_leverage: 10
  position_size: 0.1  # 10% of capital
```

### Multi-Strategy Portfolio

```
from execution.portfolio_manager import PortfolioManager

portfolio = PortfolioManager()
portfolio.add_strategy('short_term', short_term_model, short_term_config)
portfolio.add_strategy('long_term', long_term_model, long_term_config)

# Get coordinated signals
signals = portfolio.get_signals(current_market_data)
```

## 📊 Validation Framework
### Walk-Forward Validation
- Time-series aware splitting
- Multiple validation windows to test robustness
- Performance decay monitoring to detect overfitting

### Realistic Backtesting
- Transaction costs (fees, slippage)
- Realistic execution assumptions
- Leverage and funding costs
- Regime-specific performance analysis

## 🎯 Strategy Development Workflow
1. Feature Experimentation
- Test different technical indicators
- Optimize parameters dynamically
- Validate predictive power

2. Model Training & Validation
- Walk-forward validation
- Hyperparameter tuning
- Overfitting detection

3. Backtesting & Analysis
- Realistic scenario testing
- Drawdown analysis
- Regime robustness checking

4. Paper Trading
- Testnet deployment
- Live performance monitoring
- Infrastructure testing

5. Production Deployment
- Model serialization
- Risk management integration
- Monitoring and retraining

## ⚠️ Important Considerations

### Data Quality
- Clean, gap-free OHLCV data essential
- Timezone consistency across datasets
- Survivorship bias awareness in historical data

### Model Risk
- Overfitting is the primary risk
- Regime changes will affect performance
- Continuous validation required

### Execution Realities
- Slippage and fees significantly impact high-frequency strategies
- Liquidity constraints at scale
- Platform rate limits and infrastructure costs

## 🔮 Next Steps & Advanced Topics
### Planned Enhancements
- Reinforcement Learning integration
- LSTM/Transformer models for sequence prediction
- Alternative data sources (order book, sentiment)
- Automated hyperparameter optimization
- Real-time feature monitoring

### Advanced Research Areas
- Market microstructure modeling
- Cross-asset strategy development
- Regime-switching models
- Bayesian optimization for parameter tuning
