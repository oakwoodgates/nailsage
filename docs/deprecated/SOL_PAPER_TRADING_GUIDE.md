# SOL Swing Momentum Paper Trading Guide

> **⚠️ DEPRECATED**: This guide references `run_sol_paper_trading.py` which has been removed. The system now uses [run_multi_strategy.py](../scripts/run_multi_strategy.py) for all paper trading (Docker-based multi-strategy execution).
>
> **For Docker deployment**, see [DOCKER.md](DOCKER.md).

## Overview

This guide covers running the **profitable SOL swing momentum strategy** in paper trading with production settings.

## Strategy Details

- **Asset**: SOL/USDT
- **Timeframe**: 4 hours
- **Model**: RandomForest
- **Backtest Performance**: +65.54% return, 58% win rate
- **Lookahead**: 6 bars (24 hours)
- **Threshold**: 1.5% price move

## Production Settings

The script uses **production-ready** settings (not test mode):

| Setting | Value | Purpose |
|---------|-------|---------|
| **Confidence Threshold** | 0.5 (50%) | Only trade when model is >50% confident |
| **Cooldown** | 4 bars (16 hours) | Minimum time between trades |
| **Position Size** | $10,000 | Standard position per trade |
| **Fees** | 0.1% | Realistic taker fee |
| **Slippage** | 3 bps | Conservative slippage estimate |

## Prerequisites

1. **Trained Model**: Ensure SOL swing model exists
   ```bash
   python -c "from models.registry import ModelRegistry; r = ModelRegistry(); m = r.get_latest_model('sol_swing_momentum', 'long_term'); print('Model found!' if m else 'Model NOT found - train it first')"
   ```

2. **Kirby WebSocket**: Local Kirby Docker instance running
   ```bash
   # Check if Kirby is accessible
   curl http://localhost:8080/health  # Or your Kirby URL
   ```

3. **Environment Variables**: `.env` file configured
   ```bash
   MODE=dev  # or prod
   WEBSOCKET_URL=ws://localhost:8080/ws
   DB_PATH=execution/state/paper_trading.db
   ```

## Usage

### Dry Run (Recommended First)

Test the setup without executing trades:

```bash
python scripts/run_sol_paper_trading.py --dry-run
```

**What happens**:
- ✅ Connects to WebSocket
- ✅ Receives candle data
- ✅ Runs model inference
- ✅ Generates signals
- ❌ Does NOT execute trades
- ✅ Logs all activity

### Live Paper Trading

Run with real (simulated) trade execution:

```bash
python scripts/run_sol_paper_trading.py
```

**What happens**:
- ✅ All dry-run features
- ✅ Executes simulated trades
- ✅ Tracks positions and P&L
- ✅ Persists state to database

## What to Monitor

### Console Output

The script logs detailed information:

```
[INFO] SOL Swing Momentum Paper Trading
[INFO] Model ID: abc123...
[INFO] Backtest return: 0.6554
[INFO] Subscribed to SOL/USDT 4H
[INFO] Received candle: SOL/USDT @ $142.50
[INFO] Generated signal: LONG (confidence: 68.5%)
[INFO] Executed trade: Buy 68.97 SOL @ $145.00
```

### Database

Check trading activity:

```bash
sqlite3 execution/state/paper_trading.db
```

```sql
-- View active positions
SELECT * FROM positions WHERE status = 'open';

-- View recent trades
SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;

-- View strategy performance
SELECT
    s.strategy_name,
    COUNT(DISTINCT p.id) as total_positions,
    SUM(CASE WHEN p.status = 'closed' THEN p.realized_pnl ELSE 0 END) as total_pnl
FROM strategies s
LEFT JOIN positions p ON s.id = p.strategy_id
GROUP BY s.id;
```

## Validation Checklist

Use this checklist to verify the system is working correctly:

### Phase 1: Connection & Data
- [ ] WebSocket connects successfully
- [ ] Receives 500 historical candles (should take ~30 seconds)
- [ ] Candle buffer populates correctly
- [ ] Database connection established

### Phase 2: Model Inference
- [ ] Model loads without errors
- [ ] Features compute correctly (18 indicators)
- [ ] Predictions generated with confidence scores
- [ ] Prediction cache working (no redundant computations)

### Phase 3: Signal Generation (P0/P1 Features)
- [ ] **Confidence Filtering**: Low confidence predictions suppressed (< 50%)
- [ ] **Signal Deduplication**: Same signal not repeated
- [ ] **Cooldown**: Signals blocked within 16 hours of last trade
- [ ] Neutral signals handled correctly

### Phase 4: Order Execution
- [ ] Orders rejected if below minimum ($10)
- [ ] Orders rejected if above maximum ($50,000)
- [ ] Fill price includes slippage (3 bps)
- [ ] Fees calculated correctly (0.1% of notional)

### Phase 5: Position Tracking
- [ ] Positions saved to database
- [ ] P&L calculated correctly (unrealized and realized)
- [ ] Position status updated (open → closed)
- [ ] Multiple positions tracked independently

### Phase 6: State Persistence
- [ ] Strategies table populated
- [ ] Positions table updated
- [ ] Trades table records all executions
- [ ] Signals table tracks generation history

## Expected Behavior

### First 24 Hours

| Time | Expected Events |
|------|----------------|
| 0-30 min | Load 500 historical candles, warm up feature engine |
| 30 min - 4h | Wait for first candle close |
| 4h | First prediction, possible signal generation |
| 4h - 20h | Cooldown period (no new signals if one generated) |
| 20h+ | Next possible signal (if cooldown elapsed) |

### Typical Signal Frequency

With 4-bar cooldown on 4H timeframe:
- **Maximum**: ~1 signal every 16 hours
- **Realistic**: 1-3 signals per week (depending on market conditions)

## Troubleshooting

### No Signals Generated

**Possible causes**:
1. Model confidence below threshold (< 50%)
2. Cooldown period active
3. Signal deduplication (same signal as last)
4. No candles received (WebSocket issue)

**Debug**:
```bash
# Check signal generator stats
grep "Signal" execution/logs/paper_trading.log

# Check confidence scores
grep "confidence" execution/logs/paper_trading.log | tail -20
```

### WebSocket Connection Fails

**Check**:
1. Kirby is running: `curl http://localhost:8080/health`
2. WebSocket URL correct in `.env`
3. Starlisting ID exists and is correct

### Model Not Found

**Fix**:
```bash
# Train the SOL swing model
python strategies/short_term/train_momentum_classifier.py \
    --config configs/strategies/sol_swing_momentum_v1.yaml
```

## Production Readiness

Before running for extended periods:

1. **Monitor for 48 hours** in dry-run mode
2. **Verify all P0/P1 features** work correctly
3. **Check database** for data integrity
4. **Review logs** for any errors or warnings
5. **Calculate actual performance** vs backtest

## Performance Expectations

Based on backtest results:
- **Expected trades**: 1-2 per week
- **Average win rate**: ~58%
- **Average position duration**: 2-3 days (12-18 bars)
- **Expected volatility**: High (swing trading)

## Safety Features

The script includes multiple safety mechanisms:

1. **Order Size Limits**: $10 min, $50,000 max
2. **Confidence Filtering**: Trades only on high-confidence predictions
3. **Cooldown**: Prevents overtrading
4. **Signal Deduplication**: Avoids repeated signals
5. **Dry-Run Mode**: Test without risk
6. **Database Persistence**: Complete audit trail
7. **Graceful Shutdown**: Ctrl+C saves state

## Next Steps

After validation:
1. Run for 7 days to collect performance data
2. Compare paper trading results vs backtest
3. Analyze signal quality and execution
4. Fine-tune confidence threshold if needed
5. Consider adding stop-loss/take-profit logic

## Support

If you encounter issues:
1. Check logs: `tail -f execution/logs/paper_trading.log`
2. Review database: `sqlite3 execution/state/paper_trading.db`
3. Test WebSocket: `python scripts/test_websocket_connection.py`
4. Validate model: `python scripts/test_model_registry.py`
