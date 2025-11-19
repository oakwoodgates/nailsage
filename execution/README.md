# Paper Trading Execution Module

Real-time paper trading engine for NailSage that connects to Kirby API for live market data and simulates trading with trained ML models.

## Architecture

```
Kirby WebSocket API
        ↓
WebSocket Client (async)
        ↓
Candle Buffer (ring buffer)
        ↓
Feature Engine (real-time)
        ↓
Model Predictor (async inference)
        ↓
Signal Generator
        ↓
Portfolio Coordinator
        ↓
Order Simulator (fees/slippage)
        ↓
Position Tracker
        ↓
State Manager (SQLite)
```

## Module Structure

```
execution/
├── websocket/          # Kirby API WebSocket client
│   ├── models.py       # Pydantic message models
│   └── client.py       # Async WebSocket client
│
├── streaming/          # Real-time data processing
│   └── candle_buffer.py # Ring buffer for candles
│
├── persistence/        # State management
│   ├── schema.sql      # SQLite schema
│   └── state_manager.py # Database interface
│
├── inference/          # Model prediction (TODO)
│   ├── predictor.py
│   └── signal_generator.py
│
├── simulator/          # Order execution simulation (TODO)
│   └── order_executor.py
│
├── tracking/           # Position management (TODO)
│   └── position_tracker.py
│
├── runner/             # Orchestration (TODO)
│   ├── live_strategy.py
│   └── paper_trading_coordinator.py
│
└── state/              # SQLite databases (gitignored)
    └── paper_trading.db
```

## Configuration

Configuration is loaded from `.env` file using pydantic-settings:

```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env
```

### Environment Variables

```bash
# Environment mode (dev or prod)
MODE=dev

# Kirby WebSocket URLs
KIRBY_WS_URL_DEV=ws://localhost:8000/ws
KIRBY_WS_URL_PRO=ws://143.198.18.115:8000/ws

# Kirby API Keys (starts with kb_)
KIRBY_API_KEY_DEV=kb_your_dev_key_here
KIRBY_API_KEY_PRO=kb_your_prod_key_here

# Starlisting ID mappings
STARLISTING_BTC_USDT_15M=1
STARLISTING_SOL_USDT_4H=2

# Paper trading settings
PAPER_TRADING_INITIAL_CAPITAL=100000.0
PAPER_TRADING_DB_PATH=execution/state/paper_trading.db
PAPER_TRADING_LOG_LEVEL=INFO
PAPER_TRADING_SNAPSHOT_INTERVAL=60
```

## Usage

### Testing WebSocket Integration

```bash
# Run integration test
python scripts/test_websocket_integration.py

# Expected output:
# ✓ Connected to Kirby WebSocket
# ✓ Subscribed to starlisting 1
# ✓ Received 95 candles in 60 seconds
# ✓ Buffer and database working
```

### WebSocket Client Example

```python
import asyncio
from config.paper_trading import load_paper_trading_config
from execution.websocket.client import KirbyWebSocketClient
from execution.streaming.candle_buffer import MultiCandleBuffer

async def on_candle(candle_update):
    """Handle incoming candle."""
    print(f"Received: {candle_update.candle}")

async def main():
    # Load config
    config = load_paper_trading_config()

    # Create client and buffer
    client = KirbyWebSocketClient(config.websocket)
    buffer = MultiCandleBuffer(maxlen=500)

    # Register callback
    client.on_candle_update(on_candle)

    # Connect and subscribe
    await client.connect()
    await client.subscribe(starlisting_id=1, historical_candles=50)

    # Run until interrupted
    await client.wait_until_closed()

asyncio.run(main())
```

## WebSocket Client Features

### Connection Management
- ✅ Async connection with authentication (API key)
- ✅ Industry-standard exponential backoff reconnection
- ✅ Heartbeat monitoring (30s interval, 45s timeout)
- ✅ Automatic resubscription after reconnection

### Message Handling
- ✅ Candle updates (OHLCV data)
- ✅ Success/error messages
- ⚠️ Funding rate updates (partial - format needs fix)
- ⚠️ Open interest updates (partial - format needs fix)
- ❌ Ping/pong messages (not yet supported)

### Reconnection Strategy
```python
# Default settings (configurable)
reconnect_enabled = True
reconnect_max_attempts = 0  # Infinite
reconnect_initial_delay = 1.0  # seconds
reconnect_max_delay = 60.0  # seconds
reconnect_backoff_multiplier = 2.0  # Exponential
```

Delays: 1s → 2s → 4s → 8s → 16s → 32s → 60s (max)

## Candle Buffer

Thread-safe ring buffer for efficient candle storage:

```python
from execution.streaming.candle_buffer import CandleBuffer, MultiCandleBuffer

# Single buffer
buffer = CandleBuffer(maxlen=500, starlisting_id=1, interval="15m")
buffer.add(candle)
latest = buffer.get_latest()
df = buffer.to_dataframe()  # Convert to pandas

# Multi-buffer (manages multiple starlistings)
multi_buffer = MultiCandleBuffer(maxlen=500)
multi_buffer.add_candle(candle, starlisting_id=1, interval="15m")
buffer = multi_buffer.get(starlisting_id=1)
```

### Features
- ✅ Fixed-size sliding window (automatic eviction)
- ✅ O(1) append/pop operations (deque)
- ✅ Thread-safe with locks
- ✅ Duplicate timestamp handling (updates existing)
- ✅ DataFrame conversion for feature engineering

## State Persistence

SQLite database with ACID guarantees:

```python
from execution.persistence.state_manager import StateManager, StateSnapshot

# Initialize
manager = StateManager("execution/state/paper_trading.db")

# Save snapshot
snapshot = StateSnapshot(
    total_equity=100000.0,
    available_capital=100000.0,
    allocated_capital=0.0,
    total_unrealized_pnl=0.0,
    total_realized_pnl=0.0,
    num_open_positions=0,
    num_strategies_active=0,
    timestamp=int(datetime.now().timestamp() * 1000)
)
manager.save_snapshot(snapshot)

# Transaction support
with manager.transaction():
    manager.save_position(position)
    manager.save_trade(trade)
    # Automatically commits on success, rolls back on error
```

### Database Schema
- `strategies` - Registered strategies
- `positions` - Active and closed positions
- `trades` - Individual trade executions
- `signals` - Strategy signals (with execution status)
- `state_snapshots` - System state for monitoring
- `performance_metrics` - Performance tracking
- `event_log` - Audit trail

## Kirby Message Format

### Candle Update
```json
{
  "type": "candle",
  "starlisting_id": 1,
  "exchange": "hyperliquid",
  "coin": "BTC",
  "quote": "USD",
  "trading_pair": "BTC/USD",
  "market_type": "perps",
  "interval": "1m",
  "data": {
    "time": "2025-11-19T17:04:00+00:00",
    "open": "89840.000000000000000000",
    "high": "89853.000000000000000000",
    "low": "89670.000000000000000000",
    "close": "89695.000000000000000000",
    "volume": "69.587600000000000000",
    "num_trades": 480
  }
}
```

### Subscribe Request
```json
{
  "action": "subscribe",
  "starlisting_ids": [1, 2],
  "historical_candles": 200
}
```

### Success Response
```json
{
  "type": "success",
  "message": "Subscribed to 1 starlisting(s)",
  "starlisting_ids": [1]
}
```

## Known Issues

1. **Funding Rate Format** - Kirby sends additional fields not in our model
2. **Open Interest Format** - Similar to funding rate
3. **Ping Messages** - Not yet handled (unknown message type warning)

These are non-breaking - candles work perfectly, which is the critical path.

## Testing

### Integration Test Results
```
✓ WebSocket URL: ws://localhost:8000/ws
✓ API Key: kb_abc9fca...
✓ Connected successfully
✓ Subscribed to starlisting 1
✓ Candles received: 95
✓ Candles buffered: 2
✓ DataFrame conversion: 2 rows
✓ State snapshot saved
✅ All tests passed!
```

### Test Coverage
- ✅ WebSocket connection and authentication
- ✅ Subscribe/unsubscribe
- ✅ Candle reception and parsing
- ✅ Buffer operations
- ✅ Database persistence
- ✅ End-to-end data flow

## Performance

- **Candle Reception**: 95 candles/minute (1-2 per second)
- **Buffer Operations**: O(1) append/pop
- **Database**: WAL mode for concurrent access
- **Memory**: Ring buffer prevents unbounded growth

## Next Steps

1. **Feature Streaming** - Connect buffer to feature engine
2. **Model Predictor** - Load trained models, run inference
3. **Signal Generator** - Convert predictions to signals
4. **Order Simulator** - Simulate trades with fees/slippage
5. **Position Tracker** - Track P&L and positions
6. **Orchestration** - Coordinate all components

---

**Status**: Phase 8 Complete (WebSocket + Buffer + Persistence) ✅
**Next**: Phase 9 (Live Inference Pipeline) 🔄
