# Nailsage WebSocket Protocol Documentation

## Overview

The Nailsage WebSocket endpoint provides real-time updates for trading activity, positions, portfolio metrics, and price feeds.

**WebSocket URL:** `ws://localhost:8000/ws`

## Connection

### Connecting

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected to Nailsage');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from Nailsage');
};
```

### Welcome Message

Upon connection, you'll receive a welcome message:

```json
{
  "type": "connected",
  "connection_id": "uuid-string",
  "message": "Connected to Nailsage API",
  "available_channels": ["trades", "positions", "portfolio", "prices", "signals"]
}
```

---

## Channels

Subscribe to channels to receive specific types of updates:

| Channel | Description | Event Types |
|---------|-------------|-------------|
| `trades` | Trade executions | `trade.new` |
| `positions` | Position lifecycle | `position.opened`, `position.closed`, `position.pnl_update` |
| `portfolio` | Portfolio metrics | `portfolio.update` |
| `prices` | Market data (via Kirby) | `price.candle`, `price.historical`, `price.funding`, `price.open_interest`, `price.historical_funding`, `price.historical_oi` |
| `signals` | Trading signals | `signal.generated` |

---

## Client Messages

### Subscribe

Subscribe to one or more channels:

```json
{
  "action": "subscribe",
  "channels": ["trades", "positions"]
}
```

**Response:**
```json
{
  "action": "subscribe",
  "channels": ["trades", "positions"],
  "status": "subscribed",
  "message": "Subscribed to 2 channel(s)"
}
```

### Subscribe with Strategy Filter

Filter events to a specific strategy:

```json
{
  "action": "subscribe",
  "channels": ["trades", "positions", "signals"],
  "filters": {
    "strategy_id": 1
  }
}
```

This will only deliver events where `data.strategy_id` matches the filter value.

### Subscribe to Prices (with Historical Data)

Subscribe to real-time price data for specific starlistings:

```json
{
  "action": "subscribe",
  "channels": ["prices"],
  "starlisting_ids": [1, 2],
  "history": 500
}
```

**Parameters:**
- `starlisting_ids`: Array of Kirby starlisting IDs to subscribe to
- `history`: Number of historical candles to receive (0-1000, default: 500)

Upon subscription, you'll receive:
1. `price.historical` - Historical candle data
2. `price.historical_funding` - Historical funding rates (perps only)
3. `price.historical_oi` - Historical open interest (perps only)
4. Then real-time `price.candle`, `price.funding`, `price.open_interest` updates

### Unsubscribe

Unsubscribe from channels:

```json
{
  "action": "unsubscribe",
  "channels": ["prices"]
}
```

**Response:**
```json
{
  "action": "unsubscribe",
  "channels": ["prices"],
  "status": "unsubscribed",
  "message": "Unsubscribed from 1 channel(s)"
}
```

### Ping

Keep-alive ping:

```json
{
  "action": "ping"
}
```

**Response:**
```json
{
  "type": "pong"
}
```

---

## Server Messages

All server messages follow this structure:

```json
{
  "type": "event.type",
  "channel": "channel_name",
  "timestamp": 1700000000000,
  "data": { ... }
}
```

### Trade Events

#### trade.new

Emitted when a trade is executed.

```json
{
  "type": "trade.new",
  "channel": "trades",
  "timestamp": 1700000000000,
  "data": {
    "trade_id": 123,
    "position_id": 45,
    "strategy_id": 1,
    "strategy_name": "btc_scalper",
    "starlisting_id": 789,
    "trade_type": "open_long",
    "side": "long",
    "size": 1000.00,
    "price": 50000.00,
    "fees": 1.00
  }
}
```

### Position Events

#### position.opened

Emitted when a new position is opened.

```json
{
  "type": "position.opened",
  "channel": "positions",
  "timestamp": 1700000000000,
  "data": {
    "position_id": 45,
    "strategy_id": 1,
    "strategy_name": "btc_scalper",
    "starlisting_id": 789,
    "side": "long",
    "size": 1000.00,
    "entry_price": 50000.00,
    "status": "open",
    "unrealized_pnl": null,
    "realized_pnl": null,
    "current_price": null,
    "exit_price": null,
    "exit_reason": null
  }
}
```

#### position.closed

Emitted when a position is closed.

```json
{
  "type": "position.closed",
  "channel": "positions",
  "timestamp": 1700000000000,
  "data": {
    "position_id": 45,
    "strategy_id": 1,
    "strategy_name": "btc_scalper",
    "starlisting_id": 789,
    "side": "long",
    "size": 1000.00,
    "entry_price": 50000.00,
    "status": "closed",
    "unrealized_pnl": null,
    "realized_pnl": 50.00,
    "current_price": 50500.00,
    "exit_price": 50500.00,
    "exit_reason": "signal"
  }
}
```

#### position.pnl_update

Emitted periodically with updated P&L for open positions.

```json
{
  "type": "position.pnl_update",
  "channel": "positions",
  "timestamp": 1700000000000,
  "data": {
    "position_id": 45,
    "strategy_id": 1,
    "strategy_name": "btc_scalper",
    "starlisting_id": 789,
    "side": "long",
    "size": 1000.00,
    "entry_price": 50000.00,
    "status": "open",
    "unrealized_pnl": 25.00,
    "realized_pnl": null,
    "current_price": 50250.00,
    "exit_price": null,
    "exit_reason": null
  }
}
```

### Portfolio Events

#### portfolio.update

Emitted when portfolio metrics change.

```json
{
  "type": "portfolio.update",
  "channel": "portfolio",
  "timestamp": 1700000000000,
  "data": {
    "total_equity_usd": 105000.00,
    "total_pnl_usd": 5000.00,
    "total_pnl_pct": 5.0,
    "realized_pnl_usd": 4500.00,
    "unrealized_pnl_usd": 500.00,
    "open_positions": 3,
    "active_strategies": 2
  }
}
```

### Price Events

Price data is proxied from Kirby's WebSocket API. Subscribe to the `prices` channel with specific `starlisting_ids` to receive updates.

#### price.historical

Sent immediately after subscribing, contains historical candle data.

```json
{
  "type": "price.historical",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 1,
    "coin": "BTC",
    "interval": "1m",
    "count": 500,
    "candles": [
      {
        "time": "2025-12-01T12:00:00+00:00",
        "open": 95000.00,
        "high": 95100.00,
        "low": 94950.00,
        "close": 95050.00,
        "volume": 123.45
      }
    ]
  }
}
```

#### price.candle

Real-time candle update (forwarded from Kirby).

```json
{
  "type": "price.candle",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 1,
    "coin": "BTC",
    "interval": "1m",
    "time": "2025-12-01T12:15:00+00:00",
    "open": 95050.00,
    "high": 95100.00,
    "low": 95000.00,
    "close": 95075.00,
    "volume": 45.67
  }
}
```

#### price.funding

Real-time funding rate update (perpetual markets only).

```json
{
  "type": "price.funding",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 1,
    "funding_rate": 0.0001,
    "time": "2025-12-01T12:00:00+00:00"
  }
}
```

#### price.open_interest

Real-time open interest update (perpetual markets only).

```json
{
  "type": "price.open_interest",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 1,
    "open_interest": 125000.50,
    "time": "2025-12-01T12:00:00+00:00"
  }
}
```

#### price.historical_funding

Historical funding rates (sent after subscribing, perps only).

```json
{
  "type": "price.historical_funding",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 1,
    "coin": "BTC",
    "count": 100,
    "funding_rates": [
      {
        "time": "2025-12-01T08:00:00+00:00",
        "funding_rate": 0.0001,
        "mark_price": 95000.00
      }
    ]
  }
}
```

#### price.historical_oi

Historical open interest (sent after subscribing, perps only).

```json
{
  "type": "price.historical_oi",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 1,
    "coin": "BTC",
    "count": 100,
    "open_interest": [
      {
        "time": "2025-12-01T08:00:00+00:00",
        "open_interest": 125000.50,
        "notional_value": 11875047500.00
      }
    ]
  }
}
```

### Signal Events

#### signal.generated

Emitted when a trading signal is generated.

```json
{
  "type": "signal.generated",
  "channel": "signals",
  "timestamp": 1700000000000,
  "data": {
    "signal_id": 567,
    "strategy_id": 1,
    "strategy_name": "btc_scalper",
    "starlisting_id": 789,
    "signal_type": "long",
    "confidence": 0.85,
    "price_at_signal": 50000.00,
    "was_executed": true,
    "rejection_reason": null
  }
}
```

---

## Example: Full Client Implementation

```javascript
class NailsageWebSocket {
  constructor(url = 'ws://localhost:8000/ws') {
    this.url = url;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.handlers = {};
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected to Nailsage');
      this.reconnectAttempts = 0;
      // Subscribe to desired channels
      this.subscribe(['trades', 'positions', 'portfolio']);
      // Subscribe to price data for specific starlistings
      this.subscribePrices([1, 2], 500);
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.ws.onclose = () => {
      console.log('Disconnected');
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  subscribe(channels, filters = null) {
    const msg = {
      action: 'subscribe',
      channels: channels
    };
    if (filters) {
      msg.filters = filters;
    }
    this.send(msg);
  }

  subscribePrices(starlistingIds, history = 500) {
    this.send({
      action: 'subscribe',
      channels: ['prices'],
      starlisting_ids: starlistingIds,
      history: history
    });
  }

  unsubscribe(channels) {
    this.send({
      action: 'unsubscribe',
      channels: channels
    });
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  handleMessage(message) {
    const { type, channel, data } = message;

    // Call registered handlers
    if (this.handlers[type]) {
      this.handlers[type](data);
    }

    // Generic channel handler
    if (this.handlers[`channel:${channel}`]) {
      this.handlers[`channel:${channel}`](message);
    }
  }

  on(eventType, handler) {
    this.handlers[eventType] = handler;
  }

  onChannel(channel, handler) {
    this.handlers[`channel:${channel}`] = handler;
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(() => this.connect(), delay);
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage
const client = new NailsageWebSocket();

client.on('trade.new', (data) => {
  console.log('New trade:', data);
});

client.on('position.opened', (data) => {
  console.log('Position opened:', data);
});

client.on('portfolio.update', (data) => {
  console.log('Portfolio update:', data);
});

client.on('price.historical', (data) => {
  console.log(`Received ${data.count} historical candles for ${data.coin}`);
  // Initialize chart with historical data
  initChart(data.candles);
});

client.on('price.candle', (data) => {
  console.log('Live candle:', data);
  // Update chart with new candle
  updateChart(data);
});

client.onChannel('prices', (message) => {
  console.log('Price update:', message.type, message.data);
});

client.connect();
```

---

## Connection Limits

- Maximum concurrent connections: 100
- Heartbeat interval: 30 seconds (server may close idle connections)

---

## Error Handling

If an error occurs, you'll receive:

```json
{
  "type": "error",
  "message": "Error description",
  "code": "ERROR_CODE"
}
```

Common error codes:
- `INVALID_JSON`: Malformed JSON message
- `UNKNOWN_ACTION`: Unrecognized action type
- `SUBSCRIPTION_ERROR`: Error subscribing to channel
