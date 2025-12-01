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
| `prices` | Market data | `price.candle` |
| `signals` | Trading signals | `signal.generated`, `signal.executed` |

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

#### price.candle

Emitted for each candle update (forwarded from Kirby).

```json
{
  "type": "price.candle",
  "channel": "prices",
  "timestamp": 1700000000000,
  "data": {
    "starlisting_id": 789,
    "coin": "BTC",
    "interval": "1m",
    "open": 50000.00,
    "high": 50100.00,
    "low": 49950.00,
    "close": 50050.00,
    "volume": 123.45
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

  subscribe(channels) {
    this.send({
      action: 'subscribe',
      channels: channels
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

client.onChannel('prices', (message) => {
  console.log('Price update:', message.data);
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
