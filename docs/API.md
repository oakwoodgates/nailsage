# Nailsage Trading API Documentation

## Overview

The Nailsage Trading API provides REST endpoints for managing and monitoring trading strategies, positions, trades, and portfolio metrics.

**Base URL:** `http://localhost:8000`

**API Version:** v1 (all endpoints prefixed with `/api/v1/`)

## Authentication

Currently, the API is open (no authentication required). A future version will support API key authentication via the `X-API-Key` header.

## Endpoints

### Health Checks

#### GET /health

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

#### GET /ready

Readiness check that verifies database and required tables.

**Response:**
```json
{
  "ready": true,
  "checks": {
    "database": true,
    "tables": true
  },
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

---

### Strategies

#### GET /api/v1/strategies

List all strategies.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| active_only | boolean | true | Filter to active strategies only |

**Response:**
```json
{
  "strategies": [
    {
      "id": 1,
      "strategy_name": "btc_scalper",
      "version": "v1",
      "starlisting_id": 123,
      "arena_id": 1,
      "interval": "15m",
      "model_id": "model_abc123",
      "is_active": true,
      "initial_bankroll": 10000.00,
      "current_bankroll": 9500.00,
      "created_at": 1700000000000,
      "updated_at": 1700000000000,
      "arena": {
        "id": 1,
        "starlisting_id": 123,
        "trading_pair": "BTC/USD",
        "interval": "15m",
        "coin": "BTC",
        "coin_name": "Bitcoin",
        "quote": "USD",
        "quote_name": "US Dollar",
        "exchange": "hyperliquid",
        "exchange_name": "Hyperliquid",
        "market_type": "perps",
        "market_name": "Perpetuals"
      }
    }
  ],
  "total": 1
}
```

#### GET /api/v1/strategies/by-arena/{arena_id}

Get strategies for a specific arena.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| active_only | boolean | true | Filter to active strategies only |

**Response:** Same structure as `/api/v1/strategies` - list of strategies filtered by arena.

#### GET /api/v1/strategies/{id}

Get strategy by ID.

**Response:**
```json
{
  "id": 1,
  "strategy_name": "btc_scalper",
  "version": "v1",
  "starlisting_id": 123,
  "arena_id": 1,
  "interval": "15m",
  "model_id": "model_abc123",
  "is_active": true,
  "initial_bankroll": 10000.00,
  "current_bankroll": 9500.00,
  "created_at": 1700000000000,
  "updated_at": 1700000000000,
  "arena": {
    "id": 1,
    "starlisting_id": 123,
    "trading_pair": "BTC/USD",
    "interval": "15m",
    "coin": "BTC",
    "coin_name": "Bitcoin",
    "quote": "USD",
    "quote_name": "US Dollar",
    "exchange": "hyperliquid",
    "exchange_name": "Hyperliquid",
    "market_type": "perps",
    "market_name": "Perpetuals"
  }
}
```

#### GET /api/v1/strategies/{id}/stats

Get strategy with performance statistics.

**Response:**
```json
{
  "id": 1,
  "strategy_name": "btc_scalper",
  "version": "v1",
  "starlisting_id": 123,
  "interval": "15m",
  "model_id": "model_abc123",
  "is_active": true,
  "initial_bankroll": 10000.00,
  "current_bankroll": 9500.00,
  "created_at": 1700000000000,
  "updated_at": 1700000000000,
  "total_trades": 150,
  "open_positions": 2,
  "total_pnl_usd": 5000.00,
  "total_pnl_pct": 5.0,
  "realized_pnl_usd": 4500.00,
  "unrealized_pnl_usd": 500.00,
  "win_count": 90,
  "loss_count": 60,
  "win_rate": 60.0,
  "avg_win_usd": 100.00,
  "avg_loss_usd": -50.00,
  "profit_factor": 2.0
}
```

#### GET /api/v1/strategies/{id}/trades

Get trades for a specific strategy.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 100 | Number of trades (1-1000) |
| offset | integer | 0 | Pagination offset |

#### GET /api/v1/strategies/{id}/positions

Get positions for a specific strategy.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| status | string | null | Filter by status (open/closed) |
| limit | integer | 100 | Number of positions (1-1000) |
| offset | integer | 0 | Pagination offset |

#### GET /api/v1/strategies/{id}/bankroll

Get current bankroll for a strategy.

**Response:**
```json
{
  "strategy_id": 1,
  "strategy_name": "btc_scalper",
  "initial_bankroll": 10000.00,
  "current_bankroll": 9500.00,
  "pnl": -500.00,
  "pnl_pct": -5.0,
  "is_active": true
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `strategy_id` | integer | Strategy ID |
| `strategy_name` | string | Strategy name |
| `initial_bankroll` | float | Starting capital (USDT) |
| `current_bankroll` | float | Current capital after P&L (USDT) |
| `pnl` | float | Total P&L (current - initial) |
| `pnl_pct` | float | P&L as percentage |
| `is_active` | boolean | Whether strategy can trade (bankroll > 0) |

#### PATCH /api/v1/strategies/{id}/bankroll

Update/replenish strategy bankroll. Use this endpoint to manually adjust a strategy's bankroll, such as replenishing a depleted strategy.

**Request Body:**
```json
{
  "bankroll": 15000.00
}
```

**Request Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bankroll` | float | Yes | New bankroll value (USDT), must be > 0 |

**Response:** Same as GET /api/v1/strategies/{id}/bankroll

**Example: Replenish a depleted strategy**
```bash
curl -X PATCH http://localhost:8000/api/v1/strategies/1/bankroll \
  -H "Content-Type: application/json" \
  -d '{"bankroll": 10000.00}'
```

---

### Arenas

Trading arenas represent unique combinations of exchange, trading pair, and interval. Arena data is synced from the Kirby API and cached locally.

#### GET /api/v1/arenas

List all arenas.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| active_only | boolean | true | Filter to active arenas only |
| exchange_id | integer | null | Filter by exchange ID |
| coin_id | integer | null | Filter by coin (base asset) ID |

**Response:**
```json
{
  "arenas": [
    {
      "id": 1,
      "starlisting_id": 123,
      "trading_pair": "BTC/USD",
      "trading_pair_id": 1,
      "coin": {
        "id": 1,
        "symbol": "BTC",
        "name": "Bitcoin"
      },
      "quote": {
        "id": 2,
        "symbol": "USD",
        "name": "US Dollar"
      },
      "exchange": {
        "id": 1,
        "slug": "hyperliquid",
        "name": "Hyperliquid"
      },
      "market_type": {
        "id": 1,
        "type": "perps",
        "name": "Perpetuals"
      },
      "interval": "15m",
      "interval_seconds": 900,
      "is_active": true,
      "last_synced_at": 1700000000000,
      "created_at": 1700000000000,
      "updated_at": 1700000000000,
      "strategies": [1, 2]
    }
  ],
  "total": 1
}
```

**Note:** The `strategies` field contains an array of active strategy IDs that are using this arena.

#### GET /api/v1/arenas/{id}

Get arena by ID.

**Response:** Same structure as arena object above.

#### GET /api/v1/arenas/by-starlisting/{starlisting_id}

Get arena by Kirby starlisting ID.

**Response:** Same structure as arena object above.

#### GET /api/v1/arenas/popular

List arenas ranked by number of strategies using them.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 10 | Max arenas to return (1-100) |

**Response:**
```json
{
  "arenas": [
    {
      "id": 1,
      "starlisting_id": 123,
      "trading_pair": "BTC/USD",
      "trading_pair_id": 1,
      "coin": {
        "id": 1,
        "symbol": "BTC",
        "name": "Bitcoin"
      },
      "quote": {
        "id": 2,
        "symbol": "USD",
        "name": "US Dollar"
      },
      "exchange": {
        "id": 1,
        "slug": "hyperliquid",
        "name": "Hyperliquid"
      },
      "market_type": {
        "id": 1,
        "type": "perps",
        "name": "Perpetuals"
      },
      "interval": "15m",
      "interval_seconds": 900,
      "is_active": true,
      "last_synced_at": 1700000000000,
      "created_at": 1700000000000,
      "updated_at": 1700000000000,
      "strategies": [1, 2, 3, 4, 5],
      "strategy_count": 5
    }
  ],
  "total": 1
}
```

**Note:** In addition to `strategy_count`, this endpoint also returns `strategies` containing the actual IDs of active strategies using this arena.

#### POST /api/v1/arenas/sync

Sync arena metadata from Kirby API. Creates or updates local arena record.

**Request Body:**
```json
{
  "starlisting_id": 123
}
```

**Response:**
```json
{
  "arena": { ... },
  "created": true
}
```

---

### Lookup Tables

#### GET /api/v1/exchanges

List all exchanges.

**Response:**
```json
{
  "exchanges": [
    {
      "id": 1,
      "slug": "hyperliquid",
      "name": "Hyperliquid"
    }
  ],
  "total": 1
}
```

#### GET /api/v1/coins

List all coins/assets.

**Response:**
```json
{
  "coins": [
    {
      "id": 1,
      "symbol": "BTC",
      "name": "Bitcoin"
    },
    {
      "id": 2,
      "symbol": "USD",
      "name": "US Dollar"
    }
  ],
  "total": 2
}
```

#### GET /api/v1/market-types

List all market types.

**Response:**
```json
{
  "market_types": [
    {
      "id": 1,
      "type": "perps",
      "name": "Perpetuals"
    }
  ],
  "total": 1
}
```

---

### Trades

#### GET /api/v1/trades

List trades with optional filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| strategy_id | integer | null | Filter by strategy |
| limit | integer | 100 | Number of trades (1-1000) |
| offset | integer | 0 | Pagination offset |

**Response:**
```json
{
  "trades": [
    {
      "id": 1,
      "position_id": 1,
      "strategy_id": 1,
      "starlisting_id": 123,
      "trade_type": "open_long",
      "size": 1000.00,
      "price": 50000.00,
      "fees": 1.00,
      "slippage": 0.50,
      "timestamp": 1700000000000,
      "signal_id": 1,
      "created_at": 1700000000000,
      "strategy_name": "btc_scalper",
      "position_side": "long",
      "arena_id": 1
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

#### GET /api/v1/trades/recent

Get most recent trades.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 10 | Number of trades (1-100) |

#### GET /api/v1/trades/{id}

Get trade by ID.

---

### Positions

#### GET /api/v1/positions

List positions with optional filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| strategy_id | integer | null | Filter by strategy |
| status | string | null | Filter by status (open/closed) |
| limit | integer | 100 | Number of positions (1-1000) |
| offset | integer | 0 | Pagination offset |

**Response:**
```json
{
  "positions": [
    {
      "id": 1,
      "strategy_id": 1,
      "starlisting_id": 123,
      "side": "long",
      "size": 1000.00,
      "entry_price": 50000.00,
      "entry_timestamp": 1700000000000,
      "exit_price": null,
      "exit_timestamp": null,
      "exit_reason": null,
      "realized_pnl": null,
      "unrealized_pnl": 50.00,
      "fees_paid": 1.00,
      "status": "open",
      "stop_loss_price": 49000.00,
      "take_profit_price": 52000.00,
      "created_at": 1700000000000,
      "updated_at": 1700000000000,
      "strategy_name": "btc_scalper",
      "duration_minutes": null,
      "pnl_pct": null
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

#### GET /api/v1/positions/open

Get all open positions.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| strategy_id | integer | null | Filter by strategy |

#### GET /api/v1/positions/closed

Get closed positions (historical).

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| strategy_id | integer | null | Filter by strategy |
| limit | integer | 100 | Number of positions (1-1000) |
| offset | integer | 0 | Pagination offset |

#### GET /api/v1/positions/{id}

Get position by ID.

---

### Portfolio

#### GET /api/v1/portfolio/summary

Get complete portfolio overview.

**Response:**
```json
{
  "initial_capital_usd": 100000.00,
  "total_equity_usd": 105000.00,
  "available_capital_usd": 95000.00,
  "allocated_capital_usd": 10000.00,
  "total_pnl_usd": 5000.00,
  "total_pnl_pct": 5.0,
  "realized_pnl_usd": 4500.00,
  "unrealized_pnl_usd": 500.00,
  "total_open_positions": 3,
  "max_positions": 10,
  "position_utilization_pct": 30.0,
  "total_exposure_usd": 10000.00,
  "long_exposure_usd": 7000.00,
  "short_exposure_usd": 3000.00,
  "net_exposure_usd": 4000.00,
  "active_strategies": 2,
  "total_strategies": 3,
  "trades_today": 5,
  "total_fees_paid": 50.00,
  "timestamp": 1700000000000
}
```

#### GET /api/v1/portfolio/allocation

Get capital allocation by strategy.

**Response:**
```json
{
  "allocations": [
    {
      "strategy_id": 1,
      "strategy_name": "btc_scalper",
      "allocated_usd": 5000.00,
      "allocation_pct": 50.0,
      "open_positions": 2,
      "unrealized_pnl_usd": 300.00
    }
  ],
  "total_allocated_usd": 10000.00,
  "unallocated_usd": 90000.00
}
```

#### GET /api/v1/portfolio/exposure

Get current market exposure.

**Response:**
```json
{
  "total_exposure_usd": 10000.00,
  "long_exposure_usd": 7000.00,
  "short_exposure_usd": 3000.00,
  "net_exposure_usd": 4000.00,
  "gross_exposure_usd": 10000.00,
  "exposure_pct_of_equity": 9.5,
  "by_starlisting": [
    {
      "starlisting_id": 123,
      "side": "long",
      "exposure_usd": 5000.00
    }
  ]
}
```

#### GET /api/v1/portfolio/history

Get equity curve data.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 1000 | Max data points (1-10000) |

**Response:**
```json
{
  "points": [
    {
      "timestamp": 1700000000000,
      "equity_usd": 100500.00,
      "realized_pnl_usd": 500.00,
      "unrealized_pnl_usd": 0.00,
      "open_positions": 1
    }
  ],
  "initial_capital_usd": 100000.00,
  "current_equity_usd": 105000.00,
  "max_equity_usd": 106000.00,
  "min_equity_usd": 99000.00,
  "max_drawdown_usd": 2000.00,
  "max_drawdown_pct": 1.9
}
```

---

### Statistics

#### GET /api/v1/stats/summary

Get comprehensive financial summary.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| strategy_id | integer | null | Filter by strategy |

**Response:**
```json
{
  "total_equity_usd": 105000.00,
  "total_pnl_usd": 5000.00,
  "total_pnl_pct": 5.0,
  "realized_pnl_usd": 4500.00,
  "unrealized_pnl_usd": 500.00,
  "total_trades": 150,
  "total_wins": 90,
  "total_losses": 60,
  "win_rate": 60.0,
  "avg_win_usd": 100.00,
  "avg_loss_usd": -50.00,
  "largest_win_usd": 500.00,
  "largest_loss_usd": -200.00,
  "profit_factor": 2.0,
  "avg_trade_usd": 30.00,
  "expectancy_usd": 30.00,
  "max_drawdown_usd": 2000.00,
  "max_drawdown_pct": 1.9,
  "total_fees_paid": 150.00,
  "timestamp": 1700000000000
}
```

#### GET /api/v1/stats/daily

Get daily P&L breakdown.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| days | integer | 30 | Number of days (1-365) |
| strategy_id | integer | null | Filter by strategy |

**Response:**
```json
{
  "daily": [
    {
      "date": "2025-01-15",
      "timestamp_start": 1700000000000,
      "timestamp_end": 1700086399999,
      "pnl_usd": 500.00,
      "pnl_pct": 0.5,
      "cumulative_pnl_usd": 5000.00,
      "trades": 10,
      "wins": 6,
      "losses": 4,
      "starting_equity_usd": 104500.00,
      "ending_equity_usd": 105000.00
    }
  ],
  "total_days": 30,
  "profitable_days": 20,
  "losing_days": 10,
  "avg_daily_pnl_usd": 166.67,
  "best_day_usd": 1000.00,
  "worst_day_usd": -300.00
}
```

#### GET /api/v1/stats/by-strategy

Get statistics grouped by strategy.

**Response:**
```json
[
  {
    "strategy_id": 1,
    "strategy_name": "btc_scalper",
    "is_active": true,
    "total_pnl_usd": 3000.00,
    "realized_pnl_usd": 2800.00,
    "unrealized_pnl_usd": 200.00,
    "total_trades": 100,
    "open_positions": 2,
    "win_count": 60,
    "loss_count": 40,
    "win_rate": 60.0,
    "avg_win_usd": 80.00,
    "avg_loss_usd": -40.00,
    "profit_factor": 2.0
  }
]
```

#### GET /api/v1/stats/leaderboard

Get strategy leaderboard.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| metric | string | total_pnl | Ranking metric (total_pnl, win_rate, profit_factor) |
| limit | integer | 10 | Number of entries (1-100) |

**Response:**
```json
{
  "entries": [
    {
      "rank": 1,
      "strategy_id": 1,
      "strategy_name": "btc_scalper",
      "interval": "15m",
      "total_pnl_usd": 3000.00,
      "total_pnl_pct": 3.0,
      "win_rate": 60.0,
      "total_trades": 100,
      "profit_factor": 2.0,
      "avg_trade_usd": 30.00,
      "max_drawdown_pct": null
    }
  ],
  "metric": "total_pnl",
  "total_strategies": 3
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "detail": null
}
```

**Common HTTP Status Codes:**
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found
- 500: Internal Server Error

---

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
