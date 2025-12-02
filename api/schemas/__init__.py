"""Pydantic schemas for API request/response models."""

from api.schemas.common import (
    PaginationParams,
    PaginatedResponse,
    ErrorResponse,
    SuccessResponse,
    TimestampMixin,
)
from api.schemas.strategies import (
    StrategyBase,
    StrategyResponse,
    StrategyWithStats,
    StrategyListResponse,
)
from api.schemas.trades import (
    TradeBase,
    TradeResponse,
    TradeListResponse,
)
from api.schemas.positions import (
    PositionBase,
    PositionResponse,
    PositionListResponse,
)
from api.schemas.portfolio import (
    PortfolioSummary,
    AllocationItem,
    AllocationResponse,
    ExposureSummary,
    EquityPoint,
    EquityHistoryResponse,
)
from api.schemas.stats import (
    FinancialSummary,
    DailyPnL,
    DailyPnLResponse,
    StrategyStats,
    LeaderboardEntry,
    LeaderboardResponse,
)
from api.schemas.websocket import (
    WebSocketMessage,
    SubscribeRequest,
    UnsubscribeRequest,
    SubscribeResponse,
    TradeEvent,
    PositionEvent,
    PortfolioEvent,
    PriceEvent,
    SignalEvent,
)

__all__ = [
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "ErrorResponse",
    "SuccessResponse",
    "TimestampMixin",
    # Strategies
    "StrategyBase",
    "StrategyResponse",
    "StrategyWithStats",
    "StrategyListResponse",
    # Trades
    "TradeBase",
    "TradeResponse",
    "TradeListResponse",
    # Positions
    "PositionBase",
    "PositionResponse",
    "PositionListResponse",
    # Portfolio
    "PortfolioSummary",
    "AllocationItem",
    "AllocationResponse",
    "ExposureSummary",
    "EquityPoint",
    "EquityHistoryResponse",
    # Stats
    "FinancialSummary",
    "DailyPnL",
    "DailyPnLResponse",
    "StrategyStats",
    "LeaderboardEntry",
    "LeaderboardResponse",
    # WebSocket
    "WebSocketMessage",
    "SubscribeRequest",
    "UnsubscribeRequest",
    "SubscribeResponse",
    "TradeEvent",
    "PositionEvent",
    "PortfolioEvent",
    "PriceEvent",
    "SignalEvent",
]
