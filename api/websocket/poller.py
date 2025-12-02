"""Database poller for WebSocket event emission.

Bridges the gap between the trading containers and the API container
by polling the database for changes and emitting WebSocket events.
"""

import asyncio
import logging
from typing import Dict, Optional

from api.websocket.events import (
    EventDispatcher,
    EventType,
    emit_trade_event,
    emit_position_event,
    emit_portfolio_event,
    emit_signal_event,
)

logger = logging.getLogger(__name__)


class DatabasePoller:
    """
    Polls database for new trades/positions and emits WebSocket events.

    This bridges the gap between the trading container (nailsage-binance)
    and the API container (nailsage-api) by detecting database changes
    and emitting appropriate WebSocket events.
    """

    def __init__(
        self,
        state_manager,
        event_dispatcher: EventDispatcher,
        interval: float = 3.0,
    ):
        """Initialize the database poller.

        Args:
            state_manager: StateManager instance for database access
            event_dispatcher: EventDispatcher for emitting events
            interval: Polling interval in seconds
        """
        self._state_manager = state_manager
        self._event_dispatcher = event_dispatcher
        self._interval = interval

        # Tracking state
        self._last_trade_id: int = 0
        self._last_signal_id: int = 0
        self._known_positions: Dict[int, dict] = {}  # id -> {status, updated_at}
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info(f"DatabasePoller initialized with {interval}s interval")

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return

        # Initialize with current state
        await self._initialize_state()

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("DatabasePoller started")

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DatabasePoller stopped")

    async def _initialize_state(self) -> None:
        """Initialize tracking state from current database."""
        try:
            # Get current max trade ID
            self._last_trade_id = await asyncio.to_thread(
                self._get_max_trade_id
            )

            # Get current max signal ID
            self._last_signal_id = await asyncio.to_thread(
                self._get_max_signal_id
            )

            # Get current open positions
            positions = await asyncio.to_thread(
                self._state_manager.get_open_positions
            )
            for pos in positions:
                self._known_positions[pos.id] = {
                    "status": pos.status,
                    "updated_at": getattr(pos, "updated_at", 0),
                    "unrealized_pnl": getattr(pos, "unrealized_pnl", None),
                }

            logger.info(
                f"Initialized: last_trade_id={self._last_trade_id}, "
                f"last_signal_id={self._last_signal_id}, "
                f"positions={len(self._known_positions)}"
            )
        except Exception as e:
            logger.error(f"Error initializing poller state: {e}")

    def _get_max_trade_id(self) -> int:
        """Get the maximum trade ID from database."""
        try:
            engine = self._state_manager._get_engine()
            from sqlalchemy import text

            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COALESCE(MAX(id), 0) FROM trades")
                )
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error getting max trade ID: {e}")
            return 0

    def _get_max_signal_id(self) -> int:
        """Get the maximum signal ID from database."""
        try:
            engine = self._state_manager._get_engine()
            from sqlalchemy import text

            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COALESCE(MAX(id), 0) FROM signals")
                )
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error getting max signal ID: {e}")
            return 0

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_new_trades()
                await self._check_new_signals()
                await self._check_position_changes()
                await self._emit_portfolio_update()
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")

            await asyncio.sleep(self._interval)

    async def _check_new_trades(self) -> None:
        """Check for new trades and emit events."""
        try:
            engine = self._state_manager._get_engine()
            from sqlalchemy import text

            query = """
                SELECT t.*, s.strategy_name, p.side as position_side
                FROM trades t
                LEFT JOIN strategies s ON t.strategy_id = s.id
                LEFT JOIN positions p ON t.position_id = p.id
                WHERE t.id > :last_id
                ORDER BY t.id ASC
            """

            new_trades = await asyncio.to_thread(
                self._fetch_new_trades, engine, query, self._last_trade_id
            )

            for trade in new_trades:
                await emit_trade_event(
                    trade_id=trade["id"],
                    position_id=trade["position_id"],
                    strategy_id=trade["strategy_id"],
                    strategy_name=trade.get("strategy_name", "Unknown"),
                    starlisting_id=trade["starlisting_id"],
                    trade_type=trade["trade_type"],
                    side=trade.get("position_side", "unknown"),
                    size=trade["size"],
                    price=trade["price"],
                    fees=trade["fees"],
                )
                self._last_trade_id = trade["id"]
                logger.debug(f"Emitted trade event for trade #{trade['id']}")

        except Exception as e:
            logger.error(f"Error checking new trades: {e}")

    def _fetch_new_trades(self, engine, query: str, last_id: int) -> list:
        """Fetch new trades from database (sync)."""
        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text(query), {"last_id": last_id})
            return [dict(row._mapping) for row in result]

    async def _check_new_signals(self) -> None:
        """Check for new signals and emit events."""
        try:
            engine = self._state_manager._get_engine()
            from sqlalchemy import text

            query = """
                SELECT s.*, str.strategy_name
                FROM signals s
                LEFT JOIN strategies str ON s.strategy_id = str.id
                WHERE s.id > :last_id
                ORDER BY s.id ASC
            """

            new_signals = await asyncio.to_thread(
                self._fetch_new_signals, engine, query, self._last_signal_id
            )

            for signal in new_signals:
                await emit_signal_event(
                    signal_id=signal["id"],
                    strategy_id=signal["strategy_id"],
                    strategy_name=signal.get("strategy_name", "Unknown"),
                    starlisting_id=signal["starlisting_id"],
                    signal_type=signal["signal_type"],
                    confidence=signal.get("confidence"),
                    price_at_signal=signal["price_at_signal"],
                    was_executed=signal["was_executed"],
                    rejection_reason=signal.get("rejection_reason"),
                )
                self._last_signal_id = signal["id"]
                logger.debug(f"Emitted signal event for signal #{signal['id']}")

        except Exception as e:
            logger.error(f"Error checking new signals: {e}")

    def _fetch_new_signals(self, engine, query: str, last_id: int) -> list:
        """Fetch new signals from database (sync)."""
        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text(query), {"last_id": last_id})
            return [dict(row._mapping) for row in result]

    async def _check_position_changes(self) -> None:
        """Check for position changes and emit events."""
        try:
            # Get open positions
            open_positions = await asyncio.to_thread(
                self._state_manager.get_open_positions
            )

            # Get strategy names for positions
            strategies = await asyncio.to_thread(
                self._state_manager.get_active_strategies
            )
            strategy_names = {s.id: s.strategy_name for s in strategies}

            # Track which positions we've seen this cycle
            current_open_ids = set()

            for pos in open_positions:
                pos_id = pos.id
                current_open_ids.add(pos_id)
                strategy_name = strategy_names.get(pos.strategy_id, "Unknown")

                if pos_id not in self._known_positions:
                    # New position opened
                    await emit_position_event(
                        event_type=EventType.POSITION_OPENED,
                        position_id=pos_id,
                        strategy_id=pos.strategy_id,
                        strategy_name=strategy_name,
                        starlisting_id=pos.starlisting_id,
                        side=pos.side,
                        size=pos.size,
                        entry_price=pos.entry_price,
                        status=pos.status,
                    )
                    self._known_positions[pos_id] = {
                        "status": "open",
                        "strategy_id": pos.strategy_id,
                        "strategy_name": strategy_name,
                        "starlisting_id": pos.starlisting_id,
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "unrealized_pnl": getattr(pos, "unrealized_pnl", None),
                    }
                    logger.debug(f"Emitted position.opened for #{pos_id}")

                else:
                    # Check for P&L updates on open positions
                    known = self._known_positions[pos_id]
                    current_pnl = getattr(pos, "unrealized_pnl", None)
                    if current_pnl != known.get("unrealized_pnl"):
                        await emit_position_event(
                            event_type=EventType.POSITION_PNL_UPDATE,
                            position_id=pos_id,
                            strategy_id=pos.strategy_id,
                            strategy_name=strategy_name,
                            starlisting_id=pos.starlisting_id,
                            side=pos.side,
                            size=pos.size,
                            entry_price=pos.entry_price,
                            status="open",
                            unrealized_pnl=current_pnl,
                        )
                        self._known_positions[pos_id]["unrealized_pnl"] = current_pnl

            # Check for closed positions (were open, now not in open list)
            for pos_id, known in list(self._known_positions.items()):
                if known["status"] == "open" and pos_id not in current_open_ids:
                    # Position was closed - fetch details
                    closed_pos = await asyncio.to_thread(
                        self._state_manager.get_position_by_id, pos_id
                    )
                    if closed_pos and closed_pos.status == "closed":
                        await emit_position_event(
                            event_type=EventType.POSITION_CLOSED,
                            position_id=pos_id,
                            strategy_id=closed_pos.strategy_id,
                            strategy_name=known.get("strategy_name", "Unknown"),
                            starlisting_id=closed_pos.starlisting_id,
                            side=closed_pos.side,
                            size=closed_pos.size,
                            entry_price=closed_pos.entry_price,
                            status="closed",
                            realized_pnl=getattr(closed_pos, "realized_pnl", None),
                            exit_price=getattr(closed_pos, "exit_price", None),
                            exit_reason=getattr(closed_pos, "exit_reason", None),
                        )
                        self._known_positions[pos_id]["status"] = "closed"
                        logger.debug(f"Emitted position.closed for #{pos_id}")

        except Exception as e:
            logger.error(f"Error checking position changes: {e}")

    async def _emit_portfolio_update(self) -> None:
        """Emit portfolio update event."""
        try:
            from api.services.portfolio_service import PortfolioService

            service = PortfolioService(self._state_manager)
            summary = await asyncio.to_thread(service.get_summary)

            await emit_portfolio_event(
                total_equity_usd=summary.total_equity_usd,
                total_pnl_usd=summary.total_pnl_usd,
                total_pnl_pct=summary.total_pnl_pct,
                realized_pnl_usd=summary.realized_pnl_usd,
                unrealized_pnl_usd=summary.unrealized_pnl_usd,
                open_positions=summary.total_open_positions,
                active_strategies=summary.active_strategies,
            )
        except Exception as e:
            logger.error(f"Error emitting portfolio update: {e}")
