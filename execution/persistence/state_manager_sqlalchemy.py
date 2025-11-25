# This file contains the SQLAlchemy-compatible methods to patch state_manager.py
# These will be integrated into the main file

# Save position method (SQLAlchemy version)
SAVE_POSITION = """
def save_position(self, position: Position) -> int:
    now = int(datetime.now().timestamp() * 1000)
    engine = self._get_engine()

    with engine.begin() as conn:
        if position.id is None:
            result = conn.execute(
                text('''
                    INSERT INTO positions
                    (strategy_id, starlisting_id, side, size, entry_price, entry_timestamp,
                     exit_price, exit_timestamp, realized_pnl, unrealized_pnl, fees_paid,
                     status, stop_loss_price, take_profit_price, exit_reason, created_at, updated_at)
                    VALUES (:strategy_id, :starlisting_id, :side, :size, :entry_price, :entry_timestamp,
                            :exit_price, :exit_timestamp, :realized_pnl, :unrealized_pnl, :fees_paid,
                            :status, :stop_loss_price, :take_profit_price, :exit_reason, :created_at, :updated_at)
                '''),
                {
                    "strategy_id": position.strategy_id,
                    "starlisting_id": position.starlisting_id,
                    "side": position.side,
                    "size": position.size,
                    "entry_price": position.entry_price,
                    "entry_timestamp": position.entry_timestamp,
                    "exit_price": position.exit_price,
                    "exit_timestamp": position.exit_timestamp,
                    "realized_pnl": position.realized_pnl,
                    "unrealized_pnl": position.unrealized_pnl,
                    "fees_paid": position.fees_paid,
                    "status": position.status,
                    "stop_loss_price": position.stop_loss_price,
                    "take_profit_price": position.take_profit_price,
                    "exit_reason": position.exit_reason,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            position_id = result.lastrowid
            logger.info(f"Saved new position: {position.side} {position.size:.2f} @ {position.entry_price:.2f} (ID: {position_id})")
        else:
            conn.execute(
                text('''
                    UPDATE positions
                    SET strategy_id = :strategy_id, starlisting_id = :starlisting_id, side = :side, size = :size,
                        entry_price = :entry_price, entry_timestamp = :entry_timestamp, exit_price = :exit_price,
                        exit_timestamp = :exit_timestamp, realized_pnl = :realized_pnl, unrealized_pnl = :unrealized_pnl,
                        fees_paid = :fees_paid, status = :status, stop_loss_price = :stop_loss_price,
                        take_profit_price = :take_profit_price, exit_reason = :exit_reason, updated_at = :updated_at
                    WHERE id = :id
                '''),
                {
                    "strategy_id": position.strategy_id,
                    "starlisting_id": position.starlisting_id,
                    "side": position.side,
                    "size": position.size,
                    "entry_price": position.entry_price,
                    "entry_timestamp": position.entry_timestamp,
                    "exit_price": position.exit_price,
                    "exit_timestamp": position.exit_timestamp,
                    "realized_pnl": position.realized_pnl,
                    "unrealized_pnl": position.unrealized_pnl,
                    "fees_paid": position.fees_paid,
                    "status": position.status,
                    "stop_loss_price": position.stop_loss_price,
                    "take_profit_price": position.take_profit_price,
                    "exit_reason": position.exit_reason,
                    "updated_at": now,
                    "id": position.id,
                },
            )
            position_id = position.id
            logger.debug(f"Updated position ID {position_id}")

    return position_id
"""

# NOTE: Due to the large size of the state_manager.py file,
# the remaining methods need to follow this same pattern:
# 1. Replace self._get_connection() with self._get_engine()
# 2. Use `with engine.connect() as conn:` for reads or `with engine.begin() as conn:` for writes
# 3. Replace cursor.execute(sql, params) with conn.execute(text(sql), {...named params...})
# 4. Replace ? placeholders with :named_params
# 5. Replace cursor.fetchall()/fetchone() with result.fetchall()/fetchone()
# 6. Update _row_to_* methods to handle SQLAlchemy Row objects (use hasattr checks)
