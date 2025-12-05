"""Quick script to check paper trading statistics.

This script works with both SQLite (local development) and PostgreSQL (Docker).
It uses StateManager for database-agnostic access.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execution.persistence.state_manager import StateManager
from sqlalchemy import text


def format_timestamp(unix_ms: int) -> str:
    """Convert Unix milliseconds to readable datetime string."""
    return datetime.fromtimestamp(unix_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Display paper trading statistics."""
    try:
        # Initialize StateManager (auto-detects SQLite or PostgreSQL from DATABASE_URL)
        # If DATABASE_URL points to Docker's PostgreSQL (not accessible locally), fall back to SQLite
        database_url = os.getenv('DATABASE_URL')

        if database_url and 'postgres:5432' in database_url:
            # Docker PostgreSQL URL - use local SQLite instead
            print("Note: DATABASE_URL points to Docker PostgreSQL, using local SQLite instead\n")
            db_path = Path("execution/state/paper_trading.db")
            state_manager = StateManager(database_url=f"sqlite:///{db_path}")
        elif not database_url:
            # No DATABASE_URL - use local SQLite
            db_path = Path("execution/state/paper_trading.db")
            state_manager = StateManager(database_url=f"sqlite:///{db_path}")
        else:
            # Use provided DATABASE_URL (local PostgreSQL or other)
            state_manager = StateManager(database_url=database_url)

        engine = state_manager._get_engine()

        print(f"Connected to database: {state_manager._get_safe_url()}")
        print("=" * 70)

        with engine.connect() as conn:
            # Check signals
            result = conn.execute(text("SELECT COUNT(*) FROM signals"))
            signal_count = result.scalar()
            print(f"Total signals generated: {signal_count}")

            # Check positions
            result = conn.execute(text("SELECT COUNT(*) FROM positions WHERE status = 'open'"))
            open_positions = result.scalar()
            print(f"Open positions: {open_positions}")

            result = conn.execute(text("SELECT COUNT(*) FROM positions WHERE status = 'closed'"))
            closed_positions = result.scalar()
            print(f"Closed positions: {closed_positions}")

            # Check trades
            result = conn.execute(text("SELECT COUNT(*) FROM trades"))
            trade_count = result.scalar()
            print(f"Total trades: {trade_count}")

            # Show recent signals (if any)
            if signal_count > 0:
                print("\n" + "=" * 70)
                print("Recent Signals (last 5):")
                print("=" * 70)
                result = conn.execute(text("""
                    SELECT
                        sig.timestamp,
                        s.strategy_name,
                        sig.signal_type,
                        sig.confidence,
                        sig.price_at_signal,
                        sig.was_executed,
                        sig.rejection_reason
                    FROM signals sig
                    JOIN strategies s ON sig.strategy_id = s.id
                    ORDER BY sig.timestamp DESC
                    LIMIT 5
                """))

                for row in result:
                    timestamp = format_timestamp(row[0])
                    strategy = row[1]
                    signal_type = row[2]
                    confidence = f"{row[3]:.2%}" if row[3] else "N/A"
                    price = f"${row[4]:,.2f}" if row[4] else "N/A"
                    executed = "[EXEC]" if row[5] else "[SUPP]"
                    reason = f" ({row[6]})" if row[6] else ""

                    print(f"  {timestamp} | {strategy:20s} | {signal_type:8s} | "
                          f"Conf: {confidence:7s} | Price: {price:12s} | {executed}{reason}")

            # Show recent trades (if any)
            if trade_count > 0:
                print("\n" + "=" * 70)
                print("Recent Trades (last 5):")
                print("=" * 70)
                result = conn.execute(text("""
                    SELECT
                        t.timestamp,
                        s.strategy_name,
                        t.trade_type,
                        t.size,
                        t.price,
                        t.fees
                    FROM trades t
                    JOIN strategies s ON t.strategy_id = s.id
                    ORDER BY t.timestamp DESC
                    LIMIT 5
                """))

                for row in result:
                    timestamp = format_timestamp(row[0])
                    strategy = row[1]
                    trade_type = row[2]
                    size = f"${row[3]:,.2f}"
                    price = f"${row[4]:,.2f}"
                    fees = f"${row[5]:.2f}"

                    print(f"  {timestamp} | {strategy:20s} | {trade_type:12s} | "
                          f"Size: {size:12s} | Price: {price:12s} | Fees: {fees}")

            # Show strategy performance summary
            print("\n" + "=" * 70)
            print("Strategy Performance Summary:")
            print("=" * 70)
            result = conn.execute(text("""
                SELECT
                    strategy_name,
                    num_open_positions,
                    num_closed_positions,
                    total_unrealized_pnl,
                    total_realized_pnl,
                    num_wins,
                    num_losses
                FROM v_strategy_performance
            """))

            rows = result.fetchall()
            if rows:
                for row in rows:
                    strategy = row[0]
                    open_pos = row[1]
                    closed_pos = row[2]
                    unrealized = row[3] or 0.0
                    realized = row[4] or 0.0
                    wins = row[5] or 0
                    losses = row[6] or 0
                    total_pnl = unrealized + realized
                    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

                    print(f"\n  Strategy: {strategy}")
                    print(f"    Open positions: {open_pos} | Closed positions: {closed_pos}")
                    print(f"    Unrealized P&L: ${unrealized:,.2f} | Realized P&L: ${realized:,.2f}")
                    print(f"    Total P&L: ${total_pnl:,.2f}")
                    print(f"    Win rate: {win_rate:.1f}% ({wins}W / {losses}L)")
            else:
                print("  No strategies found")

        print("\n" + "=" * 70)
        state_manager.close()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. DATABASE_URL is set in your .env file, or")
        print("2. The database exists at execution/state/paper_trading.db (SQLite)")
        sys.exit(1)


if __name__ == "__main__":
    main()
