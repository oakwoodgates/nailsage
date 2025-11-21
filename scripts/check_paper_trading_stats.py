"""Quick script to check paper trading statistics."""
import sqlite3
from pathlib import Path

db_path = Path("execution/state/paper_trading.db")

if not db_path.exists():
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check signals
cursor.execute("SELECT COUNT(*) FROM signals")
signal_count = cursor.fetchone()[0]
print(f"Total signals generated: {signal_count}")

# Check positions
cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
open_positions = cursor.fetchone()[0]
print(f"Open positions: {open_positions}")

cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'closed'")
closed_positions = cursor.fetchone()[0]
print(f"Closed positions: {closed_positions}")

# Check trades
cursor.execute("SELECT COUNT(*) FROM trades")
trade_count = cursor.fetchone()[0]
print(f"Total trades: {trade_count}")

# Show recent signals (if any)
if signal_count > 0:
    print("\nRecent signals:")
    cursor.execute("""
        SELECT timestamp, strategy_name, signal_type, confidence, position_size_usd
        FROM signals
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row}")

# Show recent trades (if any)
if trade_count > 0:
    print("\nRecent trades:")
    cursor.execute("""
        SELECT timestamp, position_id, side, quantity, price
        FROM trades
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row}")

conn.close()
