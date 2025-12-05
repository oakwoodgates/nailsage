"""Test saving a signal to debug SQLite error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execution.persistence.state_manager import StateManager, Signal, Strategy

# Initialize database
db_path = Path("execution/state/test_signals.db")
if db_path.exists():
    db_path.unlink()

state_manager = StateManager(db_path)

# Create a strategy first
strategy = Strategy(
    id=None,
    strategy_name="test_strategy",
    version="1.0",
    starlisting_id=2,
    interval="15m",
    model_id="test_model_123",
    config_path=None,
    is_active=True,
)

strategy_id = state_manager.save_strategy(strategy)
print(f"Created strategy with ID: {strategy_id}")

# Create a signal
signal = Signal(
    id=None,
    strategy_id=strategy_id,
    starlisting_id=2,
    signal_type="long",
    confidence=0.85,
    price_at_signal=89500.50,
    timestamp=1732118400000,  # Unix ms
    was_executed=False,
    rejection_reason="confidence_threshold_or_dedup",
)

print(f"\nAttempting to save signal:")
print(f"  strategy_id: {signal.strategy_id} (type: {type(signal.strategy_id)})")
print(f"  starlisting_id: {signal.starlisting_id} (type: {type(signal.starlisting_id)})")
print(f"  signal_type: {signal.signal_type} (type: {type(signal.signal_type)})")
print(f"  confidence: {signal.confidence} (type: {type(signal.confidence)})")
print(f"  price_at_signal: {signal.price_at_signal} (type: {type(signal.price_at_signal)})")
print(f"  timestamp: {signal.timestamp} (type: {type(signal.timestamp)})")
print(f"  was_executed: {signal.was_executed} (type: {type(signal.was_executed)})")
print(f"  rejection_reason: {signal.rejection_reason} (type: {type(signal.rejection_reason)})")

try:
    signal_id = state_manager.save_signal(signal)
    print(f"\n✓ Successfully saved signal with ID: {signal_id}")
except Exception as e:
    print(f"\n✗ Error saving signal: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
state_manager.close()
db_path.unlink()
print("\nTest complete.")
