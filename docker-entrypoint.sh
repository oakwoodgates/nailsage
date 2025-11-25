#!/bin/bash
set -e

echo "========================================="
echo "Nailsage Docker Entrypoint"
echo "========================================="

# Validate required environment variables
echo "Validating environment variables..."

if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

# Only validate Kirby settings for strategy containers
if [ ! -z "$EXCHANGE" ]; then
    echo "Container type: Strategy ($EXCHANGE)"

    if [ -z "$KIRBY_WS_URL" ]; then
        echo "ERROR: KIRBY_WS_URL is not set"
        exit 1
    fi

    if [ -z "$KIRBY_API_KEY" ]; then
        echo "ERROR: KIRBY_API_KEY is not set"
        exit 1
    fi

    if [ -z "$STRATEGY_IDS" ]; then
        echo "WARNING: STRATEGY_IDS is not set, no strategies will run"
    else
        echo "Strategies to run: $STRATEGY_IDS"
    fi
fi

# Wait for database to be ready
echo "Waiting for database to be ready..."
timeout=30
counter=0

while ! python -c "
import sys
import os
try:
    from sqlalchemy import create_engine
    engine = create_engine(os.getenv('DATABASE_URL'))
    conn = engine.connect()
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f'Database not ready: {e}')
    sys.exit(1)
" 2>/dev/null; do
    counter=$((counter + 1))
    if [ $counter -ge $timeout ]; then
        echo "ERROR: Database connection timeout after ${timeout} seconds"
        exit 1
    fi
    echo "Waiting for database... ($counter/$timeout)"
    sleep 1
done

echo "Database is ready!"

# Initialize database tables if they don't exist (only for strategy containers)
if [ ! -z "$EXCHANGE" ]; then
    echo "Checking database schema..."
    python -c "
import os
from sqlalchemy import create_engine, inspect

engine = create_engine(os.getenv('DATABASE_URL'))
inspector = inspect(engine)
tables = inspector.get_table_names()

if 'strategies' not in tables:
    print('Database tables not found. They should be initialized by schema.sql.')
    print('Available tables:', tables)
else:
    print(f'Database schema verified. Found {len(tables)} tables.')
"
fi

# Create runtime directories if they don't exist
echo "Creating runtime directories..."
mkdir -p /app/logs
mkdir -p /app/features/cache

# Set proper permissions (non-fatal for volume mounts)
chmod -R 755 /app/logs 2>/dev/null || echo "Note: Could not set permissions on /app/logs (volume mount)"
chmod -R 755 /app/features/cache 2>/dev/null || echo "Note: Could not set permissions on /app/features/cache (volume mount)"

echo "========================================="
echo "Starting application..."
echo "Command: $@"
echo "========================================="

# Execute the main command
exec "$@"
