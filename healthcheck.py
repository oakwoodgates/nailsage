#!/usr/bin/env python3
"""
Health check script for Docker containers.
Returns exit code 0 (healthy) or 1 (unhealthy).
"""

import argparse
import os
import sys
from pathlib import Path


def check_database_connection() -> bool:
    """Check if database is accessible."""
    try:
        import psycopg2
        database_url = os.getenv('DATABASE_URL', '')

        if not database_url:
            print("ERROR: DATABASE_URL not set", file=sys.stderr)
            return False

        # Parse DATABASE_URL
        if database_url.startswith('postgresql://'):
            # Try to connect
            conn = psycopg2.connect(database_url)
            conn.close()
            return True
        elif database_url.startswith('sqlite://'):
            # SQLite - check if file exists
            db_path = database_url.replace('sqlite:///', '')
            return Path(db_path).exists()
        else:
            print(f"ERROR: Unknown database type: {database_url}", file=sys.stderr)
            return False

    except ImportError:
        # psycopg2 not installed, try sqlalchemy
        try:
            from sqlalchemy import create_engine
            database_url = os.getenv('DATABASE_URL', '')
            if not database_url:
                return False
            engine = create_engine(database_url)
            conn = engine.connect()
            conn.close()
            return True
        except Exception as e:
            print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        return False


def check_strategy_container() -> bool:
    """Health check for strategy containers."""
    # Check database connection
    if not check_database_connection():
        return False

    # Check if models directory is accessible
    models_dir = Path('/app/models/trained')
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}", file=sys.stderr)
        return False

    # Check if at least one model file exists
    model_files = list(models_dir.glob('*.joblib'))
    if not model_files:
        print(f"WARNING: No model files found in {models_dir}", file=sys.stderr)
        # Don't fail health check, might be in training phase

    return True


def check_api_container() -> bool:
    """Health check for API container."""
    # Check database connection
    if not check_database_connection():
        return False

    # Check if API server is responding (optional, HTTP check is better)
    return True


def main():
    parser = argparse.ArgumentParser(description='Docker container health check')
    parser.add_argument(
        '--container-type',
        choices=['strategy', 'api'],
        default='strategy',
        help='Type of container to check'
    )
    args = parser.parse_args()

    try:
        if args.container_type == 'strategy':
            healthy = check_strategy_container()
        elif args.container_type == 'api':
            healthy = check_api_container()
        else:
            print(f"ERROR: Unknown container type: {args.container_type}", file=sys.stderr)
            sys.exit(1)

        if healthy:
            print(f"Container ({args.container_type}) is healthy")
            sys.exit(0)
        else:
            print(f"Container ({args.container_type}) is unhealthy", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Health check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
