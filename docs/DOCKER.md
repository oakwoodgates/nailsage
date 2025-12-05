# Docker Deployment Guide for Nailsage

This guide covers Docker containerization for the Nailsage trading platform, from local development to production deployment on DigitalOcean.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Development Workflow](#development-workflow)
- [Production Deployment](#production-deployment)
- [Container Management](#container-management)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Nailsage platform runs in a multi-container Docker environment with:

- **PostgreSQL**: Persistent database for all trading data
- **API Container**: FastAPI server for dashboard backend (REST + WebSocket)
- **Strategy Containers**: One per exchange (Binance, Hyperliquid, etc.)

### Key Benefits

- **Exchange-level fault isolation**: Binance outage doesn't affect Hyperliquid strategies
- **Shared resources**: Multiple strategies per exchange share WebSocket connection
- **Easy scaling**: Add new strategies or exchanges by updating docker-compose.yml
- **Portable**: Same Docker setup works locally and in production

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│            Frontend Dashboard (External)         │
└───────────────────┬─────────────────────────────┘
                    │ HTTP/WebSocket
                    ▼
┌─────────────────────────────────────────────────┐
│          nailsage-api (Port 8000)                │
│  - REST API endpoints                            │
│  - WebSocket for live updates                    │
│  - Polls database every 2 seconds                │
└───────────────────┬─────────────────────────────┘
                    │ SQL Queries
                    ▼
┌─────────────────────────────────────────────────┐
│          PostgreSQL Database                     │
│  - Tables: strategies, positions, trades         │
│  - Persistent volume: postgres-data              │
└─────┬─────────────────────┬─────────────────────┘
      │                     │
      ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ nailsage-binance │  │nailsage-hyperl..│
│  - BTC momentum  │  │  - BTC perps    │
│  - SOL swing     │  │  - ETH perps    │
│  - Shared WS     │  │  - Shared WS    │
└──────────────────┘  └──────────────────┘
```

---

## Prerequisites

### Local Development

- Docker Desktop (Windows/Mac) or Docker Engine + Docker Compose (Linux)
- 8 GB RAM minimum (16 GB recommended)
- 20 GB free disk space

### Production (DigitalOcean)

- DigitalOcean account
- Droplet: Basic plan with 2 GB RAM ($12/mo) for MVP
- SSH key for secure access

---

## Local Development

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd nailsage
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set:
# - MODE=dev
# - KIRBY_WS_URL_DEV=ws://localhost:8000/ws (or your local Kirby)
# - KIRBY_API_KEY_DEV=your_dev_api_key
# - DATABASE_URL=sqlite:///execution/state/paper_trading.db (for local SQLite)
```

**Important**: For local development, you can use SQLite (default) or Postgres. For production, use Postgres.

### Step 3: Build Docker Images

```bash
# Build all images (takes 5-10 minutes first time due to TA-Lib compilation)
docker-compose build

# Or build specific service:
docker-compose build nailsage-binance
```

### Step 4: Start Services

```bash
# Start all services
docker-compose up -d

# Or start specific services:
docker-compose up -d postgres nailsage-api nailsage-binance

# View logs (follow mode):
docker-compose logs -f

# View logs for specific service:
docker-compose logs -f nailsage-binance
```

### Step 5: Verify Services

```bash
# Check running containers
docker ps

# Check API health
curl http://localhost:8000/health

# Check strategies
curl http://localhost:8000/strategies

# Check database (if using Postgres locally)
docker-compose exec postgres psql -U nailsage -d nailsage -c "SELECT * FROM strategies;"
```

### Step 6: Stop Services

```bash
# Stop all containers (keeps data)
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v
```

---

## Development Workflow

This section covers the iterative development workflow for testing and debugging strategies in Docker.

### Quick Command Reference

```bash
# Most common: Rebuild after code changes
docker compose build nailsage-binance && docker compose up -d nailsage-binance

# Just restart (no code changes)
docker compose restart nailsage-binance

# Stop container
docker compose stop nailsage-binance

# View live logs
docker logs nailsage-binance -f

# View last 100 lines
docker logs nailsage-binance --tail 100

# Check status
docker compose ps

# Shell into container
docker exec -it nailsage-binance sh
```

### When to Rebuild vs Restart

**Rebuild Required** (Python code changes):
- Changes to `.py` files in `scripts/`, `execution/`, `features/`, `models/`, etc.
- Changes to `pyproject.toml` dependencies
- Changes to `Dockerfile` or `docker-entrypoint.sh`

**Command**:
```bash
docker compose build nailsage-binance && docker compose up -d nailsage-binance
```

**No Rebuild Needed** (volume-mounted files):
- Model files in `models/trained/` or `models/metadata/`
- Strategy configs in `strategies/` (YAML files)
- Data files in `data/raw/`
- `.env` file changes
- `docker-compose.yml` changes

**Command**:
```bash
docker compose up -d nailsage-binance
# or for just a restart:
docker compose restart nailsage-binance
```

### Typical Development Iteration

1. **Make code changes** (e.g., update signal generator logic)

2. **Rebuild and restart**:
   ```bash
   docker compose build nailsage-binance && docker compose up -d nailsage-binance
   ```

3. **Monitor logs**:
   ```bash
   docker logs nailsage-binance -f
   ```

4. **Debug issues**:
   ```bash
   # Shell into container
   docker exec -it nailsage-binance sh

   # Check Python environment
   python --version
   pip list | grep <package>

   # List files
   ls -la /app/models/metadata/

   # Test imports
   python -c "from execution.inference.predictor import ModelPredictor"
   ```

5. **Repeat** until working

### Testing New Strategies

**Add a new strategy without stopping running containers:**

1. **Create strategy config**:
   ```bash
   # Create strategies/btc_scalp_v1.yaml
   # (no rebuild needed - volume mounted)
   ```

2. **Train and register model**:
   ```bash
   # Train locally (outside Docker)
   python scripts/train_strategy.py strategies/btc_scalp_v1.yaml

   # Model artifacts automatically available to container via volume mount
   ```

3. **Update environment**:
   ```bash
   # Edit .env
   BINANCE_STRATEGY_IDS=sol_swing_momentum_v1,btc_scalp_v1
   BINANCE_STARLISTING_IDS=2,3
   ```

4. **Restart container** (picks up new .env):
   ```bash
   docker compose up -d nailsage-binance
   ```

### Switching Between Dev and Prod Kirby

**Local Kirby (Development)**:
```bash
# Edit .env
MODE=dev
KIRBY_WS_URL_DEV=ws://host.docker.internal:8000/ws

# Or edit docker-compose.yml
MODE: dev

# Restart
docker compose up -d nailsage-binance
```

**Production Kirby (DigitalOcean)**:
```bash
# Edit docker-compose.yml
MODE: prod

# Restart
docker compose up -d nailsage-binance
```

### Database Operations

**View data in PostgreSQL**:
```bash
# Connect to database
docker exec -it nailsage-postgres psql -U nailsage -d nailsage

# Query positions
SELECT * FROM positions ORDER BY entry_timestamp DESC LIMIT 10;

# Query signals
SELECT * FROM signals WHERE was_executed = TRUE ORDER BY timestamp DESC;

# Exit
\q
```

**Reset database** (delete all data):
```bash
docker compose down -v  # Removes volumes
docker compose up -d    # Recreates with fresh schema
```

**Backup database**:
```bash
docker exec nailsage-postgres pg_dump -U nailsage nailsage > backup.sql
```

**Restore database**:
```bash
cat backup.sql | docker exec -i nailsage-postgres psql -U nailsage -d nailsage
```

### Performance Optimization

**View resource usage**:
```bash
docker stats nailsage-binance
```

**Check container logs size**:
```bash
docker inspect nailsage-binance --format='{{.LogPath}}' | xargs ls -lh
```

**Limit log size** (add to docker-compose.yml):
```yaml
nailsage-binance:
  logging:
    driver: "json-file"
    options:
      max-size: "10m"
      max-file: "3"
```

### Troubleshooting Common Issues

**Container keeps restarting**:
```bash
# Check logs for errors
docker logs nailsage-binance --tail 200

# Common causes:
# - Missing .env variables
# - Database connection failure
# - Model files not found
# - Invalid strategy config
```

**Model not found**:
```bash
# Verify model files are present
docker exec nailsage-binance ls -la /app/models/trained/
docker exec nailsage-binance ls -la /app/models/metadata/

# Check if metadata has correct strategy_name
docker exec nailsage-binance cat /app/models/metadata/<model_id>.json
```

**WebSocket connection failed**:
```bash
# Check Kirby URL
docker exec nailsage-binance env | grep KIRBY

# Test connection from container
docker exec nailsage-binance ping 143.198.18.115
```

**Database errors**:
```bash
# Check database is running
docker compose ps postgres

# Check database logs
docker logs nailsage-postgres --tail 100

# Verify connection
docker exec nailsage-binance python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
conn = engine.connect()
print('Connected!')
"
```

---

## Production Deployment

### Step 1: Provision DigitalOcean Droplet

1. **Create Droplet**:
   - Ubuntu 22.04 LTS
   - Basic plan: 2 GB RAM / 1 vCPU ($12/mo)
   - Choose datacenter region (closest to you/users)
   - Add SSH key for authentication

2. **Enable Backups** (optional, +20%):
   - Weekly snapshots for disaster recovery

3. **Configure Firewall**:
   ```bash
   # Allow SSH and API port only
   ufw allow 22/tcp
   ufw allow 8000/tcp
   ufw enable
   ```

### Step 2: Setup Droplet

SSH into your Droplet:

```bash
ssh root@your-droplet-ip
```

Install Docker and Docker Compose:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose plugin
apt-get install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version
```

Create directories:

```bash
# Create project directories
mkdir -p /opt/nailsage/{app,models/trained,data/raw,secrets,logs,backups}

# Set permissions
chown -R $USER:$USER /opt/nailsage
```

### Step 3: Upload Code and Data

From your local machine:

```bash
# Upload code (use git clone in production)
git clone <your-repo> /opt/nailsage/app

# Or rsync from local:
rsync -avz --exclude 'data/raw' --exclude 'models/trained' \
  ./ root@your-droplet-ip:/opt/nailsage/app/

# Upload models
rsync -avz models/trained/ root@your-droplet-ip:/opt/nailsage/models/trained/

# Upload data (if needed for training in production)
rsync -avz data/raw/ root@your-droplet-ip:/opt/nailsage/data/raw/
```

### Step 4: Configure Environment

On the Droplet:

```bash
# Create production .env
nano /opt/nailsage/secrets/.env
```

Add production configuration:

```bash
# Environment
MODE=prod
LOG_LEVEL=INFO

# Database (PostgreSQL in production)
DATABASE_URL=postgresql://nailsage:YOUR_SECURE_PASSWORD@postgres:5432/nailsage
DB_PASSWORD=YOUR_SECURE_PASSWORD

# Kirby Production
KIRBY_WS_URL_PRO=wss://your-kirby-domain.com/ws
KIRBY_API_KEY_PRO=your_production_api_key

# Starlistings (get from Kirby)
STARLISTING_BTC_USDT_15M=1
STARLISTING_SOL_USDT_4H=4

# Multi-strategy configuration
BINANCE_STRATEGY_IDS=btc_momentum_v1,sol_swing_v1
BINANCE_STARLISTING_IDS=1,4

# Capital
PAPER_TRADING_INITIAL_CAPITAL=100000.0

# API
POLL_INTERVAL=2
```

Secure the file:

```bash
chmod 600 /opt/nailsage/secrets/.env
```

### Step 5: Create Production docker-compose.yml

```bash
cd /opt/nailsage/app
nano docker-compose.prod.yml
```

Production configuration:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: nailsage-postgres
    env_file: /opt/nailsage/secrets/.env
    environment:
      POSTGRES_DB: nailsage
      POSTGRES_USER: nailsage
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - /opt/nailsage/postgres-data:/var/lib/postgresql/data
      - ./execution/persistence/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro
    restart: always
    networks:
      - nailsage-network

  nailsage-api:
    build: .
    container_name: nailsage-api
    command: uvicorn api.server:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: /opt/nailsage/secrets/.env
    volumes:
      - /opt/nailsage/logs:/app/logs
    depends_on:
      - postgres
    restart: unless-stopped
    networks:
      - nailsage-network

  nailsage-binance:
    build: .
    container_name: nailsage-binance
    command: ["sh", "/app/docker-entrypoint.sh", "python", "execution/cli/run_multi_strategy.py"]
    env_file: /opt/nailsage/secrets/.env
    environment:
      EXCHANGE: binance
    volumes:
      - /opt/nailsage/models/trained:/app/models/trained
      - /opt/nailsage/data/raw:/app/data/raw
      - /opt/nailsage/logs:/app/logs
    depends_on:
      - postgres
    restart: unless-stopped
    networks:
      - nailsage-network

networks:
  nailsage-network:
    driver: bridge
```

### Step 6: Deploy

```bash
cd /opt/nailsage/app

# Build images
docker compose -f docker-compose.prod.yml build

# Start services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

### Step 7: Verify Deployment

```bash
# Check containers
docker ps

# Check API
curl http://localhost:8000/health

# Check database
docker compose -f docker-compose.prod.yml exec postgres psql -U nailsage -d nailsage -c "\dt"

# Check strategies
curl http://localhost:8000/strategies
```

---

## Container Management

### Common Commands

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs (last 100 lines)
docker logs --tail 100 nailsage-binance

# Follow logs in real-time
docker logs -f nailsage-binance

# Execute command in container
docker exec -it nailsage-binance python --version

# Access container shell
docker exec -it nailsage-binance bash

# Restart specific container
docker restart nailsage-binance

# Stop container
docker stop nailsage-binance

# Remove stopped containers
docker compose down
```

### Updating Code

```bash
# Pull latest code
cd /opt/nailsage/app
git pull

# Rebuild and restart
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Or for specific service:
docker compose -f docker-compose.prod.yml build nailsage-binance
docker compose -f docker-compose.prod.yml up -d nailsage-binance
```

### Updating Models

```bash
# Upload new model from local machine
rsync -avz models/trained/new_model.joblib root@droplet-ip:/opt/nailsage/models/trained/

# Restart strategy containers to pick up new model
docker restart nailsage-binance
```

### Database Backup

```bash
# Backup PostgreSQL database
docker exec nailsage-postgres pg_dump -U nailsage nailsage | gzip > /opt/nailsage/backups/nailsage_$(date +%Y%m%d).sql.gz

# Restore from backup
gunzip < /opt/nailsage/backups/nailsage_20250124.sql.gz | docker exec -i nailsage-postgres psql -U nailsage nailsage
```

### Viewing Resource Usage

```bash
# Container resource stats
docker stats

# Disk usage
docker system df

# Clean up unused images/containers
docker system prune -a
```

---

## Troubleshooting

### Container Won't Start

**Symptom**: Container exits immediately

**Check logs**:
```bash
docker logs nailsage-binance
```

**Common causes**:
1. Missing environment variables
   - Solution: Check `/opt/nailsage/secrets/.env`
2. Database not ready
   - Solution: Check `docker logs nailsage-postgres`
3. Model files missing
   - Solution: Upload models to `/opt/nailsage/models/trained/`

### Database Connection Errors

**Symptom**: `could not connect to server`

**Check**:
```bash
# Postgres container running?
docker ps | grep postgres

# Can connect from host?
docker exec nailsage-postgres psql -U nailsage -d nailsage -c "SELECT 1;"
```

**Solution**:
```bash
# Restart postgres
docker restart nailsage-postgres

# Check for port conflicts
netstat -tulpn | grep 5432
```

### WebSocket Connection Failures

**Symptom**: Strategies not receiving candles

**Check**:
```bash
# Container logs
docker logs nailsage-binance | grep WebSocket

# Test Kirby connection from container
docker exec nailsage-binance curl -v https://your-kirby-domain.com
```

**Common causes**:
1. Wrong KIRBY_WS_URL
2. Invalid API key
3. Firewall blocking WebSocket
4. Kirby server down

### High Memory Usage

**Symptom**: Container using >1 GB RAM

**Check**:
```bash
docker stats
```

**Solution**:
```bash
# Limit container resources in docker-compose.yml
services:
  nailsage-binance:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

### Logs Not Showing

**Symptom**: `docker logs` shows nothing

**Check**:
```bash
# Ensure app logs to stdout (not file)
# Check docker-entrypoint.sh for redirects
```

### API Returns 500 Error

**Symptom**: API endpoints return internal server error

**Check**:
```bash
# API container logs
docker logs nailsage-api

# Database accessible?
docker exec nailsage-api python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
conn = engine.connect()
print('Database connected')
"
```

---

## Best Practices

### Security

1. **Never commit .env files to git**
2. **Use strong database passwords** (32+ characters)
3. **Restrict firewall** (only ports 22, 8000)
4. **Disable SSH password auth** (use keys only)
5. **Regular security updates**: `apt-get update && apt-get upgrade`

### Monitoring

1. **Check logs daily**: `docker compose logs --tail 100`
2. **Monitor disk space**: `df -h`
3. **Track resource usage**: `docker stats`
4. **Enable DO monitoring** (free in control panel)

### Backups

1. **Database**: Daily automated backups (cron job)
2. **Models**: Version control or S3/Spaces storage
3. **Droplet snapshots**: Weekly via DigitalOcean

### Performance

1. **Log rotation**: Prevent logs from filling disk
2. **Prune regularly**: `docker system prune -a --volumes`
3. **Monitor P&L**: Ensure strategies are profitable
4. **Scale up Droplet** if RAM/CPU consistently >80%

---

## Support

For issues or questions:
- Check GitHub Issues: [your-repo/issues](https://github.com)
- Review logs: `docker compose logs`
- Contact: [your-email]

---

## Appendix

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `MODE` | Environment mode | `dev` or `prod` |
| `DATABASE_URL` | Database connection string | `postgresql://user:pass@host/db` |
| `KIRBY_WS_URL_PRO` | Production Kirby WebSocket URL | `wss://api.domain.com/ws` |
| `KIRBY_API_KEY_PRO` | Production Kirby API key | `kb_xxxxx` |
| `BINANCE_STRATEGY_IDS` | Strategies for Binance container | `btc_momentum_v1,sol_swing_v1` |
| `BINANCE_STARLISTING_IDS` | Starlistings for Binance | `1,4` |
| `POLL_INTERVAL` | API WebSocket poll interval (sec) | `2` |

### Port Reference

| Port | Service | Purpose |
|------|---------|---------|
| 5432 | PostgreSQL | Database (internal only) |
| 8000 | API | Dashboard backend (exposed) |

### Volume Reference

| Volume | Path | Purpose |
|--------|------|---------|
| `postgres-data` | `/var/lib/postgresql/data` | Database files |
| `/opt/nailsage/models/trained` | `/app/models/trained` | ML models |
| `/opt/nailsage/data/raw` | `/app/data/raw` | Training data |
| `/opt/nailsage/logs` | `/app/logs` | Application logs |
