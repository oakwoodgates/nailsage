# ============================================================================
# Stage 1: Build TA-Lib (Technical Analysis Library)
# ============================================================================
FROM python:3.10-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and compile TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# ============================================================================
# Stage 2: Production Runtime
# ============================================================================
FROM python:3.10-slim-bookworm

# Copy TA-Lib libraries from builder stage
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[exchange]"

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for runtime data
RUN mkdir -p \
    /app/execution/state \
    /app/models/trained \
    /app/data/raw \
    /app/logs \
    /app/features/cache \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check (will be overridden in docker-compose for specific services)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (will be overridden in docker-compose)
CMD ["python", "--version"]
