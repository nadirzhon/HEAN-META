FROM python:3.11-slim AS cpp-builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY cpp_core ./cpp_core
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install nanobind && \
    cd cpp_core && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install || echo "C++ build skipped (optional)"

FROM python:3.11-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add build-essential for compiling native extensions (polars, orjson, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libffi-dev \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Build wheels - with retry for network issues
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --wheel-dir /wheels . || \
    (echo "Retrying wheel build..." && pip wheel --wheel-dir /wheels .)

FROM python:3.11-slim

# Create non-root user BEFORE any file operations
RUN groupadd --gid 1000 hean && \
    useradd --uid 1000 --gid hean --shell /bin/false --create-home hean

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOME=/home/hean

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

COPY --from=builder /wheels /wheels

RUN pip install --upgrade pip && \
    pip install /wheels/*.whl && \
    pip install "openai>=1.0.0" "anthropic>=0.18.0" && \
    pip install "google-generativeai>=0.3.0" || true && \
    pip install "protobuf>=4.25.0" && \
    pip install nanobind || true && \
    rm -rf /wheels

# Copy source code with correct ownership
COPY --chown=hean:hean src ./src

# Copy compiled C++ modules AFTER src (optional - may not exist)
# Using shell to handle missing files gracefully
RUN mkdir -p /app/src/hean/cpp_modules || true

# Create necessary directories with correct ownership
RUN mkdir -p /app/logs /app/data && \
    chown -R hean:hean /app

# Switch to non-root user
USER hean

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "hean.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
