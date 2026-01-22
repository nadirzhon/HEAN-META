FROM python:3.11-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --wheel-dir /wheels .

FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --upgrade pip && \
    pip install /wheels/*.whl && \
    pip install "openai>=1.0.0" "anthropic>=0.18.0" && \
    pip install "google-generativeai>=0.3.0" --no-deps || pip install "google-generativeai>=0.3.0" && \
    pip install "protobuf>=4.25.0" && \
    python3 -c "import google.generativeai; print('✅ Gemini package installed successfully')" || (echo "⚠ Warning: Gemini package check failed, but continuing..." && pip install "google-generativeai>=0.3.0" --force-reinstall)

COPY src ./src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "hean.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

