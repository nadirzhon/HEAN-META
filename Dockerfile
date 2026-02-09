# HEAN SYMBIONT X - Docker Image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY backend.env .env

# Create data directories
RUN mkdir -p /app/data/symbiont_data

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-c", "from hean.symbiont_x import HEANSymbiontX; print('HEAN SYMBIONT X - Docker image ready')"]
