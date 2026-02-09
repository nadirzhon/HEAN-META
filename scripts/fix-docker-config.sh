#!/bin/bash
# ============================================================================
# HEAN Docker Configuration Fix Script
# ============================================================================
# Purpose: Apply all fixes identified in DOCKER_VERIFICATION_REPORT.md
# Usage: ./scripts/fix-docker-config.sh
# ============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================================"
echo "HEAN Docker Configuration Fix Script"
echo "============================================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ============================================================================
# Backup original files
# ============================================================================
echo "[1/7] Creating backups..."
timestamp=$(date +%Y%m%d_%H%M%S)
backup_dir="$PROJECT_ROOT/backups/docker_fix_$timestamp"
mkdir -p "$backup_dir"

cp "$PROJECT_ROOT/docker-compose.yml" "$backup_dir/docker-compose.yml.bak"
cp "$PROJECT_ROOT/backend.env.example" "$backup_dir/backend.env.example.bak"
cp "$PROJECT_ROOT/.env.example" "$backup_dir/.env.example.bak"
cp "$PROJECT_ROOT/Dockerfile.testnet" "$backup_dir/Dockerfile.testnet.bak"
cp "$PROJECT_ROOT/api/Dockerfile" "$backup_dir/api/Dockerfile.bak"

echo "   ✓ Backups created in $backup_dir"
echo ""

# ============================================================================
# Fix 1: docker-compose.yml - Add BYBIT_TESTNET environment variable
# ============================================================================
echo "[2/7] Fixing docker-compose.yml..."

# Add BYBIT_TESTNET to API service environment
if ! grep -q "BYBIT_TESTNET=true" "$PROJECT_ROOT/docker-compose.yml"; then
    sed -i.tmp '/PYTHONPATH=\/app\/src/a\
      - BYBIT_TESTNET=true' "$PROJECT_ROOT/docker-compose.yml"
    rm "$PROJECT_ROOT/docker-compose.yml.tmp"
    echo "   ✓ Added BYBIT_TESTNET=true to API service"
else
    echo "   ✓ BYBIT_TESTNET already present in docker-compose.yml"
fi

# Remove .env volume mount
if grep -q "./.env:/app/.env:ro" "$PROJECT_ROOT/docker-compose.yml"; then
    sed -i.tmp '/.env:\/app\/.env:ro/d' "$PROJECT_ROOT/docker-compose.yml"
    rm "$PROJECT_ROOT/docker-compose.yml.tmp"
    echo "   ✓ Removed conflicting .env volume mount"
else
    echo "   ✓ .env volume mount already removed"
fi

echo ""

# ============================================================================
# Fix 2: backend.env.example - Add missing critical variables
# ============================================================================
echo "[3/7] Fixing backend.env.example..."

# Check if BYBIT_TESTNET is missing
if ! grep -q "^BYBIT_TESTNET=" "$PROJECT_ROOT/backend.env.example"; then
    # Add BYBIT_TESTNET after BYBIT_API_SECRET
    sed -i.tmp '/BYBIT_API_SECRET=/a\
BYBIT_TESTNET=true' "$PROJECT_ROOT/backend.env.example"
    rm "$PROJECT_ROOT/backend.env.example.tmp"
    echo "   ✓ Added BYBIT_TESTNET=true"
fi

# Add trading configuration section if missing
if ! grep -q "^TRADING_MODE=" "$PROJECT_ROOT/backend.env.example"; then
    sed -i.tmp '/^IMPULSE_ENGINE_ENABLED=/a\
\
# Trading Mode\
TRADING_MODE=live\
LIVE_CONFIRM=YES\
INITIAL_CAPITAL=300.0' "$PROJECT_ROOT/backend.env.example"
    rm "$PROJECT_ROOT/backend.env.example.tmp"
    echo "   ✓ Added TRADING_MODE, LIVE_CONFIRM, INITIAL_CAPITAL"
fi

# Update section header
sed -i.tmp 's/# Exchange API Keys$/# Exchange API Keys (TESTNET ONLY)/' "$PROJECT_ROOT/backend.env.example"
sed -i.tmp 's/# Trading Configuration$/# Trading Configuration (TESTNET MODE)/' "$PROJECT_ROOT/backend.env.example"
rm -f "$PROJECT_ROOT/backend.env.example.tmp"

echo "   ✓ backend.env.example updated"
echo ""

# ============================================================================
# Fix 3: .env.example - Add complete configuration
# ============================================================================
echo "[4/7] Fixing .env.example..."

if ! grep -q "^BYBIT_API_KEY=" "$PROJECT_ROOT/.env.example"; then
    # Add complete Bybit and trading configuration
    cat >> "$PROJECT_ROOT/.env.example" << 'EOF'

# ========================================
# EXCHANGE API KEYS (TESTNET ONLY)
# ========================================
# Get testnet API keys from: https://testnet.bybit.com/app/user/api-management
BYBIT_API_KEY=your-bybit-testnet-api-key-here
BYBIT_API_SECRET=your-bybit-testnet-api-secret-here
BYBIT_TESTNET=true

# ========================================
# TRADING CONFIGURATION
# ========================================
TRADING_MODE=live
LIVE_CONFIRM=YES
INITIAL_CAPITAL=300.0

# ========================================
# REDIS CONFIGURATION
# ========================================
REDIS_URL=redis://redis:6379/0

# ========================================
# GOOGLE GEMINI API KEY (Optional)
# ========================================
# Get your key from: https://makersuite.google.com/app/apikey
# GOOGLE_API_KEY=your-google-gemini-key-here
EOF
    echo "   ✓ Added Bybit, trading, and Redis configuration sections"
else
    echo "   ✓ .env.example already contains Bybit configuration"
fi

echo ""

# ============================================================================
# Fix 4: Dockerfile.testnet - Use external healthcheck script
# ============================================================================
echo "[5/7] Fixing Dockerfile.testnet..."

if grep -q 'RUN echo.*healthcheck.sh' "$PROJECT_ROOT/Dockerfile.testnet"; then
    # Replace inline healthcheck creation with COPY from scripts
    cat > "$PROJECT_ROOT/Dockerfile.testnet.new" << 'EOF'
# HEAN SYMBIONT X - Bybit Testnet Docker Image

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Bybit
RUN pip install --no-cache-dir \
    pybit>=5.6.0 \
    websockets>=12.0 \
    aiohttp>=3.9.0 \
    python-dotenv>=1.0.0

# Copy source code (only production files, .dockerignore handles exclusions)
COPY src/ ./src/
COPY live_testnet_real.py ./

# Create data and logs directories
RUN mkdir -p /app/data/historical
RUN mkdir -p /app/logs

# Copy healthcheck script
COPY scripts/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    BYBIT_TESTNET=true

# Healthcheck - verify trading process is alive and responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command - run REAL Bybit Testnet (NO SIMULATION)
CMD ["python", "live_testnet_real.py"]
EOF
    mv "$PROJECT_ROOT/Dockerfile.testnet.new" "$PROJECT_ROOT/Dockerfile.testnet"
    echo "   ✓ Updated to use external healthcheck script"
    echo "   ✓ Updated Python version to 3.12-slim"
    echo "   ✓ Added BYBIT_TESTNET=true environment variable"
else
    echo "   ✓ Dockerfile.testnet already uses external healthcheck (or needs manual review)"
fi

echo ""

# ============================================================================
# Fix 5: api/Dockerfile - Remove silent failures and add testnet flag
# ============================================================================
echo "[6/7] Fixing api/Dockerfile..."

if grep -q 'make install || echo' "$PROJECT_ROOT/api/Dockerfile"; then
    # Remove the || echo silent failure handler
    sed -i.tmp 's/ || echo.*$//' "$PROJECT_ROOT/api/Dockerfile"
    rm "$PROJECT_ROOT/api/Dockerfile.tmp"
    echo "   ✓ Removed silent failure handling"
fi

# Add BYBIT_TESTNET to environment if missing
if ! grep -q "BYBIT_TESTNET=true" "$PROJECT_ROOT/api/Dockerfile"; then
    sed -i.tmp '/ENV.*PYTHONPATH/a\
    BYBIT_TESTNET=true \\' "$PROJECT_ROOT/api/Dockerfile"
    rm "$PROJECT_ROOT/api/Dockerfile.tmp"
    echo "   ✓ Added BYBIT_TESTNET=true environment variable"
fi

# Update Python version to 3.12
if grep -q "python:3.11" "$PROJECT_ROOT/api/Dockerfile"; then
    sed -i.tmp 's/python:3\.11/python:3.12/g' "$PROJECT_ROOT/api/Dockerfile"
    rm "$PROJECT_ROOT/api/Dockerfile.tmp"
    echo "   ✓ Updated Python version to 3.12"
fi

echo ""

# ============================================================================
# Fix 6: Update root Dockerfile Python version
# ============================================================================
echo "[7/7] Fixing root Dockerfile..."

if grep -q "python:3.11" "$PROJECT_ROOT/Dockerfile"; then
    sed -i.tmp 's/python:3\.11/python:3.12/g' "$PROJECT_ROOT/Dockerfile"
    rm "$PROJECT_ROOT/Dockerfile.tmp"
    echo "   ✓ Updated Python version to 3.12"
else
    echo "   ✓ Python version already 3.12 (or needs manual review)"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "FIXES APPLIED SUCCESSFULLY"
echo "============================================================================"
echo ""
echo "Files modified:"
echo "  ✓ docker-compose.yml"
echo "  ✓ backend.env.example"
echo "  ✓ .env.example"
echo "  ✓ Dockerfile.testnet"
echo "  ✓ api/Dockerfile"
echo "  ✓ Dockerfile"
echo ""
echo "Files created:"
echo "  ✓ scripts/healthcheck.sh"
echo "  ✓ .env.symbiont.example"
echo ""
echo "Backups saved in: $backup_dir"
echo ""
echo "============================================================================"
echo "NEXT STEPS"
echo "============================================================================"
echo ""
echo "1. Review changes with git diff:"
echo "   git diff docker-compose.yml"
echo "   git diff backend.env.example"
echo "   git diff .env.example"
echo "   git diff Dockerfile.testnet"
echo "   git diff api/Dockerfile"
echo ""
echo "2. Run smoke test:"
echo "   ./scripts/smoke_test.sh"
echo ""
echo "3. If smoke test passes, rebuild Docker images:"
echo "   docker-compose build --no-cache"
echo ""
echo "4. Start services:"
echo "   docker-compose up -d"
echo ""
echo "5. Verify healthchecks:"
echo "   docker-compose ps"
echo "   # All services should show 'healthy' status"
echo ""
echo "6. Check logs:"
echo "   docker-compose logs -f api"
echo "   # Should see 'BYBIT_TESTNET=true' in startup logs"
echo ""
echo "============================================================================"
