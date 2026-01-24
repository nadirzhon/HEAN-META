# HEAN-META Production Deployment Guide

**Version:** 1.0
**Last Updated:** 2026-01-24
**Status:** Production Ready ✅

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Environment Configuration](#environment-configuration)
4. [Docker Deployment](#docker-deployment)
5. [Post-Deployment Verification](#post-deployment-verification)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Disk: 20GB SSD
- OS: Ubuntu 20.04+ / Debian 11+ / RHEL 8+
- Docker: 20.10+
- Docker Compose: 1.29+

**Recommended:**
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB NVMe SSD
- Low-latency network connection

### Software Prerequisites

```bash
# Check Docker
docker --version  # Should be 20.10+
docker-compose --version  # Should be 1.29+

# Check system resources
free -h
df -h
nproc
```

### Network Requirements

- Stable internet connection
- Access to Bybit API endpoints:
  - **Testnet:** https://api-testnet.bybit.com
  - **Mainnet:** https://api.bybit.com
- Outbound HTTPS (443) allowed
- Optional: Inbound access to ports 3000, 8000 (if exposing UI/API)

---

## Pre-Deployment Checklist

### 1. Bybit API Setup

- [ ] Create Bybit account (testnet or mainnet)
- [ ] Generate API keys with appropriate permissions:
  - **Testnet:** https://testnet.bybit.com/app/user/api-management
  - **Mainnet:** https://www.bybit.com/app/user/api-management
- [ ] Required API permissions:
  - Read-only: Market data, Account info
  - Trading: Place/Cancel orders (for live trading)
  - Wallet: Read balance
- [ ] Store API keys securely (password manager, secrets vault)
- [ ] Test API connection (see Testing section)

### 2. Security Review

- [ ] API keys stored securely (NOT in git)
- [ ] `.env` file configured with real credentials
- [ ] `.env` excluded from version control (check `.gitignore`)
- [ ] Risk limits configured appropriately
- [ ] Killswitch parameters set
- [ ] Understand live trading implications

### 3. Risk Management Configuration

- [ ] Set `MAX_DAILY_DRAWDOWN_PCT` (recommended: 5-10%)
- [ ] Set `MAX_TRADE_RISK_PCT` (recommended: 0.5-2%)
- [ ] Set `INITIAL_CAPITAL` to actual trading capital
- [ ] Review strategy configurations
- [ ] Understand profit capture settings

---

## Environment Configuration

### Step 1: Clone and Navigate

```bash
# Clone repository
git clone <your-repo-url> hean-meta
cd hean-meta

# Ensure on correct branch
git checkout claude/smoke-test-production-PER5E
```

### Step 2: Create Production Environment File

```bash
# Copy production template
cp .env.production .env

# Edit with your settings
nano .env  # or vim, emacs, etc.
```

### Step 3: Configure Critical Variables

Edit `.env` and set these **REQUIRED** variables:

```bash
# =============================================================================
# BYBIT API (REQUIRED)
# =============================================================================
BYBIT_TESTNET=true  # Set to false for mainnet!
BYBIT_API_KEY=your-actual-api-key-here
BYBIT_API_SECRET=your-actual-api-secret-here

# =============================================================================
# TRADING MODE (REQUIRED)
# =============================================================================
TRADING_MODE=paper  # Change to "live" when ready!
# LIVE_CONFIRM=YES  # Uncomment for live trading!

# =============================================================================
# RISK LIMITS (REQUIRED)
# =============================================================================
INITIAL_CAPITAL=10000  # Your actual trading capital
MAX_DAILY_DRAWDOWN_PCT=5.0
MAX_TRADE_RISK_PCT=1.0

# =============================================================================
# STRATEGIES (CONFIGURE AS NEEDED)
# =============================================================================
FUNDING_HARVESTER_ENABLED=true
BASIS_ARBITRAGE_ENABLED=true
IMPULSE_ENGINE_ENABLED=true

# =============================================================================
# SYMBOLS (CONFIGURE AS NEEDED)
# =============================================================================
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
```

### Step 4: Validate Configuration

```bash
# Check .env file exists
test -f .env && echo "✅ .env file found" || echo "❌ .env file missing!"

# Verify critical variables are set
grep -q "BYBIT_API_KEY=your-" .env && echo "⚠️ WARNING: API key not set!" || echo "✅ API key configured"
grep -q "TRADING_MODE" .env && echo "✅ Trading mode set" || echo "❌ Trading mode missing!"
```

---

## Docker Deployment

### Step 1: Build Docker Images

```bash
# Build all services
docker-compose build

# Expected output:
# - Building api...
# - Building ui...
# - Pulling redis...
```

### Step 2: Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Check service status
docker-compose ps

# Expected output:
# NAME          STATE    PORTS
# hean-api      Up       0.0.0.0:8000->8000/tcp
# hean-ui       Up       0.0.0.0:3000->80/tcp
# hean-redis    Up       0.0.0.0:6379->6379/tcp
```

### Step 3: Monitor Startup

```bash
# Watch API logs
docker-compose logs -f api

# Look for these key messages:
# - "Started server process"
# - "Application startup complete"
# - "Uvicorn running on http://0.0.0.0:8000"
# - "Redis connection OK"

# Press Ctrl+C to exit logs
```

### Step 4: Alternative - Development Mode with Hot Reload

```bash
# Start UI in development mode with hot-reload
docker-compose --profile dev up -d

# This starts:
# - api (with --reload)
# - ui-dev (with Vite hot-reload on port 5173)
# - redis
```

---

## Post-Deployment Verification

### Step 1: Run Smoke Test

```bash
# Run comprehensive smoke test
./scripts/smoke_test.sh localhost 8000

# Expected output:
# [TEST 1] Health check ... ✓ PASS
# [TEST 2] Telemetry ping ... ✓ PASS
# [TEST 3] Telemetry summary ... ✓ PASS
# ...
# ✅ ALL TESTS PASSED - System is operational
```

### Step 2: Verify API Endpoints

```bash
# Health check
curl http://localhost:8000/health | jq

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "api": "healthy",
#     "event_bus": "running",
#     "redis": "healthy",
#     "engine": "running"
#   }
# }

# Trading diagnostics
curl http://localhost:8000/trading/why | jq

# Portfolio summary
curl http://localhost:8000/portfolio/summary | jq
```

### Step 3: Check Frontend UI

```bash
# Access in browser
open http://localhost:3000

# Or test with curl
curl http://localhost:3000 | grep -q "HEAN" && echo "✅ UI accessible" || echo "❌ UI error"
```

### Step 4: Verify Trading Engine

```bash
# Check engine state
curl -s http://localhost:8000/trading/why | jq '.engine_state'
# Expected: "running"

# Check active positions
curl -s http://localhost:8000/orders/positions | jq 'length'
# Shows number of active positions

# Check orders
curl -s http://localhost:8000/orders | jq 'length'
# Shows number of orders
```

---

## Monitoring & Maintenance

### Real-Time Monitoring

```bash
# Monitor all logs
docker-compose logs -f

# Monitor specific service
docker-compose logs -f api
docker-compose logs -f ui

# Monitor last 100 lines
docker-compose logs --tail=100 api

# Monitor with timestamps
docker-compose logs -f --timestamps api
```

### Health Checks

```bash
# Check all container health
docker-compose ps

# Detailed health status
docker inspect hean-api --format='{{.State.Health.Status}}'
docker inspect hean-ui --format='{{.State.Health.Status}}'
docker inspect hean-redis --format='{{.State.Health.Status}}'
```

### Resource Monitoring

```bash
# Check container resource usage
docker stats

# Check disk usage
docker system df

# Check logs disk usage
du -sh logs/
```

### Regular Maintenance Tasks

```bash
# Daily: Check system health
./scripts/smoke_test.sh localhost 8000

# Daily: Review logs for errors
docker-compose logs api | grep -i error | tail -50

# Weekly: Clean up old logs
find logs/ -name "*.log" -mtime +7 -delete

# Monthly: Update Docker images
docker-compose pull
docker-compose up -d

# Monthly: Prune unused Docker resources
docker system prune -af --volumes
```

---

## Troubleshooting

### Issue: API Container Won't Start

**Symptoms:**
- `docker-compose ps` shows api container as "Exit 1" or "Restarting"

**Solution:**
```bash
# Check logs
docker-compose logs api | tail -50

# Common fixes:
# 1. Redis connection issue
#    - Check Redis is running: docker-compose ps redis
#    - Check Redis logs: docker-compose logs redis

# 2. Syntax errors in Python code
#    - Review smoke test report: cat SMOKE_TEST_REPORT.md
#    - Check for recent code changes

# 3. Missing environment variables
#    - Verify .env file exists: ls -la .env
#    - Check critical vars: grep "BYBIT_API_KEY" .env

# Restart after fixes
docker-compose restart api
```

### Issue: UI Not Accessible

**Symptoms:**
- Cannot access http://localhost:3000
- nginx returns 502 Bad Gateway

**Solution:**
```bash
# Check UI container status
docker-compose ps ui

# Check if API is healthy (UI depends on it)
curl http://localhost:8000/health

# Check nginx logs
docker-compose logs ui | tail -50

# Verify nginx config
docker exec hean-ui cat /etc/nginx/conf.d/default.conf

# Restart UI
docker-compose restart ui
```

### Issue: Killswitch Triggered

**Symptoms:**
- `/trading/why` shows `killswitch_state.triggered: true`
- Trading stopped

**Solution:**
```bash
# Check reason
curl -s http://localhost:8000/trading/why | jq '.killswitch_state'

# Common triggers:
# - Daily drawdown exceeded: Adjust MAX_DAILY_DRAWDOWN_PCT
# - Repeated errors: Check logs for errors
# - Equity drop: Review position management

# Reset killswitch (API endpoint)
curl -X POST http://localhost:8000/engine/reset-killswitch

# Monitor after reset
watch -n 5 'curl -s http://localhost:8000/trading/why | jq .killswitch_state'
```

### Issue: No Orders Being Created

**Symptoms:**
- `/orders` returns empty or very few orders
- Engine running but not trading

**Solution:**
```bash
# Check decision reasons
curl -s http://localhost:8000/trading/why | jq '.top_reason_codes_last_5m'

# Common reasons:
# - "DRY_RUN": Check TRADING_MODE in .env
# - "KILLSWITCH": See killswitch troubleshooting above
# - "NO_SIGNAL": Market conditions not favorable
# - "LIMIT_REACHED": Position or risk limits hit

# Check strategy status
curl -s http://localhost:8000/strategies | jq

# Enable more strategies if needed
# Edit .env and set *_ENABLED=true, then:
docker-compose restart api
```

### Issue: High Memory Usage

**Symptoms:**
- `docker stats` shows high memory usage
- System becoming slow

**Solution:**
```bash
# Check current usage
docker stats --no-stream

# Reduce resource limits in docker-compose.yml:
# api:
#   deploy:
#     resources:
#       limits:
#         memory: 1G  # Reduce from 2G

# Apply changes
docker-compose up -d --force-recreate

# Monitor Redis memory
docker exec hean-redis redis-cli INFO memory | grep used_memory_human
```

---

## Rollback Procedures

### Quick Rollback

```bash
# Stop all services
docker-compose down

# Restore previous .env if needed
cp .env.backup .env

# Pull previous image version
docker pull hean-api:previous-version
docker pull hean-ui:previous-version

# Start with previous version
docker-compose up -d
```

### Data Preservation

```bash
# Before any major changes, backup:

# 1. Backup environment
cp .env .env.backup.$(date +%Y%m%d)

# 2. Backup Redis data
docker exec hean-redis redis-cli SAVE
docker cp hean-redis:/data/dump.rdb ./backup/redis-dump-$(date +%Y%m%d).rdb

# 3. Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/

# 4. Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz .env docker-compose.yml backend.env ui.env
```

### Emergency Stop

```bash
# Stop all trading immediately
docker-compose stop api

# Or stop everything
docker-compose down

# Verify stopped
docker-compose ps
```

---

## Production Checklist

### Before Going Live

- [ ] ✅ Smoke test passed (see SMOKE_TEST_REPORT.md)
- [ ] ✅ All syntax errors fixed
- [ ] ✅ API endpoints responding correctly
- [ ] ✅ Redis connected and operational
- [ ] ✅ Docker services healthy
- [ ] ⚠️ Set `BYBIT_TESTNET=false` for mainnet
- [ ] ⚠️ Set `TRADING_MODE=live`
- [ ] ⚠️ Set `LIVE_CONFIRM=YES`
- [ ] ⚠️ Real API credentials configured
- [ ] ⚠️ Risk limits appropriate for capital
- [ ] ⚠️ Monitoring and alerts configured
- [ ] ⚠️ Backup and rollback procedures tested

### Going Live Procedure

```bash
# 1. Start with paper trading for 24-48 hours
TRADING_MODE=paper docker-compose up -d

# 2. Monitor and validate
./scripts/smoke_test.sh localhost 8000
# Monitor logs for 24 hours

# 3. Switch to live trading
# Edit .env:
# - Set TRADING_MODE=live
# - Set LIVE_CONFIRM=YES

# 4. Restart with live settings
docker-compose restart api

# 5. Start with minimal capital
# Monitor closely for first few hours

# 6. Gradually increase capital
# Only after validating successful operation
```

---

## Support & Resources

### Documentation
- [README.md](README.md) - Project overview and quick start
- [SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md) - Detailed test results
- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Docker-specific documentation
- [docs/API.md](docs/API.md) - API endpoint reference

### Logs
- API logs: `docker-compose logs api`
- System logs: `logs/` directory
- Redis logs: `docker-compose logs redis`

### Health Endpoints
- System health: http://localhost:8000/health
- Trading diagnostics: http://localhost:8000/trading/why
- Telemetry: http://localhost:8000/telemetry/summary

---

## Appendix

### Useful Commands

```bash
# Quick health check
curl -s http://localhost:8000/health | jq .status

# Check trading state
curl -s http://localhost:8000/trading/why | jq .engine_state

# View recent events
curl -s http://localhost:8000/telemetry/summary | jq .last_event_ts

# Restart specific service
docker-compose restart api

# View resource usage
docker stats --no-stream

# Clean up stopped containers
docker-compose down && docker-compose up -d
```

### Performance Tuning

```bash
# Optimize Redis for trading
docker exec hean-redis redis-cli CONFIG SET maxmemory 512mb
docker exec hean-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Increase API workers (if needed)
# Edit docker-compose.yml:
# command: uvicorn hean.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Monitor latency
docker-compose logs api | grep -i "latency"
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-24
**Status:** ✅ Production Ready
**Deployment Confidence:** 🟢 HIGH

**Ready for Production Deployment!**
