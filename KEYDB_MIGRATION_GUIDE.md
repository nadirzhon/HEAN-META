# KeyDB Migration Guide

## 🚀 Overview

This guide covers the migration from Redis to **KeyDB**, a high-performance, multi-threaded fork of Redis that offers 2-5x better performance for multi-core systems.

### Why KeyDB?

- **Multi-threaded**: Utilizes multiple CPU cores (vs Redis single-threaded)
- **Drop-in replacement**: Compatible with Redis protocol and clients
- **Better performance**: 2-5x faster for multi-threaded workloads
- **Same features**: Supports all Redis data structures and commands
- **Active development**: Regular updates and improvements

## 📋 Prerequisites

- Docker and Docker Compose installed
- Existing Redis data (optional - can start fresh)
- Python 3.8+ with `redis` and `tqdm` packages

```bash
pip install redis tqdm
```

## 🔄 Migration Steps

### Step 1: Backup Existing Redis Data (If Applicable)

```bash
# If you have existing Redis data, create a backup
docker exec hean-redis redis-cli BGSAVE

# Or export to RDB file
docker exec hean-redis redis-cli --rdb /data/dump.rdb
```

### Step 2: Update Docker Compose Configuration

Use the new `docker-compose.keydb.yml` configuration:

```bash
# Stop current services
docker-compose down

# Use KeyDB configuration
cp docker-compose.keydb.yml docker-compose.yml
```

**Key changes**:
- `redis` service replaced with `keydb`
- Multi-threading enabled (4 threads by default)
- Updated resource limits (4 CPUs vs 1 for Redis)

### Step 3: Start KeyDB

```bash
# Start KeyDB service
docker-compose up -d keydb

# Verify KeyDB is running
docker-compose logs keydb

# Check health
docker exec hean-keydb keydb-cli ping
# Should return: PONG
```

### Step 4: Migrate Data (Optional)

If you have existing Redis data to migrate:

```bash
# Run migration script
python scripts/migrate_redis_to_keydb.py \
  --redis-host old-redis-host \
  --keydb-host localhost \
  --redis-port 6379 \
  --keydb-port 6379

# Verification is included by default
# Use --no-verify to skip
```

**Migration features**:
- ✅ Migrates all Redis data types (strings, hashes, lists, sets, sorted sets)
- ✅ Preserves TTLs (expiration times)
- ✅ Progress tracking with progress bar
- ✅ Automatic verification
- ✅ Error handling and reporting

### Step 5: Update Application Configuration

Update your `.env` or `backend.env` to point to KeyDB:

```bash
# Change from:
# REDIS_URL=redis://redis:6379

# To:
REDIS_URL=redis://keydb:6379
```

### Step 6: Start All Services

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Verify API health
curl http://localhost:8000/health
```

## 📊 Performance Benchmarking

Run the benchmark script to verify KeyDB performance:

```bash
# Benchmark KeyDB
python scripts/benchmark_keydb.py --host localhost --port 6379 --num-ops 10000
```

**Expected results** (on 4-core system):
```
Operation       Throughput
---------------------------------
PIPELINE        100,000+ ops/sec
SET              50,000+ ops/sec
GET              60,000+ ops/sec
HSET             45,000+ ops/sec
LPUSH            40,000+ ops/sec
SADD             40,000+ ops/sec
ZADD             35,000+ ops/sec
```

Compare with Redis baseline to see improvement (typically 2-5x).

## 🔧 Configuration Options

### KeyDB Configuration (`keydb.conf`)

Key settings for HEAN-META:

```conf
# Multi-threading (adjust based on CPU cores)
server-threads 4
server-thread-affinity true

# Memory limit
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence (RDB + AOF)
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

### Tuning for Your Hardware

**2 CPU cores**:
```conf
server-threads 2
maxmemory 1gb
```

**8+ CPU cores**:
```conf
server-threads 8
maxmemory 4gb
```

**High-throughput trading**:
```conf
server-threads 4
maxmemory 4gb
appendfsync no  # Better performance, slightly less durability
```

## 🐛 Troubleshooting

### KeyDB Won't Start

```bash
# Check logs
docker-compose logs keydb

# Common issues:
# 1. Port 6379 already in use
docker-compose down
sudo lsof -i :6379  # Kill any process using port 6379

# 2. Permission issues with config file
chmod 644 keydb.conf

# 3. Invalid configuration
docker exec hean-keydb keydb-server --test-memory 1024
```

### Migration Errors

```bash
# Check connectivity
docker exec hean-keydb keydb-cli ping

# Check Redis source
redis-cli -h old-redis-host ping

# Run migration with verbose output
python scripts/migrate_redis_to_keydb.py --redis-host old-redis --keydb-host localhost 2>&1 | tee migration.log
```

### Application Can't Connect

```bash
# Verify KeyDB is accessible
docker exec hean-api ping -c 1 keydb

# Check environment variables
docker exec hean-api env | grep REDIS

# Test connection from API container
docker exec hean-api python -c "import redis; r=redis.Redis(host='keydb', port=6379); print(r.ping())"
```

## 🔐 Security Considerations

### Enable Password Authentication

Edit `keydb.conf`:

```conf
# Set a strong password
requirepass your_strong_password_here
```

Update application configuration:

```bash
REDIS_URL=redis://:your_strong_password_here@keydb:6379
```

Restart services:

```bash
docker-compose restart keydb api
```

### Network Security

KeyDB is only accessible within Docker network by default. To expose externally:

```yaml
# docker-compose.yml
ports:
  - "127.0.0.1:6379:6379"  # Only localhost
  # or
  - "6379:6379"  # All interfaces (NOT RECOMMENDED without password)
```

## 📈 Monitoring

### KeyDB Metrics

```bash
# Get server info
docker exec hean-keydb keydb-cli INFO

# Monitor in real-time
docker exec hean-keydb keydb-cli --stat

# Check memory usage
docker exec hean-keydb keydb-cli INFO memory | grep used_memory_human

# Check connected clients
docker exec hean-keydb keydb-cli CLIENT LIST
```

### Integration with Prometheus (Coming in Part 3)

KeyDB metrics can be exported via:
- [Redis Exporter](https://github.com/oliver006/redis_exporter)
- Native KeyDB metrics endpoint

## 🔄 Rollback to Redis

If you need to rollback to Redis:

```bash
# Stop KeyDB
docker-compose down

# Restore original docker-compose.yml (with redis service)
git checkout docker-compose.yml

# Start Redis
docker-compose up -d redis
```

## ✅ Verification Checklist

After migration, verify:

- [ ] KeyDB service is running: `docker-compose ps keydb`
- [ ] Health check passes: `docker exec hean-keydb keydb-cli ping`
- [ ] API connects to KeyDB: `curl http://localhost:8000/health`
- [ ] Data is accessible (if migrated)
- [ ] Application logs show no Redis connection errors
- [ ] Performance meets expectations (run benchmark)

## 📚 Additional Resources

- [KeyDB Documentation](https://docs.keydb.dev/)
- [KeyDB GitHub](https://github.com/Snapchat/KeyDB)
- [Redis to KeyDB Migration Guide](https://docs.keydb.dev/docs/migration/)
- [Performance Benchmarks](https://docs.keydb.dev/docs/benchmarks/)

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs keydb`
2. Verify configuration: `docker exec hean-keydb keydb-server --test-memory 1024`
3. Test connectivity: `docker exec hean-api ping keydb`
4. Review this guide's troubleshooting section
5. Check KeyDB documentation: https://docs.keydb.dev/

---

**Next Steps**: Once KeyDB migration is complete, proceed to:
- Part 2: C++ Order Execution Engine
- Part 3: Rust Market Data Service
- Part 4: Monitoring Stack (Prometheus + Grafana)
