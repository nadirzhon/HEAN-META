# Docker Quick Reference - HEAN System

## Quick Start Commands

### Development
```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Clean rebuild (remove volumes)
docker-compose down -v
docker-compose up -d --build
```

### Production
```bash
# Build production images
make prod-build

# Start production stack
make prod-up

# Start with monitoring
make prod-with-monitoring

# View logs
make prod-logs

# Stop production
make prod-down
```

### Monitoring
```bash
# Start monitoring stack
make monitoring-up

# Access dashboards
# Prometheus: http://localhost:9091
# Grafana: http://localhost:3001 (admin/admin)

# Stop monitoring
make monitoring-down
```

---

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| UI | 3000 | http://localhost:3000 |
| Redis | 6379 | redis://localhost:6379 |
| Prometheus | 9091 | http://localhost:9091 |
| Grafana | 3001 | http://localhost:3001 |

---

## Health Check Endpoints

### API
```bash
curl http://localhost:8000/health
```

### UI
```bash
curl http://localhost:3000
```

### Redis
```bash
docker exec hean-redis redis-cli ping
```

---

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs api

# Check health status
docker inspect hean-api | grep -A 10 "Health"

# Restart specific service
docker-compose restart api
```

### Permission Issues
```bash
# Verify non-root user
docker run --rm hean-api:latest id
# Expected: uid=1000(hean) gid=1000(hean)

# Fix volume permissions
sudo chown -R 1000:1000 ./logs ./data
```

### Build Failures
```bash
# Clean build cache
docker builder prune -a

# Rebuild from scratch
docker-compose build --no-cache api
```

### Redis Connection Issues
```bash
# Check Redis is running
docker ps | grep redis

# Test connection
docker exec hean-redis redis-cli ping

# Check logs
docker-compose logs redis
```

---

## Development Workflow

### 1. Code Changes
```bash
# Edit files in src/
vim src/hean/api/main.py

# Rebuild API container
docker-compose up -d --build api

# View logs
docker-compose logs -f api
```

### 2. Testing
```bash
# Run tests inside container
docker-compose exec api pytest

# Run specific test
docker-compose exec api pytest tests/test_api.py -v

# Run smoke test
./scripts/smoke_test.sh
```

### 3. Debugging
```bash
# Shell into API container
docker-compose exec api bash

# Check Python environment
docker-compose exec api python --version
docker-compose exec api pip list

# Check environment variables
docker-compose exec api env | grep BYBIT
```

---

## Resource Management

### View Resource Usage
```bash
docker stats
```

### Set Resource Limits
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '0.5'
      memory: 512M
```

### Clean Up
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

---

## Security

### Non-root User Verification
```bash
# Check user inside container
docker run --rm hean-api:latest id
# Expected: uid=1000(hean) gid=1000(hean) groups=1000(hean)

# Check file ownership
docker-compose exec api ls -la /app
# Expected: drwxr-xr-x hean hean
```

### Secret Management
```bash
# Never commit .env files!
# Use .env.example as template

cp .env.example .env
# Edit .env with your credentials
vim .env

# Verify .env is in .gitignore
grep "^\.env$" .gitignore
```

### Vulnerability Scanning
```bash
# Install Trivy
brew install trivy  # macOS

# Scan images
trivy image hean-api:latest
trivy image hean-ui:latest
```

---

## Backup and Restore

### Redis Backup
```bash
# Create backup
docker-compose exec redis redis-cli BGSAVE
docker cp hean-redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d-%H%M%S).rdb

# Restore backup
docker cp ./backups/redis-backup.rdb hean-redis:/data/dump.rdb
docker-compose restart redis
```

### Volume Backup
```bash
# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz ./logs

# Backup data
tar -czf data-backup-$(date +%Y%m%d).tar.gz ./data
```

---

## Performance Tuning

### Build Cache Optimization
```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker-compose build

# Parallel builds
docker-compose build --parallel
```

### Multi-replica API
Edit `docker-compose.production.yml`:
```yaml
api:
  deploy:
    replicas: 3
```

### Redis Optimization
Edit `docker-compose.yml`:
```yaml
redis:
  command: >
    redis-server
    --appendonly yes
    --maxmemory 512mb
    --maxmemory-policy allkeys-lru
    --save 60 1000
```

---

## Logging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Follow with timestamps
docker-compose logs -f -t api
```

### Log Rotation
Already configured in `docker-compose.yml`:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
    compress: "true"
```

---

## Network Inspection

### List Networks
```bash
docker network ls
```

### Inspect Network
```bash
docker network inspect hean-network
```

### Test Connectivity
```bash
# From API to Redis
docker-compose exec api ping redis

# Check DNS resolution
docker-compose exec api nslookup redis
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build images
        run: docker-compose build

      - name: Run smoke test
        run: |
          docker-compose up -d
          sleep 30
          ./scripts/smoke_test.sh
          docker-compose down
```

---

## Makefile Targets

```bash
make install              # Install Python dependencies
make test                # Run tests
make lint                # Run linter
make run                 # Run CLI

make docker-build        # Build Docker images
make docker-up           # Start containers
make docker-down         # Stop containers
make docker-logs         # View logs
make docker-clean        # Remove containers and volumes

make prod-build          # Build production images
make prod-up             # Start production
make prod-down           # Stop production
make prod-with-monitoring # Start with monitoring

make monitoring-up       # Start monitoring stack
make monitoring-down     # Stop monitoring
make monitoring-logs     # View monitoring logs

make redis-cli           # Connect to Redis CLI
make shell-api           # Open shell in API container
make stats               # Show resource usage
```

---

## Environment Variables

### Required
```bash
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=true
```

### Optional
```bash
ANTHROPIC_API_KEY=your_anthropic_key
INITIAL_CAPITAL=300
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379/0
```

### Check Current Values
```bash
docker-compose exec api env | grep BYBIT
docker-compose exec api env | grep REDIS
```

---

## Common Issues and Solutions

### Issue: Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Issue: Container Exits Immediately
```bash
# Check exit code
docker inspect hean-api | grep ExitCode

# View full logs
docker-compose logs api

# Run interactively for debugging
docker-compose run --rm api bash
```

### Issue: Slow Build Times
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Use parallel builds
docker-compose build --parallel

# Check .dockerignore excludes unnecessary files
cat .dockerignore
```

### Issue: Out of Disk Space
```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a --volumes

# Remove specific images
docker rmi $(docker images -q -f dangling=true)
```

---

## Production Checklist

Before deploying to production:

- [ ] Environment variables set correctly
- [ ] Secrets not committed to git
- [ ] Healthchecks pass: `./scripts/smoke_test.sh`
- [ ] Resource limits configured
- [ ] Logging configured
- [ ] Backup strategy in place
- [ ] Monitoring stack running
- [ ] Security scan passed: `trivy image hean-api:latest`
- [ ] Non-root user verified
- [ ] Volume permissions correct

---

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Multi-stage Build Guide](https://docs.docker.com/develop/develop-images/multistage-build/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

**Last Updated**: 2026-02-08
**HEAN Version**: 2.0
