# HEAN Docker & Kubernetes Deployment Guide

Complete guide for deploying HEAN trading system using Docker and Kubernetes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Local Development](#local-development)
4. [Production Deployment](#production-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring & Logging](#monitoring--logging)
7. [Troubleshooting](#troubleshooting)
8. [Security Best Practices](#security-best-practices)
9. [Performance Optimization](#performance-optimization)
10. [Production Checklist](#production-checklist)

---

## Quick Start

### Prerequisites

- Docker 24.0+ with BuildKit enabled
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Run Locally (Development)

```bash
# Start all services
make dev

# Access the application
# UI: http://localhost:3000
# API: http://localhost:8000/docs
```

### Run Production Build

```bash
# Build optimized images
make prod-build

# Start production services
make prod-up

# Check status
make stats
```

---

## Architecture Overview

### Services

```
┌─────────────────────────────────────────────┐
│              Load Balancer/Ingress          │
└────────────┬─────────────────┬──────────────┘
             │                 │
    ┌────────▼────────┐  ┌────▼────────────┐
    │   UI (Nginx)    │  │  API (FastAPI)  │
    │   Port: 80/443  │  │   Port: 8000    │
    │   Size: ~25MB   │  │   Size: ~350MB  │
    └─────────────────┘  └────┬─────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Redis (Cache)    │
                    │   Port: 6379       │
                    │   Persistence: AOF │
                    └────────────────────┘
```

### Image Sizes

| Component | Size | Notes |
|-----------|------|-------|
| Python API | ~350MB | Multi-stage build |
| React UI | ~25MB | Nginx Alpine |
| Rust Gateway | ~15MB | Distroless |
| Redis | ~35MB | Alpine |

---

## Local Development

### Development Environment

```bash
# Start with hot reload
make dev

# View logs
make docker-logs

# Open shell in API container
make shell-api

# Run tests
make test
```

### Development Docker Compose

The `docker-compose.yml` includes:
- Hot reload for both API and UI
- Volume mounts for source code
- Development ports exposed
- Debug logging enabled

### Environment Variables

Create `backend.env`:

```env
# API Keys
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=your_key

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Trading Configuration
FUNDING_HARVESTER_ENABLED=true
BASIS_ARBITRAGE_ENABLED=true
IMPULSE_ENGINE_ENABLED=true

# Logging
LOG_LEVEL=INFO
```

---

## Production Deployment

### 1. Build Production Images

```bash
# Build all services
make prod-build

# Or build individually
docker-compose -f docker-compose.production.yml build api
docker-compose -f docker-compose.production.yml build ui
```

### 2. Configure Environment

Update `backend.env` with production values:

```env
LOG_LEVEL=WARNING
# Add production API keys
# Enable only required strategies
```

### 3. Deploy

```bash
# Deploy all services
./scripts/docker-deploy.sh production deploy

# Or use Makefile
make prod-up
```

### 4. Verify Health

```bash
# Check API health
curl http://localhost/api/health

# Check UI
curl http://localhost/

# View logs
make prod-logs
```

### Production Features

- **3 API replicas** for load balancing
- **Auto-restart** on failure
- **Resource limits** (4CPU, 4GB RAM per API)
- **Health checks** every 30s
- **Redis persistence** with AOF + RDB
- **Log rotation** (50MB, 5 files)
- **Network isolation** (frontend/backend separation)

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Ingress controller (nginx)
- cert-manager for TLS (optional)

### 1. Create Secrets

```bash
kubectl create secret generic hean-secrets \
  --from-literal=BYBIT_API_KEY='your-key' \
  --from-literal=BYBIT_API_SECRET='your-secret' \
  --from-literal=OPENAI_API_KEY='sk-...' \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-...' \
  --from-literal=GOOGLE_API_KEY='your-key' \
  -n hean-production
```

### 2. Deploy

```bash
# Deploy all resources
./scripts/k8s-deploy.sh deploy

# Or use kubectl
kubectl apply -f k8s/
```

### 3. Check Status

```bash
# Use deployment script
./scripts/k8s-deploy.sh status

# Or kubectl
kubectl get all -n hean-production
```

### 4. Access Application

```bash
# Get ingress URL
kubectl get ingress -n hean-production

# Port forward for testing
kubectl port-forward -n hean-production svc/hean-ui-service 3000:80
```

### Kubernetes Features

- **Horizontal Pod Autoscaler** (3-10 replicas)
  - CPU target: 70%
  - Memory target: 80%
- **Rolling updates** with zero downtime
- **Health probes** (liveness, readiness, startup)
- **Pod anti-affinity** for HA
- **Resource requests/limits**
- **Persistent volumes** for Redis
- **TLS/SSL** via cert-manager

### Scaling

```bash
# Scale API manually
./scripts/k8s-deploy.sh scale api 5

# HPA will auto-scale based on load
kubectl get hpa -n hean-production
```

### Updates & Rollbacks

```bash
# Update deployment
./scripts/k8s-deploy.sh update

# Rollback if needed
./scripts/k8s-deploy.sh rollback

# Check rollout status
kubectl rollout status deployment/hean-api -n hean-production
```

---

## Monitoring & Logging

### Enable Monitoring Stack

```bash
# Start Prometheus + Grafana
make prod-with-monitoring

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

### Prometheus Metrics

The API exposes metrics at `/metrics`:

- HTTP request rate and latency
- Trading operations counters
- Redis connection pool stats
- System resource usage
- Custom business metrics

### Grafana Dashboards

Pre-configured dashboards:
- API performance overview
- Trading system metrics
- Resource utilization
- Redis statistics

### Log Management

```bash
# View logs
make docker-logs

# Production logs
docker-compose -f docker-compose.production.yml logs -f api

# Kubernetes logs
./scripts/k8s-deploy.sh logs api

# Filter logs
kubectl logs -n hean-production -l component=api --tail=100 | grep ERROR
```

### Log Rotation

Automatic log rotation configured:
- Max size: 50MB
- Max files: 5
- Compression enabled

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Check health
docker ps
docker inspect hean-api

# Common issues:
# - Missing environment variables
# - Port conflicts
# - Out of memory
```

### Health Check Failing

```bash
# Test health endpoint
curl -v http://localhost:8000/health

# Check API logs
docker-compose logs api

# Restart container
docker-compose restart api
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Increase memory limits in docker-compose.production.yml
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G
```

### Redis Connection Issues

```bash
# Connect to Redis CLI
make redis-cli

# Check Redis logs
docker-compose logs redis

# Test connection
redis-cli -h localhost -p 6379 ping
```

### Slow Build Times

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Use build cache
docker-compose build --pull

# Check .dockerignore
# Ensure large files are excluded
```

### Kubernetes Pod Crashes

```bash
# Check pod events
kubectl describe pod <pod-name> -n hean-production

# View logs
kubectl logs <pod-name> -n hean-production

# Check resource limits
kubectl top pods -n hean-production
```

---

## Security Best Practices

### Container Security

✅ **Implemented:**
- Non-root users in all containers
- Read-only root filesystems (UI)
- Minimal base images (Alpine, Distroless)
- No secrets in images
- Security scanning with Trivy
- Regular updates

### Network Security

```yaml
# Use network isolation
networks:
  frontend:
    driver: bridge
  backend:
    internal: true  # No external access
```

### Secrets Management

**Never commit secrets!**

```bash
# Use environment variables
source .env

# Or use secret management
# - Docker secrets
# - Kubernetes secrets
# - HashiCorp Vault
# - AWS Secrets Manager
```

### TLS/SSL

For Kubernetes:

```yaml
# Enable TLS in ingress
tls:
  - hosts:
    - hean.yourdomain.com
    secretName: hean-tls
```

Use cert-manager for automatic certificate management.

### Security Scanning

```bash
# Scan images for vulnerabilities
make scan

# Or manually
trivy image hean-api:latest
trivy image hean-ui:latest
```

---

## Performance Optimization

### Build Optimization

- **Multi-stage builds** reduce image size by 71-97%
- **Layer caching** speeds up rebuilds
- **.dockerignore** excludes unnecessary files
- **BuildKit** enables advanced features

### Runtime Optimization

**API:**
- uvloop for faster async I/O
- Multiple workers (4 default)
- Resource limits prevent OOM

**UI:**
- Gzip compression in Nginx
- Static asset caching (1 year)
- Tree-shaking removes unused code

**Redis:**
- AOF + RDB persistence
- LRU eviction policy
- Optimized memory limit

### Monitoring Performance

```bash
# Container stats
docker stats

# API metrics
curl http://localhost:8000/metrics

# Grafana dashboards
# Monitor: CPU, memory, latency, throughput
```

---

## Production Checklist

Before deploying to production:

### Configuration
- [ ] Update all API keys in secrets
- [ ] Set LOG_LEVEL=WARNING
- [ ] Configure proper resource limits
- [ ] Set up log rotation
- [ ] Enable monitoring stack

### Security
- [ ] No secrets in source code
- [ ] TLS/SSL configured
- [ ] Firewall rules set
- [ ] Security scanning passed
- [ ] Non-root containers verified

### Infrastructure
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] Backup strategy defined
- [ ] Disaster recovery plan
- [ ] Auto-scaling configured

### Monitoring
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards configured
- [ ] Alerts set up
- [ ] Log aggregation working
- [ ] Health checks passing

### Testing
- [ ] Smoke tests passing
- [ ] Load testing completed
- [ ] Rollback tested
- [ ] Backup/restore tested
- [ ] Monitoring verified

### Documentation
- [ ] Deployment documented
- [ ] Runbook created
- [ ] On-call procedures defined
- [ ] Team trained

---

## Commands Reference

### Docker Development

```bash
make dev              # Start dev environment
make dev-build        # Build dev images
make docker-logs      # View logs
make docker-restart   # Restart containers
make docker-clean     # Clean up
```

### Docker Production

```bash
make prod-build       # Build production images
make prod-up          # Start production
make prod-down        # Stop production
make prod-logs        # View production logs
make prod-with-monitoring  # Start with monitoring
```

### Kubernetes

```bash
make k8s-deploy       # Deploy to K8s
make k8s-status       # Check status
make k8s-logs         # View logs (specify component)

# Using scripts
./scripts/k8s-deploy.sh deploy
./scripts/k8s-deploy.sh status
./scripts/k8s-deploy.sh logs api
./scripts/k8s-deploy.sh scale api 5
./scripts/k8s-deploy.sh rollback
```

### Database

```bash
make redis-cli        # Connect to Redis
make redis-backup     # Backup Redis data
```

### Utilities

```bash
make shell-api        # Shell in API container
make stats            # Resource usage
make scan             # Security scan
```

---

## Support

For issues:
1. Check logs: `make docker-logs`
2. Review this guide
3. Check GitHub issues
4. Contact DevOps team

---

**Last Updated:** 2026-01-27
**Version:** 1.0.0
