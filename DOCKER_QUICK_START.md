# HEAN Docker - Quick Start Guide

Get HEAN running in Docker in 5 minutes.

---

## 🚀 Quick Start (Development)

### 1. Prerequisites

```bash
# Check Docker is installed
docker --version
docker-compose --version

# Verify at least 4GB RAM available
docker info | grep "Total Memory"
```

### 2. Clone & Configure

```bash
# Navigate to project
cd HEAN

# Copy environment template
cp backend.env.example backend.env

# Edit with your API keys (required)
nano backend.env
```

**Minimum required:**
```env
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
OPENAI_API_KEY=sk-...
```

### 3. Start Services

```bash
# One command to rule them all
make dev
```

That's it! 🎉

### 4. Access

- **UI:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

---

## 🏭 Quick Start (Production)

### 1. Build Optimized Images

```bash
make prod-build
```

### 2. Deploy

```bash
make prod-up
```

### 3. Verify

```bash
# Check services are running
make stats

# View logs
make prod-logs
```

### 4. Access

- **UI:** http://localhost
- **API:** http://localhost/api

---

## ☸️ Quick Start (Kubernetes)

### 1. Create Secrets

```bash
kubectl create secret generic hean-secrets \
  --from-literal=BYBIT_API_KEY='your-key' \
  --from-literal=OPENAI_API_KEY='sk-...' \
  -n hean-production
```

### 2. Deploy

```bash
./scripts/k8s-deploy.sh deploy
```

### 3. Check Status

```bash
./scripts/k8s-deploy.sh status
```

---

## 🛠️ Useful Commands

```bash
# View logs
make docker-logs

# Restart services
make docker-restart

# Stop everything
make docker-down

# Clean up
make docker-clean

# Backup Redis
make redis-backup

# Security scan
make scan
```

---

## 🆘 Troubleshooting

### Container won't start?

```bash
# Check logs
docker-compose logs api

# Verify environment variables
cat backend.env
```

### Port already in use?

```bash
# Find what's using the port
lsof -i :8000
lsof -i :3000

# Kill the process or change ports in docker-compose.yml
```

### Out of memory?

```bash
# Check usage
docker stats

# Increase Docker memory in Docker Desktop settings
```

### Need help?

```bash
make help
```

---

## 📚 Full Documentation

See [DOCKER_DEPLOYMENT_GUIDE.md](./DOCKER_DEPLOYMENT_GUIDE.md) for complete documentation.

---

**Ready to trade!** 🚀📈
