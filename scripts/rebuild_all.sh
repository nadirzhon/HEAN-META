#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[HEAN] Stopping and removing existing containers and volumes..."
docker compose down -v || docker-compose down -v || true

echo "[HEAN] Pruning unused Docker resources..."
docker system prune -af || true
docker volume prune -f || true

echo "[HEAN] Building images without cache..."
docker compose build --no-cache || docker-compose build --no-cache

echo "[HEAN] Starting stack in detached mode..."
docker compose up -d || docker-compose up -d

echo "[HEAN] Done. Stack is rebuilding/starting."

