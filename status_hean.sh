#!/bin/bash

# ═══════════════════════════════════════════════════════════════════
# HEAN Status Script - Проверка статуса сервисов
# ═══════════════════════════════════════════════════════════════════

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "════════════════════════════════════════════════════════════════════"
echo "📊 HEAN Status Report"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker не запущен${NC}"
    exit 1
fi

# Check containers
echo "Контейнеры:"
docker compose ps

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check service health
services=(
    "hean-redis:Redis:6379"
    "hean-api:Backend API:8000"
    "hean-symbiont-testnet:Trading Bot:-"
    "hean-collector:Collector:-"
    "hean-physics:Physics:-"
    "hean-brain:Brain:-"
    "hean-risk-svc:Risk:-"
)

healthy=0
unhealthy=0

for service_info in "${services[@]}"; do
    IFS=':' read -r container name port <<< "$service_info"

    if docker ps --filter "name=$container" --filter "status=running" --format '{{.Names}}' | grep -q "$container"; then
        status="${GREEN}✓ Running${NC}"
        ((healthy++))

        # Check port if specified
        if [ "$port" != "-" ]; then
            if netstat -an 2>/dev/null | grep -q ":$port.*LISTEN" || lsof -i ":$port" > /dev/null 2>&1; then
                port_status=" (port $port ✓)"
            else
                port_status=" (port $port ?)"
            fi
        else
            port_status=""
        fi

        echo -e "$status $name$port_status"
    else
        echo -e "${RED}✗ Stopped${NC} $name"
        ((unhealthy++))
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Итого: $healthy запущено, $unhealthy остановлено"
echo "════════════════════════════════════════════════════════════════════"
