#!/bin/bash

# ═══════════════════════════════════════════════════════════════════
# HEAN Complete Startup Script
# ═══════════════════════════════════════════════════════════════════
# Автоматически запускает все сервисы проекта HEAN:
# - Backend API (FastAPI)
# - Redis
# - Symbiont Trading Bot
# - Collector, Physics, Brain, Risk сервисы
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "🚀 HEAN Complete Startup Script"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 1: Check required files
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[1/7]${NC} Проверка файлов конфигурации..."

required_files=(
    ".env"
    ".env.symbiont"
    "backend.env"
    "docker-compose.yml"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    else
        echo -e "  ${GREEN}✓${NC} $file"
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "${RED}✗ Отсутствуют файлы:${NC}"
    for file in "${missing_files[@]}"; do
        echo -e "  ${RED}✗${NC} $file"
    done
    echo ""
    echo "Создайте недостающие файлы из примеров:"
    echo "  cp .env.example .env"
    echo "  cp .env.symbiont.example .env.symbiont"
    echo "  cp backend.env.example backend.env"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 2: Check API keys
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[2/7]${NC} Проверка API ключей..."

check_api_key() {
    local file=$1
    local key_name=$2
    local value=$(grep "^${key_name}=" "$file" 2>/dev/null | cut -d'=' -f2)

    if [ -z "$value" ] || [ "$value" == "your-"* ] || [ "$value" == "sk-"* ]; then
        echo -e "  ${YELLOW}⚠${NC} $key_name не настроен в $file"
        return 1
    else
        echo -e "  ${GREEN}✓${NC} $key_name настроен"
        return 0
    fi
}

warnings=0

# Check Bybit keys
if check_api_key "backend.env" "BYBIT_API_KEY"; then
    :
else
    ((warnings++))
fi

if check_api_key "backend.env" "BYBIT_API_SECRET"; then
    :
else
    ((warnings++))
fi

if [ $warnings -gt 0 ]; then
    echo -e "${YELLOW}⚠ Некоторые API ключи не настроены${NC}"
    echo "Отредактируйте backend.env и добавьте ваши ключи от Bybit"
    read -p "Продолжить? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 3: Stop existing containers
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[3/7]${NC} Остановка существующих контейнеров..."

docker compose down 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Контейнеры остановлены"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 4: Build Docker images
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[4/7]${NC} Сборка Docker образов..."

if docker compose build --no-cache; then
    echo -e "  ${GREEN}✓${NC} Образы собраны успешно"
else
    echo -e "  ${RED}✗${NC} Ошибка сборки образов"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 5: Start services
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[5/7]${NC} Запуск сервисов..."

# Start Redis first
echo "  Запуск Redis..."
docker compose up -d redis
sleep 5

# Check Redis health
if docker compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Redis запущен"
else
    echo -e "  ${RED}✗${NC} Redis не отвечает"
    exit 1
fi

# Start all other services
echo "  Запуск остальных сервисов..."
docker compose up -d

echo -e "  ${GREEN}✓${NC} Все сервисы запущены"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 6: Wait for services to be healthy
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[6/7]${NC} Ожидание готовности сервисов..."

sleep 10

# Check service status
services=(
    "hean-redis:Redis"
    "hean-api:Backend API"
    "hean-symbiont-testnet:Symbiont Trading Bot"
    "hean-collector:Market Data Collector"
    "hean-physics:Physics Engine"
    "hean-brain:AI Brain"
    "hean-risk-svc:Risk Management"
)

healthy=0
unhealthy=0

for service_info in "${services[@]}"; do
    IFS=':' read -r container name <<< "$service_info"

    if docker ps --filter "name=$container" --filter "status=running" --format '{{.Names}}' | grep -q "$container"; then
        echo -e "  ${GREEN}✓${NC} $name"
        ((healthy++))
    else
        echo -e "  ${YELLOW}⚠${NC} $name (не запущен)"
        ((unhealthy++))
    fi
done

echo ""
echo "Статус: $healthy запущено, $unhealthy проблем"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Step 7: Show access information
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}[7/7]${NC} Информация о доступе..."
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ HEAN запущен успешно!${NC}"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📡 Backend API:     http://localhost:8000"
echo "📊 API Docs:        http://localhost:8000/docs"
echo "🔴 Redis:           localhost:6379"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "📋 Полезные команды:"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Просмотр логов всех сервисов:"
echo "  docker compose logs -f"
echo ""
echo "Просмотр логов конкретного сервиса:"
echo "  docker compose logs -f symbiont-testnet"
echo "  docker compose logs -f api"
echo ""
echo "Остановка всех сервисов:"
echo "  docker compose down"
echo ""
echo "Перезапуск:"
echo "  docker compose restart"
echo ""
echo "Статус контейнеров:"
echo "  docker compose ps"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Ask if user wants to see logs
read -p "Показать логи в реальном времени? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Нажмите Ctrl+C для выхода из логов"
    sleep 2
    docker compose logs -f
fi
