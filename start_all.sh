#!/bin/bash
# HEAN Full Stack Launcher
# Запускает backend + открывает iOS проект

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  HEAN Full Stack Launcher"
echo "=========================================="

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 1. Проверка .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[!] .env файл не найден, копирую из .env.example${NC}"
    cp .env.example .env
    echo -e "${RED}[!] Отредактируйте .env и добавьте BYBIT_API_KEY и BYBIT_API_SECRET${NC}"
fi

# 2. Проверка Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker не установлен${NC}"
    exit 1
fi

# 3. Запуск backend через Docker
echo -e "\n${GREEN}[1/3] Запуск Backend (Docker)...${NC}"
docker-compose down 2>/dev/null || true
docker-compose up -d --build

# Ждём пока API станет доступен
echo -e "${YELLOW}Ожидание запуска API...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API запущен на http://localhost:8000${NC}"
        break
    fi
    sleep 1
    echo -n "."
done

# 4. Проверка здоровья
echo -e "\n${GREEN}[2/3] Проверка сервисов...${NC}"
echo "API Health:"
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool 2>/dev/null || echo "API не отвечает"

echo -e "\nTelemetry:"
curl -s http://localhost:8000/api/v1/telemetry/ping | python3 -m json.tool 2>/dev/null || echo "Telemetry не отвечает"

# 5. Открываем Xcode
echo -e "\n${GREEN}[3/3] Открытие iOS проекта в Xcode...${NC}"
if [ -d "ios/HEAN.xcodeproj" ]; then
    open ios/HEAN.xcodeproj
    echo -e "${GREEN}✓ Xcode открыт${NC}"
else
    echo -e "${RED}[!] iOS проект не найден в ios/HEAN.xcodeproj${NC}"
fi

echo -e "\n=========================================="
echo -e "${GREEN}  Готово!${NC}"
echo "=========================================="
echo ""
echo "Backend API:     http://localhost:8000"
echo "API Docs:        http://localhost:8000/docs"
echo "WebSocket:       ws://localhost:8000/ws"
echo ""
echo "В Xcode:"
echo "  1. Выберите симулятор iPhone"
echo "  2. Cmd+Shift+K (Clean)"
echo "  3. Cmd+R (Run)"
echo ""
echo "Логи backend:    docker-compose logs -f"
echo "Остановить:      docker-compose down"
echo "=========================================="
