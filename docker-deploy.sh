#!/bin/bash
# HEAN Docker Deployment Script - Production Ready
# Полная проверка и запуск микросервисной архитектуры

set -e  # Остановка при первой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Логирование
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка Docker
check_docker() {
    log_info "Проверка Docker..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker не установлен!"
        log_info "Установите Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon не запущен!"
        log_info "Запустите Docker Desktop или systemctl start docker"
        exit 1
    fi

    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose не установлен!"
        exit 1
    fi

    log_success "Docker OK ($(docker --version))"
}

# Проверка файлов конфигурации
check_config_files() {
    log_info "Проверка конфигурационных файлов..."

    local required_files=(
        "docker-compose.yml"
        "backend.env"
        ".env.symbiont"
        "api/Dockerfile"
        "apps/ui/Dockerfile"
        "Dockerfile.testnet"
    )

    local missing_files=()

    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_files[@]} -ne 0 ]; then
        log_error "Отсутствуют файлы:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi

    log_success "Все конфигурационные файлы присутствуют"
}

# Проверка переменных окружения
check_env_variables() {
    log_info "Проверка переменных окружения..."

    # Проверка backend.env
    if ! grep -q "BYBIT_API_KEY" backend.env; then
        log_warning "BYBIT_API_KEY не установлен в backend.env"
    fi

    if ! grep -q "BYBIT_API_SECRET" backend.env; then
        log_warning "BYBIT_API_SECRET не установлен в backend.env"
    fi

    # Проверка .env.symbiont
    if ! grep -q "BYBIT_API_KEY" .env.symbiont; then
        log_warning "BYBIT_API_KEY не установлен в .env.symbiont"
    fi

    log_success "Переменные окружения проверены"
}

# Проверка портов
check_ports() {
    log_info "Проверка доступности портов..."

    local ports=(8000 3000 6379)
    local busy_ports=()

    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            busy_ports+=("$port")
        fi
    done

    if [ ${#busy_ports[@]} -ne 0 ]; then
        log_warning "Порты уже используются: ${busy_ports[*]}"
        log_info "Остановка существующих контейнеров..."
        docker compose down 2>/dev/null || true
    fi

    log_success "Порты проверены"
}

# Очистка старых контейнеров и образов
cleanup() {
    log_info "Очистка старых контейнеров..."

    # Остановка и удаление контейнеров
    docker compose down --remove-orphans 2>/dev/null || true

    # Удаление dangling images
    docker image prune -f >/dev/null 2>&1 || true

    log_success "Очистка завершена"
}

# Сборка образов
build_images() {
    log_info "Сборка Docker образов..."
    log_warning "Это может занять 5-10 минут при первой сборке..."

    # Сборка с выводом прогресса
    if ! docker compose build --progress=plain 2>&1 | tee build.log; then
        log_error "Ошибка при сборке образов!"
        log_info "Проверьте файл build.log для деталей"
        exit 1
    fi

    log_success "Образы успешно собраны"
}

# Запуск контейнеров
start_containers() {
    log_info "Запуск контейнеров..."

    # Запуск в фоновом режиме
    if ! docker compose up -d; then
        log_error "Ошибка при запуске контейнеров!"
        log_info "Проверка логов:"
        docker compose logs
        exit 1
    fi

    log_success "Контейнеры запущены"
}

# Проверка здоровья сервисов
check_health() {
    log_info "Проверка здоровья сервисов (ожидание запуска)..."

    local max_attempts=30
    local attempt=1

    # Ожидание Redis
    while [ $attempt -le $max_attempts ]; do
        if docker compose exec -T redis redis-cli ping &> /dev/null; then
            log_success "Redis: OK"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_error "Redis: не запустился"
            return 1
        fi

        sleep 1
        ((attempt++))
    done

    # Ожидание API (с увеличенным таймаутом для первого запуска)
    log_info "Ожидание запуска API (может занять до 60 секунд)..."
    attempt=1
    max_attempts=60

    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log_success "API: OK (http://localhost:8000)"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_error "API: не запустился"
            log_info "Проверка логов API:"
            docker compose logs api | tail -50
            return 1
        fi

        sleep 1
        ((attempt++))
    done

    # Проверка UI
    attempt=1
    max_attempts=30

    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:3000 > /dev/null 2>&1; then
            log_success "UI: OK (http://localhost:3000)"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_error "UI: не запустился"
            log_info "Проверка логов UI:"
            docker compose logs ui | tail -50
            return 1
        fi

        sleep 1
        ((attempt++))
    done

    return 0
}

# Тестирование API endpoints
test_api() {
    log_info "Тестирование API endpoints..."

    # Test /health
    if curl -sf http://localhost:8000/health > /dev/null; then
        log_success "GET /health - OK"
    else
        log_warning "GET /health - FAILED"
    fi

    # Test /telemetry/summary (если доступен)
    if curl -sf http://localhost:8000/telemetry/summary > /dev/null; then
        log_success "GET /telemetry/summary - OK"
    else
        log_warning "GET /telemetry/summary - может требовать запуска engine"
    fi
}

# Отображение статуса
show_status() {
    echo ""
    log_success "========================================="
    log_success "  HEAN Trading System - ГОТОВ К РАБОТЕ"
    log_success "========================================="
    echo ""
    echo -e "${GREEN}Сервисы:${NC}"
    echo "  • API Backend:  http://localhost:8000"
    echo "  • API Docs:     http://localhost:8000/docs"
    echo "  • UI Frontend:  http://localhost:3000"
    echo "  • Redis:        localhost:6379"
    echo ""
    echo -e "${GREEN}Полезные команды:${NC}"
    echo "  • Логи всех сервисов:  docker compose logs -f"
    echo "  • Логи API:            docker compose logs -f api"
    echo "  • Логи UI:             docker compose logs -f ui"
    echo "  • Статус контейнеров:  docker compose ps"
    echo "  • Остановка:           docker compose down"
    echo "  • Перезапуск:          docker compose restart"
    echo ""

    # Вывод статуса контейнеров
    docker compose ps
}

# Основная функция
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   HEAN Docker Deployment Script v2.0         ║${NC}"
    echo -e "${BLUE}║   Production-Ready Microservices Platform    ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
    echo ""

    # Выполнение проверок
    check_docker
    check_config_files
    check_env_variables
    check_ports

    # Сборка и запуск
    cleanup
    build_images
    start_containers

    # Проверка здоровья
    if check_health; then
        test_api
        show_status

        echo ""
        log_success "Развертывание завершено успешно! 🚀"
        echo ""
        log_info "Для просмотра логов в реальном времени:"
        echo "  docker compose logs -f"
        echo ""
    else
        log_error "Некоторые сервисы не запустились корректно"
        log_info "Проверьте логи: docker compose logs"
        exit 1
    fi
}

# Обработка аргументов
case "${1:-}" in
    "stop")
        log_info "Остановка всех сервисов..."
        docker compose down
        log_success "Сервисы остановлены"
        ;;
    "restart")
        log_info "Перезапуск сервисов..."
        docker compose restart
        log_success "Сервисы перезапущены"
        ;;
    "logs")
        docker compose logs -f
        ;;
    "status")
        docker compose ps
        ;;
    "rebuild")
        cleanup
        build_images
        ;;
    *)
        main
        ;;
esac
