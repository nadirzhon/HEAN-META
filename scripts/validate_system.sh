#!/bin/bash
# HEAN System Validation Script
# Проверяет конфигурацию, зависимости и готовность системы к запуску

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          HEAN SYSTEM VALIDATION SCRIPT v2.0                   ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" == "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" == "WARNING" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Track issues
CRITICAL_ISSUES=0
WARNINGS=0

echo -e "${BLUE}[1/7] Проверка структуры проекта${NC}"
echo "───────────────────────────────────────────────────────────────"

# Check critical directories
for dir in "src/hean" "tests" "cpp_core" "scripts"; do
    if [ -d "$dir" ]; then
        print_status "OK" "Директория $dir найдена"
    else
        print_status "ERROR" "Директория $dir не найдена!"
        ((CRITICAL_ISSUES++))
    fi
done

# Check critical files
for file in "src/hean/main.py" "src/hean/config.py" "backend.env" "docker-compose.yml"; do
    if [ -f "$file" ]; then
        print_status "OK" "Файл $file найден"
    else
        print_status "ERROR" "Файл $file не найден!"
        ((CRITICAL_ISSUES++))
    fi
done
echo ""

echo -e "${BLUE}[2/7] Проверка backend.env конфигурации${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "backend.env" ]; then
    # Check for duplicate keys
    DUPLICATES=$(grep -E "^[A-Z_]+=" backend.env | cut -d= -f1 | sort | uniq -d)
    if [ -z "$DUPLICATES" ]; then
        print_status "OK" "Нет дублирующихся параметров"
    else
        print_status "ERROR" "Найдены дублирующиеся параметры: $DUPLICATES"
        ((CRITICAL_ISSUES++))
    fi

    # Check critical parameters
    if grep -q "^PROCESS_FACTORY_ENABLED=true" backend.env; then
        print_status "OK" "Process Factory включен"
    else
        print_status "WARNING" "Process Factory отключен (упускаются пассивные доходы)"
        ((WARNINGS++))
    fi

    if grep -q "^PROFIT_CAPTURE_ENABLED=true" backend.env; then
        print_status "OK" "Profit Capture включен"
    else
        print_status "WARNING" "Profit Capture отключен (нет защиты прибыли)"
        ((WARNINGS++))
    fi

    if grep -q "^MULTI_SYMBOL_ENABLED=true" backend.env; then
        print_status "OK" "Multi-symbol торговля включена"
        SYMBOLS=$(grep "^TRADING_SYMBOLS=" backend.env | cut -d= -f2)
        SYMBOL_COUNT=$(echo "$SYMBOLS" | tr ',' '\n' | wc -l)
        print_status "OK" "Настроено символов: $SYMBOL_COUNT ($SYMBOLS)"
    else
        print_status "WARNING" "Multi-symbol торговля отключена"
        ((WARNINGS++))
    fi

    # Check trading mode
    if grep -q "^BYBIT_TESTNET=false" backend.env; then
        print_status "WARNING" "⚠️  ВНИМАНИЕ: LIVE TRADING MODE АКТИВЕН! ⚠️"
        ((WARNINGS++))
        if grep -q "^LIVE_CONFIRM=YES" backend.env && grep -q "^REQUIRE_LIVE_CONFIRM=true" backend.env; then
            print_status "OK" "Live trading подтверждён корректно"
        else
            print_status "ERROR" "Live trading включен без подтверждения!"
            ((CRITICAL_ISSUES++))
        fi
    else
        print_status "OK" "Testnet режим активен (безопасно)"
    fi

    # Check API keys
    if grep -q "^BYBIT_API_KEY=" backend.env && [ ! -z "$(grep "^BYBIT_API_KEY=" backend.env | cut -d= -f2)" ]; then
        print_status "OK" "Bybit API ключ установлен"
    else
        print_status "ERROR" "Bybit API ключ не установлен!"
        ((CRITICAL_ISSUES++))
    fi
else
    print_status "ERROR" "backend.env не найден!"
    ((CRITICAL_ISSUES++))
fi
echo ""

echo -e "${BLUE}[3/7] Проверка C++ модулей${NC}"
echo "───────────────────────────────────────────────────────────────"

CPP_MODULES_DIR="src/hean/cpp_modules"
if [ -d "$CPP_MODULES_DIR" ]; then
    MODULE_COUNT=$(find "$CPP_MODULES_DIR" -name "*.so" -o -name "*.dylib" 2>/dev/null | wc -l)
    if [ "$MODULE_COUNT" -gt 0 ]; then
        print_status "OK" "C++ модули найдены: $MODULE_COUNT файлов"
        find "$CPP_MODULES_DIR" -name "*.so" -o -name "*.dylib" 2>/dev/null | while read module; do
            echo "    - $(basename $module)"
        done
    else
        print_status "WARNING" "C++ модули не собраны (система работает в медленном режиме)"
        print_status "WARNING" "Запустите: cd cpp_core && mkdir -p build && cd build && cmake .. && make"
        ((WARNINGS++))
    fi
else
    print_status "WARNING" "Директория cpp_modules не найдена"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}[4/7] Проверка тестов${NC}"
echo "───────────────────────────────────────────────────────────────"

TEST_COUNT=$(find tests -name "test_*.py" 2>/dev/null | wc -l)
if [ "$TEST_COUNT" -gt 0 ]; then
    print_status "OK" "Найдено тестов: $TEST_COUNT"

    if [ -f ".coverage" ]; then
        print_status "OK" "Coverage данные доступны"
    else
        print_status "WARNING" "Coverage данные не найдены (запустите: pytest --cov)"
        ((WARNINGS++))
    fi
else
    print_status "ERROR" "Тесты не найдены!"
    ((CRITICAL_ISSUES++))
fi
echo ""

echo -e "${BLUE}[5/7] Проверка Docker конфигурации${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "docker-compose.yml" ]; then
    print_status "OK" "docker-compose.yml найден"

    # Check for restart policy
    if grep -q "restart:" docker-compose.yml; then
        RESTART_POLICY=$(grep "restart:" docker-compose.yml | head -1 | awk '{print $2}')
        if [ "$RESTART_POLICY" == "unless-stopped" ] || [ "$RESTART_POLICY" == "always" ]; then
            print_status "OK" "Restart policy: $RESTART_POLICY"
        else
            print_status "WARNING" "Restart policy не оптимален: $RESTART_POLICY"
            ((WARNINGS++))
        fi
    else
        print_status "WARNING" "Restart policy не настроен"
        ((WARNINGS++))
    fi

    # Check for healthcheck
    if grep -q "healthcheck:" docker-compose.yml; then
        print_status "OK" "Healthcheck настроен"
    else
        print_status "WARNING" "Healthcheck не настроен"
        ((WARNINGS++))
    fi
else
    print_status "ERROR" "docker-compose.yml не найден!"
    ((CRITICAL_ISSUES++))
fi
echo ""

echo -e "${BLUE}[6/7] Проверка Python зависимостей${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "pyproject.toml" ]; then
    print_status "OK" "pyproject.toml найден"
else
    print_status "WARNING" "pyproject.toml не найден"
    ((WARNINGS++))
fi

# Check if running in virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_status "OK" "Virtual environment активен: $VIRTUAL_ENV"
else
    print_status "WARNING" "Virtual environment не активен"
    ((WARNINGS++))
fi
echo ""

echo -e "${BLUE}[7/7] Проверка стратегий${NC}"
echo "───────────────────────────────────────────────────────────────"

STRATEGY_DIR="src/hean/strategies"
if [ -d "$STRATEGY_DIR" ]; then
    STRATEGY_COUNT=$(find "$STRATEGY_DIR" -name "*.py" ! -name "__init__.py" 2>/dev/null | wc -l)
    print_status "OK" "Найдено файлов стратегий: $STRATEGY_COUNT"

    # Check for specific strategies
    for strategy in "impulse_engine.py" "funding_harvester.py" "basis_arbitrage.py"; do
        if [ -f "$STRATEGY_DIR/$strategy" ]; then
            print_status "OK" "Стратегия $strategy найдена"
        else
            print_status "WARNING" "Стратегия $strategy не найдена"
            ((WARNINGS++))
        fi
    done
else
    print_status "ERROR" "Директория стратегий не найдена!"
    ((CRITICAL_ISSUES++))
fi
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    ИТОГОВЫЙ ОТЧЁТ                             ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $CRITICAL_ISSUES -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ ОТЛИЧНО!${NC} Система полностью готова к запуску!"
    echo ""
    echo "Следующие шаги:"
    echo "  1. Проверьте режим торговли (testnet/live) в backend.env"
    echo "  2. Запустите систему: docker-compose up -d"
    echo "  3. Проверьте логи: docker-compose logs -f api"
    exit 0
elif [ $CRITICAL_ISSUES -eq 0 ]; then
    echo -e "${YELLOW}⚠ ПРЕДУПРЕЖДЕНИЯ:${NC} Найдено предупреждений: $WARNINGS"
    echo ""
    echo "Система может быть запущена, но рекомендуется исправить предупреждения"
    echo "для оптимальной производительности."
    exit 0
else
    echo -e "${RED}✗ КРИТИЧЕСКИЕ ПРОБЛЕМЫ:${NC} Найдено: $CRITICAL_ISSUES"
    echo -e "${YELLOW}⚠ Предупреждения:${NC} Найдено: $WARNINGS"
    echo ""
    echo "СИСТЕМА НЕ ГОТОВА К ЗАПУСКУ!"
    echo "Пожалуйста, исправьте критические проблемы перед запуском."
    exit 1
fi
