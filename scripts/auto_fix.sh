#!/bin/bash
# HEAN Auto-Fix Script v2.0
# Автоматически исправляет распространенные проблемы конфигурации

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          HEAN AUTO-FIX SCRIPT v2.0                            ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

FIXES_APPLIED=0
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Function to apply fix
apply_fix() {
    local description=$1
    echo -e "${YELLOW}→${NC} $description"
    ((FIXES_APPLIED++))
}

# Function to skip fix
skip_fix() {
    local description=$1
    echo -e "${GREEN}✓${NC} $description (уже исправлено)"
}

# Create backup directory
echo -e "${BLUE}[1/5] Создание резервной копии${NC}"
echo "───────────────────────────────────────────────────────────────"
mkdir -p "$BACKUP_DIR"
if [ -f "backend.env" ]; then
    cp backend.env "$BACKUP_DIR/backend.env.backup"
    echo -e "${GREEN}✓${NC} Backup создан: $BACKUP_DIR/backend.env.backup"
fi
echo ""

# Fix 1: Remove duplicate PROCESS_FACTORY_ALLOW_ACTIONS
echo -e "${BLUE}[2/5] Проверка дублирующихся параметров${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "backend.env" ]; then
    DUPLICATES=$(grep -E "^PROCESS_FACTORY_ALLOW_ACTIONS=" backend.env | wc -l)
    if [ "$DUPLICATES" -gt 1 ]; then
        apply_fix "Удаление дублирующегося PROCESS_FACTORY_ALLOW_ACTIONS"
        # Keep only the first occurrence (true), remove false
        sed -i.bak '/^PROCESS_FACTORY_ALLOW_ACTIONS=false/d' backend.env
    else
        skip_fix "Нет дублирующихся параметров"
    fi
fi
echo ""

# Fix 2: Ensure critical features are enabled
echo -e "${BLUE}[3/5] Проверка критических функций${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "backend.env" ]; then
    # Check and add PROCESS_FACTORY_ENABLED if missing
    if ! grep -q "^PROCESS_FACTORY_ENABLED=" backend.env; then
        apply_fix "Добавление PROCESS_FACTORY_ENABLED=true"
        # Find the Process Factory section and add the line
        sed -i.bak '/^# Process Factory/a\
PROCESS_FACTORY_ENABLED=true' backend.env
    else
        if grep -q "^PROCESS_FACTORY_ENABLED=false" backend.env; then
            echo -e "${YELLOW}⚠${NC} Process Factory отключен (рекомендуется включить)"
            read -p "Включить Process Factory? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sed -i.bak 's/^PROCESS_FACTORY_ENABLED=false/PROCESS_FACTORY_ENABLED=true/' backend.env
                apply_fix "Process Factory включен"
            fi
        else
            skip_fix "Process Factory уже включен"
        fi
    fi

    # Check PROFIT_CAPTURE_ENABLED
    if ! grep -q "^PROFIT_CAPTURE_ENABLED=" backend.env; then
        apply_fix "Добавление Profit Capture конфигурации"
        cat >> backend.env << 'EOF'

# Profit Capture System (Auto-added by auto_fix.sh)
PROFIT_CAPTURE_ENABLED=true
PROFIT_CAPTURE_TARGET_PCT=20.0
PROFIT_CAPTURE_TRAIL_PCT=10.0
PROFIT_CAPTURE_MODE=partial
PROFIT_CAPTURE_AFTER_ACTION=continue
PROFIT_CAPTURE_CONTINUE_RISK_MULT=0.25
EOF
    else
        skip_fix "Profit Capture уже настроен"
    fi

    # Check MULTI_SYMBOL_ENABLED
    if ! grep -q "^MULTI_SYMBOL_ENABLED=" backend.env; then
        apply_fix "Добавление Multi-Symbol конфигурации"
        cat >> backend.env << 'EOF'

# Multi-Symbol Trading (Auto-added by auto_fix.sh)
MULTI_SYMBOL_ENABLED=true
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT
EOF
    else
        skip_fix "Multi-Symbol уже настроен"
    fi
fi
echo ""

# Fix 3: Optimize trading symbols
echo -e "${BLUE}[4/5] Оптимизация списка символов${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "backend.env" ]; then
    TRADING_SYMBOLS=$(grep "^TRADING_SYMBOLS=" backend.env | cut -d= -f2)
    SYMBOL_COUNT=$(echo "$TRADING_SYMBOLS" | tr ',' '\n' | wc -l)

    if [ "$SYMBOL_COUNT" -lt 3 ]; then
        echo -e "${YELLOW}⚠${NC} Только $SYMBOL_COUNT символов настроено"
        read -p "Добавить рекомендуемые символы? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sed -i.bak 's/^TRADING_SYMBOLS=.*/TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT/' backend.env
            apply_fix "Добавлены рекомендуемые символы (5 шт)"
        fi
    else
        skip_fix "Настроено символов: $SYMBOL_COUNT"
    fi
fi
echo ""

# Fix 4: Safety checks
echo -e "${BLUE}[5/5] Проверка безопасности${NC}"
echo "───────────────────────────────────────────────────────────────"

if [ -f "backend.env" ]; then
    if grep -q "^BYBIT_TESTNET=false" backend.env; then
        echo -e "${RED}⚠ ВНИМАНИЕ!${NC} LIVE TRADING MODE АКТИВЕН!"
        echo ""
        echo "Это означает, что система будет торговать реальными деньгами."
        echo "Убедитесь, что:"
        echo "  1. Все тесты пройдены успешно"
        echo "  2. Система протестирована на testnet"
        echo "  3. Вы понимаете все риски"
        echo ""
        read -p "Переключиться на testnet для безопасности? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sed -i.bak 's/^BYBIT_TESTNET=false/BYBIT_TESTNET=true/' backend.env
            sed -i.bak 's/^LIVE_CONFIRM=YES/LIVE_CONFIRM=NO/' backend.env
            apply_fix "Переключено на testnet режим"
        fi
    else
        skip_fix "Testnet режим активен (безопасно)"
    fi

    # Check API keys
    API_KEY=$(grep "^BYBIT_API_KEY=" backend.env | cut -d= -f2)
    if [ -z "$API_KEY" ]; then
        echo -e "${RED}✗${NC} BYBIT_API_KEY не установлен!"
        echo "Получите ключи API на https://testnet.bybit.com (для тестов)"
        echo "или https://www.bybit.com (для live торговли)"
    else
        skip_fix "Bybit API ключ установлен"
    fi
fi
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    ИТОГИ                                      ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $FIXES_APPLIED -gt 0 ]; then
    echo -e "${GREEN}✓ Применено исправлений: $FIXES_APPLIED${NC}"
    echo ""
    echo "Резервная копия сохранена в: $BACKUP_DIR"
    echo ""
    echo "Следующие шаги:"
    echo "  1. Проверьте изменения: git diff backend.env"
    echo "  2. Запустите валидацию: ./scripts/validate_system.sh"
    echo "  3. Перезапустите систему: docker-compose restart"
else
    echo -e "${GREEN}✓ Все проверки пройдены!${NC} Исправления не требуются."
    echo ""
    echo "Система готова к запуску."
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
