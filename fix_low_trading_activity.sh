#!/bin/bash
# Скрипт для исправления проблемы с низкой активностью торговли

echo "🔧 ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С НИЗКОЙ АКТИВНОСТЬЮ ТОРГОВЛИ"
echo "=================================================="
echo ""

# Проверка наличия .env файла
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден!"
    exit 1
fi

echo "📝 Обновление настроек в .env..."
echo ""

# Создать backup
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Создан backup: .env.backup.*"

# Добавить/обновить настройки
echo ""
echo "Добавление/обновление настроек..."

# PAPER_TRADE_ASSIST
if grep -q "PAPER_TRADE_ASSIST" .env; then
    sed -i '' 's/^PAPER_TRADE_ASSIST=.*/PAPER_TRADE_ASSIST=true/' .env
    echo "✅ PAPER_TRADE_ASSIST обновлен"
else
    echo "PAPER_TRADE_ASSIST=true" >> .env
    echo "✅ PAPER_TRADE_ASSIST добавлен"
fi

# DEBUG_MODE (для тестирования)
if grep -q "DEBUG_MODE" .env; then
    sed -i '' 's/^DEBUG_MODE=.*/DEBUG_MODE=true/' .env
    echo "✅ DEBUG_MODE обновлен"
else
    echo "DEBUG_MODE=true" >> .env
    echo "✅ DEBUG_MODE добавлен"
fi

# MAX_DAILY_ATTEMPTS_PER_STRATEGY
if grep -q "MAX_DAILY_ATTEMPTS_PER_STRATEGY" .env; then
    sed -i '' 's/^MAX_DAILY_ATTEMPTS_PER_STRATEGY=.*/MAX_DAILY_ATTEMPTS_PER_STRATEGY=50/' .env
    echo "✅ MAX_DAILY_ATTEMPTS_PER_STRATEGY обновлен"
else
    echo "MAX_DAILY_ATTEMPTS_PER_STRATEGY=50" >> .env
    echo "✅ MAX_DAILY_ATTEMPTS_PER_STRATEGY добавлен"
fi

echo ""
echo "=================================================="
echo "✅ Настройки обновлены!"
echo ""
echo "📋 Примененные изменения:"
echo "   - PAPER_TRADE_ASSIST=true (ослабляет фильтры)"
echo "   - DEBUG_MODE=true (для тестирования)"
echo "   - MAX_DAILY_ATTEMPTS_PER_STRATEGY=50 (больше попыток)"
echo ""
echo "🔄 Перезапуск системы..."
docker-compose restart hean-afo-engine
echo ""
echo "✅ Система перезапущена!"
echo ""
echo "📊 Проверка логов (последние 20 строк):"
echo "----------------------------------------"
docker-compose logs --tail=20 hean-afo-engine
echo ""
echo "=================================================="
echo "✅ ГОТОВО!"
echo ""
echo "💡 Следующие шаги:"
echo "   1. Следите за логами: docker-compose logs -f hean-afo-engine"
echo "   2. Проверьте сигналы: docker-compose logs hean-afo-engine | grep -i signal"
echo "   3. Проверьте ордера: curl http://localhost:8000/api/orders"
echo ""
echo "⚠️  ВАЖНО: DEBUG_MODE=true только для тестирования!"
echo "   Для production установите DEBUG_MODE=false"
echo ""
