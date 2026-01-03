#!/bin/bash
# Скрипт для извлечения статистики бэктеста

LOG_FILE="backtest_30days_output.log"

echo "═══════════════════════════════════════════════════════════════"
echo "  РЕЗУЛЬТАТЫ БЭКТЕСТА НА 30 ДНЕЙ"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Проверка статуса
if ps aux | grep -q "[b]acktest --days 30"; then
    echo "⏳ Статус: Бэктест выполняется..."
    echo ""
else
    echo "✅ Статус: Бэктест завершен"
    echo ""
fi

# Статистика сигналов
SIGNALS=$(grep -c "Signal published" "$LOG_FILE" 2>/dev/null || echo "0")
echo "📊 СИГНАЛОВ СГЕНЕРИРОВАНО: $SIGNALS"
echo ""

# Статистика заполнения ордеров
if grep -q "Paper broker fill stats" "$LOG_FILE"; then
    echo "📈 СТАТИСТИКА ЗАПОЛНЕНИЯ ОРДЕРОВ:"
    grep "Paper broker fill stats" "$LOG_FILE" | tail -1 | sed 's/.*Paper broker fill stats: //' | python3 -m json.tool 2>/dev/null || grep "Paper broker fill stats" "$LOG_FILE" | tail -1
    echo ""
fi

# Поиск финального отчета
if grep -q "BACKTEST REPORT" "$LOG_FILE"; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ФИНАЛЬНЫЙ ОТЧЕТ"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    grep -A 100 "BACKTEST REPORT" "$LOG_FILE" | head -60
else
    echo "💡 Финальный отчет еще не готов"
    echo "   Бэктест обрабатывает события..."
    echo ""
    
    # Промежуточная информация
    if grep -q "Processing events" "$LOG_FILE"; then
        echo "📊 Прогресс обработки:"
        grep "Processing events" "$LOG_FILE" | tail -3
        echo ""
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"

