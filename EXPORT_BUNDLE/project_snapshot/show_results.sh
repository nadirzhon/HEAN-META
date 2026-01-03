#!/bin/bash
# Скрипт для показа результатов бэктеста

LOG_FILE="backtest_30days_output.log"

echo "═══════════════════════════════════════════════════════════════"
echo "  РЕЗУЛЬТАТЫ БЭКТЕСТА НА 30 ДНЕЙ"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Проверка статуса
if ps aux | grep -q "[b]acktest --days 30"; then
    echo "⏳ БЭКТЕСТ ВЫПОЛНЯЕТСЯ..."
    echo ""
else
    echo "✅ БЭКТЕСТ ЗАВЕРШЕН"
    echo ""
fi

# Подсчет сигналов
SIGNALS=$(grep -c "Signal published" "$LOG_FILE" 2>/dev/null || echo "0")
echo "📊 Сигналов сгенерировано: $SIGNALS"
echo ""

# Поиск финального отчета
if grep -q "BACKTEST REPORT" "$LOG_FILE"; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ФИНАЛЬНЫЙ ОТЧЕТ"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    grep -A 80 "BACKTEST REPORT" "$LOG_FILE" | head -90
else
    echo "⚠️  ФИНАЛЬНЫЙ ОТЧЕТ ЕЩЕ НЕ ГОТОВ"
    echo ""
    echo "Бэктест обрабатывает события. Финальные результаты появятся"
    echo "после завершения обработки всех событий."
    echo ""
    echo "Для мониторинга используйте:"
    echo "  tail -f $LOG_FILE | grep -E '(BACKTEST REPORT|Total Trades|Final Equity)'"
    echo ""
    
    # Показываем прогресс
    if grep -q "Processing events" "$LOG_FILE"; then
        echo "📈 Прогресс обработки:"
        grep "Processing events" "$LOG_FILE" | tail -1
        echo ""
    fi
fi

echo "═══════════════════════════════════════════════════════════════"

