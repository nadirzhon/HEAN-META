#!/bin/bash
# Скрипт для извлечения результатов бэктеста

LOG_FILE="backtest_30days_output.log"

echo "=== СТАТИСТИКА БЭКТЕСТА (30 дней) ==="
echo ""

# Проверка, завершился ли бэктест
if grep -q "BACKTEST REPORT" "$LOG_FILE"; then
    echo "✅ Бэктест завершен"
    echo ""
    grep -A 50 "BACKTEST REPORT" "$LOG_FILE" | tail -50
else
    echo "⏳ Бэктест еще выполняется..."
    echo ""
    
    # Промежуточная статистика
    echo "--- ПРОМЕЖУТОЧНАЯ СТАТИСТИКА ---"
    echo ""
    
    SIGNALS=$(grep -c "Signal published" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "📊 Сигналов сгенерировано: $SIGNALS"
    
    # Проверка статуса обработки
    if grep -q "Processing events" "$LOG_FILE"; then
        LAST_STATUS=$(grep "Processing events" "$LOG_FILE" | tail -1)
        echo "📈 Статус обработки: $LAST_STATUS"
    fi
    
    # Проверка на наличие статистики paper broker
    if grep -q "Paper broker fill stats" "$LOG_FILE"; then
        echo ""
        echo "--- СТАТИСТИКА ЗАПОЛНЕНИЯ ОРДЕРОВ ---"
        grep "Paper broker fill stats" "$LOG_FILE" | tail -1
    fi
    
    echo ""
    echo "💡 Для полных результатов дождитесь завершения бэктеста"
    echo "   Мониторинг: tail -f $LOG_FILE | grep -E '(BACKTEST REPORT|Total Trades|Final Equity)'"
fi

echo ""
echo "=== КОНЕЦ ОТЧЕТА ==="

