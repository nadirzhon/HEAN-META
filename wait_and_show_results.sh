#!/bin/bash
# Скрипт для ожидания завершения бэктеста и показа результатов

LOG_FILE="backtest_30days_output.log"

echo "⏳ Ожидание завершения бэктеста..."
echo ""

# Ждем завершения процесса
while ps aux | grep -q "[b]acktest --days 30"; do
    sleep 5
    echo -n "."
done

echo ""
echo ""
echo "✅ Бэктест завершен! Извлекаю результаты..."
echo ""

# Ждем немного, чтобы отчет записался
sleep 2

# Показываем результаты
if grep -q "BACKTEST REPORT" "$LOG_FILE"; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ БЭКТЕСТА НА 30 ДНЕЙ"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    grep -A 100 "BACKTEST REPORT" "$LOG_FILE" | head -120
else
    echo "⚠️  Отчет еще не найден, проверяю лог..."
    tail -200 "$LOG_FILE" | grep -E "(Total Trades|Initial Equity|Final Equity|Total Return|Profit Factor)" -A 2
fi

