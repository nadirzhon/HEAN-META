#!/bin/bash
# Скрипт для проверки результатов бэктеста

cd /Users/macbookpro/Desktop/HEAN

echo "=========================================="
echo "📊 Статус бэктеста на 30 дней"
echo "=========================================="
echo ""

# Проверка процесса
if ps aux | grep -q "[h]ean.main backtest"; then
    echo "⏳ Бэктест выполняется..."
    echo ""
    echo "Последние строки лога:"
    tail -5 backtest_30days_output.log 2>/dev/null | tail -3
else
    echo "✅ Бэктест завершен!"
    echo ""
fi

echo ""
echo "=========================================="
echo "📈 Результаты (если доступны):"
echo "=========================================="

if [ -f backtest_results_30days.json ]; then
    python3 << 'EOF'
import json
import sys

try:
    with open('backtest_results_30days.json', 'r') as f:
        data = json.load(f)
    
    print(f"💰 Начальный капитал: ${data.get('initial_equity', 0):,.2f}")
    print(f"💰 Финальный капитал: ${data.get('final_equity', 0):,.2f}")
    print(f"📊 Общий PnL: ${data.get('total_pnl', 0):,.2f}")
    print(f"📈 Доходность: {data.get('total_return', 0):.2f}%")
    print(f"📉 Максимальный Drawdown: {data.get('max_drawdown_pct', 0):.2f}%")
    print(f"🎯 Profit Factor: {data.get('profit_factor', 0):.2f}")
    print(f"📝 Всего сделок: {data.get('total_trades', 0)}")
    print(f"✅ Прибыльных: {data.get('winning_trades', 0)}")
    print(f"❌ Убыточных: {data.get('losing_trades', 0)}")
    if data.get('total_trades', 0) > 0:
        win_rate = (data.get('winning_trades', 0) / data.get('total_trades', 1)) * 100
        print(f"📊 Win Rate: {win_rate:.1f}%")
except Exception as e:
    print(f"⚠️  Ошибка чтения результатов: {e}")
EOF
else
    echo "⏳ Результаты еще не готовы..."
fi

echo ""
echo "=========================================="

