#!/bin/bash
# Скрипт для запуска РЕАЛЬНОЙ торговли

echo "======================================================================"
echo "🚀 ЗАПУСК РЕАЛЬНОЙ ТОРГОВЛИ"
echo "======================================================================"
echo ""

# Установить DRY_RUN=false
export DRY_RUN=false

# Проверка настроек
echo "Текущие настройки:"
python3 << PYTHON
from hean.config import settings
import os
os.environ['DRY_RUN'] = 'false'
print(f"LIVE_CONFIRM: {settings.live_confirm}")
print(f"DRY_RUN: {settings.dry_run}")
print(f"TRADING_MODE: {settings.trading_mode}")
print(f"BYBIT_TESTNET: {settings.bybit_testnet}")
PYTHON

echo ""
echo "⚠️  ВНИМАНИЕ: Запуск РЕАЛЬНОЙ торговли!"
echo "   • Используются РЕАЛЬНЫЕ деньги"
echo "   • Баланс на счету: ~30.53 USDT"
echo "   • Нажмите Ctrl+C для остановки"
echo ""
echo "Запускаю систему..."
echo ""

# Запуск системы
DRY_RUN=false python3 -m hean.main run

