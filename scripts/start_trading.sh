#!/bin/bash
# Скрипт для запуска торговли

echo "======================================================================"
echo "🚀 ЗАПУСК СИСТЕМЫ ТОРГОВЛИ"
echo "======================================================================"
echo ""

# Проверка настроек
python3 -c "from hean.config import settings; print(f'LIVE_CONFIRM: {settings.live_confirm}'); print(f'DRY_RUN: {settings.dry_run}'); print(f'TRADING_MODE: {settings.trading_mode}')"

echo ""
echo "Запускаю систему..."
echo "Нажмите Ctrl+C для остановки"
echo ""

# Запуск системы
python3 -m hean.main run
