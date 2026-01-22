#!/usr/bin/env python3
"""Диагностика проблемы с низкой активностью торговли."""

import sys

sys.path.insert(0, "src")

import asyncio
from hean.config import settings
from hean.api.engine_facade import EngineFacade
from hean.observability.no_trade_report import no_trade_report


async def diagnose():
    print("=" * 80)
    print("🔍 ДИАГНОСТИКА ПРОБЛЕМЫ С ТОРГОВЛЕЙ")
    print("=" * 80)
    print()

    # 1. Проверка настроек
    print("=" * 80)
    print("1️⃣  ПРОВЕРКА НАСТРОЕК")
    print("=" * 80)
    print()

    print(f"📊 Режим торговли: {settings.trading_mode}")
    print(f"🌐 Live режим: {settings.is_live}")
    print(f"🔒 Dry Run: {settings.dry_run}")
    print(f"🧪 Testnet: {settings.bybit_testnet}")
    print(f"💡 Paper Trade Assist: {settings.paper_trade_assist}")
    print()

    print(f"💰 Капитал: ${settings.initial_capital:,.2f}")
    print(f"📈 Макс. риск на сделку: {settings.max_trade_risk_pct}%")
    print(f"📉 Макс. дневной drawdown: {settings.max_daily_drawdown_pct}%")
    print(f"🔢 Макс. открытых позиций: {settings.max_open_positions}")
    print()

    # 2. Проверка статуса системы
    print("=" * 80)
    print("2️⃣  СТАТУС СИСТЕМЫ")
    print("=" * 80)
    print()

    facade = EngineFacade()
    status = await facade.get_status()

    print(f"Статус: {status.get('status')}")
    print(f"Equity: ${status.get('equity', 0):,.2f}")
    print()

    # 3. Проверка ордеров
    print("=" * 80)
    print("3️⃣  ОРДЕРА")
    print("=" * 80)
    print()

    orders = await facade.get_orders()
    filled = [o for o in orders if o.get("status") == "filled"]

    print(f"Всего ордеров: {len(orders)}")
    print(f"Заполнено: {len(filled)}")
    print()

    if filled:
        print("Последние заполненные ордера:")
        for o in filled[-5:]:
            print(
                f"  {o.get('symbol')} {o.get('side')} | {o.get('strategy_id')} | {o.get('timestamp')}"
            )
        print()

    # 4. Проверка блокировок сигналов
    print("=" * 80)
    print("4️⃣  БЛОКИРОВКИ СИГНАЛОВ")
    print("=" * 80)
    print()

    summary = no_trade_report.get_summary()

    print("📊 Общая статистика блокировок:")
    if summary.totals:
        for reason, count in sorted(summary.totals.items(), key=lambda x: x[1], reverse=True):
            print(f"   {reason}: {count}")
    else:
        print("   ⚠️  Нет данных о блокировках (система может быть не запущена)")
    print()

    print("📈 Pipeline счетчики:")
    if summary.pipeline_counters:
        for counter, count in sorted(
            summary.pipeline_counters.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   {counter}: {count}")
    else:
        print("   ⚠️  Нет данных")
    print()

    # 5. Анализ проблемы
    print("=" * 80)
    print("5️⃣  АНАЛИЗ ПРОБЛЕМЫ")
    print("=" * 80)
    print()

    signals_emitted = summary.pipeline_counters.get("signals_emitted", 0)
    orders_created = summary.pipeline_counters.get("orders_created", 0)

    print(f"📊 Сигналов сгенерировано: {signals_emitted}")
    print(f"📋 Ордеров создано: {orders_created}")

    if signals_emitted > 0:
        conversion_rate = (orders_created / signals_emitted) * 100
        print(f"📈 Конверсия сигналов в ордера: {conversion_rate:.2f}%")
        print()

        if conversion_rate < 10:
            print("⚠️  ПРОБЛЕМА: Очень низкая конверсия сигналов в ордера!")
            print("   Возможные причины:")
            print("   1. Слишком строгие фильтры (spread, volatility, edge)")
            print("   2. Блокировки системой управления рисками")
            print("   3. Блокировки deposit protection")
            print("   4. Достигнуты лимиты на попытки/cooldown")
            print()
    else:
        print("⚠️  ПРОБЛЕМА: Сигналы вообще не генерируются!")
        print("   Возможные причины:")
        print("   1. Стратегии не активны или не получают тики")
        print("   2. Режим рынка не подходит для стратегий")
        print("   3. Стратегии заблокированы killswitch")
        print("   4. Нет подключения к price feed")
        print()

    # 6. Рекомендации
    print("=" * 80)
    print("6️⃣  РЕКОМЕНДАЦИИ")
    print("=" * 80)
    print()

    if not settings.paper_trade_assist and settings.dry_run:
        print("✅ ВКЛЮЧИТЕ PAPER_TRADE_ASSIST для более активной торговли в paper режиме:")
        print("   Добавьте в .env: PAPER_TRADE_ASSIST=true")
        print()

    if signals_emitted == 0:
        print("✅ ПРОВЕРЬТЕ:")
        print("   1. Запущена ли система (docker-compose logs hean-afo-engine)")
        print("   2. Получает ли система тики (проверьте логи на TICK events)")
        print("   3. Активны ли стратегии (проверьте логи на 'Strategy started')")
        print()

    if signals_emitted > 0 and orders_created == 0:
        print("✅ ПРОВЕРЬТЕ ЛОГИ на наличие блокировок:")
        print("   docker-compose logs hean-afo-engine | grep -i 'blocked\\|reject'")
        print()
        print("✅ ВРЕМЕННО ОСЛАБЬТЕ ФИЛЬТРЫ для тестирования:")
        print("   Добавьте в .env:")
        print("   DEBUG_MODE=true")
        print("   PAPER_TRADE_ASSIST=true")
        print()

    print("=" * 80)
    print("✅ Диагностика завершена")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(diagnose())
