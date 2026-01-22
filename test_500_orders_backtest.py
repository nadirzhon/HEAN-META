#!/usr/bin/env python3
"""Тестовый запуск с 500 ордерами через backtest режим."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.backtest.event_sim import EventSimulator
from hean.backtest.metrics import BacktestMetrics
from hean.config import settings
from hean.logging import get_logger, setup_logging
from hean.main import TradingSystem
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)
setup_logging()


async def run_backtest_with_500_orders() -> None:
    """Запускает backtest для генерации 500 ордеров."""
    print("=" * 70)
    print("🧪 ТЕСТОВЫЙ РЕЖИМ: 500 ОРДЕРОВ (BACKTEST)")
    print("=" * 70)
    print()

    # Сохраняем оригинальные настройки
    original_debug_mode = settings.debug_mode
    original_initial_capital = settings.initial_capital

    # Включаем debug mode и увеличиваем капитал для большего количества сделок
    settings.debug_mode = True
    settings.initial_capital = 10000.0  # Больше капитала = больше сделок

    print("⚙️  Настройки:")
    print(f"   Debug Mode: {settings.debug_mode}")
    print(f"   Initial Capital: ${settings.initial_capital:,.2f}")
    print(f"   Trading Mode: paper")
    print()

    # Создаем EventSimulator для генерации большого количества событий
    # Используем больше дней для генерации большего количества сделок
    days = 7  # 7 дней должно дать достаточно сделок
    start_date = datetime.utcnow() - timedelta(days=days)
    symbols = ["BTCUSDT", "ETHUSDT"]

    print(f"📅 Период: {days} дней")
    print(f"📊 Символы: {', '.join(symbols)}")
    print()

    simulator = EventSimulator(None, symbols, start_date, days)

    # Создаем систему в evaluate mode
    system = TradingSystem(mode="evaluate")

    # Запускаем систему
    print("🚀 Запуск системы...")
    await system.start(price_feed=simulator)
    print("✅ Система запущена")
    print()

    # Запускаем симуляцию
    print("📡 Запуск симуляции...")
    await simulator.run()
    print("✅ Симуляция завершена")
    print()

    # Даем время на обработку всех событий
    print("⏳ Обработка оставшихся событий...")
    await asyncio.sleep(5)

    # Останавливаем систему
    print("🛑 Остановка системы...")
    await system.stop()
    print("✅ Система остановлена")
    print()

    # Получаем метрики
    print("=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТА")
    print("=" * 70)
    print()

    # Получаем статистику из accounting
    equity = system._accounting.get_equity()
    initial_capital = system._accounting.initial_capital
    daily_pnl = system._accounting.get_daily_pnl(equity)
    realized_pnl = system._accounting._realized_pnl
    total_fees = system._accounting._total_fees
    drawdown, drawdown_pct = system._accounting.get_drawdown(equity)

    # Подсчитываем ордера
    total_orders = system._orders_sent
    total_fills = system._orders_filled
    total_positions = len(system._accounting.get_positions())

    # Получаем метрики из paper broker
    paper_broker = system._execution_router._paper_broker
    fill_stats = {}
    if paper_broker:
        fill_stats = paper_broker.get_fill_stats()
        total_fills = fill_stats.get("total_fills", total_fills)

    # Статистика по стратегиям
    strategy_metrics = system._accounting.get_strategy_metrics()

    # Получаем стратегии для метрик
    strategies_dict: dict[str, BaseStrategy] = {}
    if system._impulse_engine:
        strategies_dict["impulse_engine"] = system._impulse_engine
    for strategy in system._strategies:
        if strategy.strategy_id not in strategies_dict:
            strategies_dict[strategy.strategy_id] = strategy

    # Вычисляем полные метрики
    metrics_calc = BacktestMetrics(
        accounting=system._accounting,
        paper_broker=paper_broker,
        strategies=strategies_dict,
        allocator=system._allocator,
        execution_router=system._execution_router,
    )

    full_metrics = metrics_calc.calculate()
    total_trades_metric = full_metrics.get("total_trades", 0)
    total_pnl_metric = full_metrics.get("total_pnl", 0.0)
    profit_factor = full_metrics.get("profit_factor", 0.0)

    print("💰 КАПИТАЛ:")
    print(f"   Начальный капитал: ${initial_capital:,.2f}")
    print(f"   Текущий капитал (Equity): ${equity:,.2f}")
    print(
        f"   Изменение: ${equity - initial_capital:,.2f} ({((equity - initial_capital) / initial_capital * 100):+.2f}%)"
    )
    print()

    print("📈 ПРИБЫЛЬ:")
    if daily_pnl > 0:
        print(f"   ✅ Дневной PnL: +${daily_pnl:,.2f} ({daily_pnl / equity * 100:+.2f}%)")
    elif daily_pnl < 0:
        print(f"   ❌ Дневной PnL: ${daily_pnl:,.2f} ({daily_pnl / equity * 100:+.2f}%)")
    else:
        print(f"   ➖ Дневной PnL: ${daily_pnl:,.2f}")
    print(f"   💸 Реализованный PnL: ${realized_pnl:,.2f}")
    print(f"   📊 Общий PnL (метрики): ${total_pnl_metric:,.2f}")
    print(f"   💳 Всего комиссий: ${total_fees:,.2f}")
    print()

    print("📊 ОРДЕРА И ПОЗИЦИИ:")
    print(f"   Отправлено ордеров: {total_orders}")
    print(f"   Исполнено ордеров: {total_fills}")
    print(f"   Всего сделок (метрики): {total_trades_metric}")
    print(f"   Открытых позиций: {total_positions}")
    print()

    print("📉 РИСКИ:")
    print(f"   Drawdown: ${drawdown:,.2f} ({drawdown_pct:.2f}%)")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print()

    # Статистика по стратегиям
    if strategy_metrics:
        print("=" * 70)
        print("📊 ПО СТРАТЕГИЯМ")
        print("=" * 70)
        print()

        for strategy_id, metrics in sorted(strategy_metrics.items()):
            trades = int(metrics.get("trades", 0))
            pnl = metrics.get("pnl", 0)
            wins = int(metrics.get("wins", 0))
            losses = int(metrics.get("losses", 0))
            win_rate = metrics.get("win_rate_pct", 0)
            profit_factor_strat = metrics.get("profit_factor", 0)

            if trades == 0:
                continue

            print(f"🎯 {strategy_id}:")
            print(f"   Сделок: {trades}")
            if pnl > 0:
                print(f"   ✅ PnL: +${pnl:,.2f}")
            elif pnl < 0:
                print(f"   ❌ PnL: ${pnl:,.2f}")
            else:
                print(f"   ➖ PnL: ${pnl:,.2f}")
            print(f"   Побед: {wins} ({win_rate:.1f}%)")
            print(f"   Поражений: {losses}")
            if profit_factor_strat > 0:
                print(f"   Profit Factor: {profit_factor_strat:.2f}")
            print()

    # Итоговая статистика
    total_trades = sum(int(m.get("trades", 0)) for m in strategy_metrics.values())
    total_pnl = sum(m.get("pnl", 0) for m in strategy_metrics.values())

    print("=" * 70)
    print("📊 ИТОГО")
    print("=" * 70)
    print()
    print(f"   Всего сделок: {total_trades} (метрики: {total_trades_metric})")
    if total_pnl > 0:
        print(f"   ✅ Общий PnL: +${total_pnl:,.2f}")
    elif total_pnl < 0:
        print(f"   ❌ Общий PnL: ${total_pnl:,.2f}")
    else:
        print(f"   ➖ Общий PnL: ${total_pnl:,.2f}")
    print()

    # Восстанавливаем настройки
    settings.debug_mode = original_debug_mode
    settings.initial_capital = original_initial_capital

    print("=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 70)
    print()

    # Итоговый вывод
    if total_pnl_metric > 0:
        print(f"🎉 УСПЕХ! Проект принес прибыль: +${total_pnl_metric:,.2f}")
        print(f"   Сделок: {total_trades_metric}")
        print(f"   Profit Factor: {profit_factor:.2f}")
    elif total_pnl_metric < 0:
        print(f"⚠️  Убыток: ${total_pnl_metric:,.2f}")
        print(f"   Сделок: {total_trades_metric}")
    else:
        print(f"➖ Прибыль нулевая")
        print(f"   Сделок: {total_trades_metric}")
    print()

    # Проверка на 500 ордеров
    if total_trades_metric >= 500:
        print(f"✅ Цель достигнута! Создано {total_trades_metric} ордеров (цель: 500)")
    else:
        print(f"⚠️  Создано только {total_trades_metric} ордеров из 500 запланированных")
        print(f"   Попробуйте увеличить количество дней или капитал")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(run_backtest_with_500_orders())
    except KeyboardInterrupt:
        print("\n⚠️  Тест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
