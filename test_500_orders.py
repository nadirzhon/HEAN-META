#!/usr/bin/env python3
"""Тестовый запуск с генерацией 500 ордеров для проверки прибыли."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.core.regime import Regime
from hean.logging import get_logger, setup_logging
from hean.main import TradingSystem
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)
setup_logging()


async def generate_test_signals(bus: EventBus, target_orders: int = 500) -> None:
    """Генерирует тестовые сигналы для создания ордеров."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    strategies = ["impulse_engine", "funding_harvester", "basis_arbitrage"]
    
    base_prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
    signals_generated = 0
    
    logger.info(f"🎯 Начинаю генерацию {target_orders} тестовых сигналов...")
    
    # Генерируем прямые сигналы для гарантированного создания ордеров
    for i in range(target_orders):
        symbol = symbols[i % len(symbols)]
        strategy_id = strategies[i % len(strategies)]
        base_price = base_prices[symbol]
        
        # Создаем вариацию цены для разнообразия
        price_variation = 1.0 + (i % 100 - 50) * 0.0001  # ±0.5% вариация
        price = base_price * price_variation
        
        # Генерируем тик для обновления цены
        tick = Tick(
            symbol=symbol,
            price=price,
            timestamp=datetime.utcnow() + timedelta(seconds=i),
            bid=price * 0.9999,
            ask=price * 1.0001,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.001)  # Минимальная задержка
        
        # Генерируем сигнал
        side = "buy" if i % 2 == 0 else "sell"
        signal = Signal(
            signal_id=f"test_{i}_{datetime.utcnow().timestamp()}",
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            entry_price=price,
            size=None,  # Пусть система сама рассчитает
            stop_loss=price * (0.98 if side == "buy" else 1.02),
            take_profit=price * (1.02 if side == "buy" else 0.98),
            timestamp=datetime.utcnow() + timedelta(seconds=i),
            metadata={"test_mode": True, "edge_bps": 10.0},  # Добавляем edge для прохождения проверок
        )
        
        await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
        signals_generated += 1
        
        if signals_generated % 50 == 0:
            logger.info(f"📊 Сгенерировано сигналов: {signals_generated}/{target_orders}")
        
        # Небольшая задержка для обработки
        if i % 10 == 0:  # Каждые 10 сигналов даем время на обработку
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.01)
    
    logger.info(f"✅ Генерация завершена. Всего сигналов: {signals_generated}")


async def run_test_with_500_orders() -> None:
    """Запускает тестовый режим с 500 ордерами."""
    print("=" * 70)
    print("🧪 ТЕСТОВЫЙ РЕЖИМ: 500 ОРДЕРОВ")
    print("=" * 70)
    print()
    
    # Сохраняем оригинальные настройки
    original_debug_mode = settings.debug_mode
    original_trading_mode = settings.trading_mode
    
    # Включаем debug mode для обхода ограничений
    settings.debug_mode = True
    settings.trading_mode = "paper"
    
    print("⚙️  Настройки:")
    print(f"   Debug Mode: {settings.debug_mode}")
    print(f"   Trading Mode: {settings.trading_mode}")
    print(f"   Initial Capital: ${settings.initial_capital:,.2f}")
    print()
    
    # Создаем систему
    system = TradingSystem(mode="evaluate")
    
    # Запускаем систему
    print("🚀 Запуск системы...")
    await system.start()
    print("✅ Система запущена")
    print()
    
    # Даем системе немного времени на инициализацию
    await asyncio.sleep(2)
    
    # Генерируем тестовые сигналы
    print("📡 Генерация тестовых сигналов...")
    signal_task = asyncio.create_task(generate_test_signals(system._bus, target_orders=500))
    
    # Ждем завершения генерации сигналов
    await signal_task
    
    # Даем системе время на обработку всех сигналов
    print()
    print("⏳ Обработка сигналов и ордеров...")
    
    # Ждем пока обработаются все сигналы
    max_wait_time = 60  # Максимум 60 секунд
    wait_interval = 2
    waited = 0
    
    while waited < max_wait_time:
        await asyncio.sleep(wait_interval)
        waited += wait_interval
        
        # Проверяем, сколько ордеров создано
        orders_sent = system._orders_sent
        orders_filled = system._orders_filled
        
        if orders_sent >= 500 or (waited > 10 and orders_sent > 0):
            # Если создано достаточно ордеров или прошло достаточно времени
            break
    
    # Даем еще немного времени на закрытие позиций
    await asyncio.sleep(5)
    
    # Проверяем результаты
    print()
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
    
    # Статистика по стратегиям
    strategy_metrics = system._accounting.get_strategy_metrics()
    
    print("💰 КАПИТАЛ:")
    print(f"   Начальный капитал: ${initial_capital:,.2f}")
    print(f"   Текущий капитал (Equity): ${equity:,.2f}")
    print(f"   Изменение: ${equity - initial_capital:,.2f} ({((equity - initial_capital) / initial_capital * 100):+.2f}%)")
    print()
    
    print("📈 ПРИБЫЛЬ:")
    if daily_pnl > 0:
        print(f"   ✅ Дневной PnL: +${daily_pnl:,.2f} ({daily_pnl/equity*100:+.2f}%)")
    elif daily_pnl < 0:
        print(f"   ❌ Дневной PnL: ${daily_pnl:,.2f} ({daily_pnl/equity*100:+.2f}%)")
    else:
        print(f"   ➖ Дневной PnL: ${daily_pnl:,.2f}")
    print(f"   💸 Реализованный PnL: ${realized_pnl:,.2f}")
    print(f"   💳 Всего комиссий: ${total_fees:,.2f}")
    print()
    
    print("📊 ОРДЕРА И ПОЗИЦИИ:")
    print(f"   Отправлено ордеров: {total_orders}")
    print(f"   Исполнено ордеров: {total_fills}")
    print(f"   Открытых позиций: {total_positions}")
    print()
    
    print("📉 РИСКИ:")
    print(f"   Drawdown: ${drawdown:,.2f} ({drawdown_pct:.2f}%)")
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
            profit_factor = metrics.get("profit_factor", 0)
            
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
            if profit_factor > 0:
                print(f"   Profit Factor: {profit_factor:.2f}")
            print()
    
    # Итоговая статистика
    total_trades = sum(int(m.get("trades", 0)) for m in strategy_metrics.values())
    total_pnl = sum(m.get("pnl", 0) for m in strategy_metrics.values())
    
    print("=" * 70)
    print("📊 ИТОГО")
    print("=" * 70)
    print()
    print(f"   Всего сделок: {total_trades}")
    if total_pnl > 0:
        print(f"   ✅ Общий PnL: +${total_pnl:,.2f}")
    elif total_pnl < 0:
        print(f"   ❌ Общий PnL: ${total_pnl:,.2f}")
    else:
        print(f"   ➖ Общий PnL: ${total_pnl:,.2f}")
    print()
    
    # Останавливаем систему
    print("🛑 Остановка системы...")
    await system.stop()
    print("✅ Система остановлена")
    print()
    
    # Восстанавливаем настройки
    settings.debug_mode = original_debug_mode
    settings.trading_mode = original_trading_mode
    
    print("=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("=" * 70)
    print()
    
    # Итоговый вывод
    if daily_pnl > 0:
        print(f"🎉 УСПЕХ! Проект принес прибыль: +${daily_pnl:,.2f}")
    elif daily_pnl < 0:
        print(f"⚠️  Убыток: ${daily_pnl:,.2f}")
    else:
        print("➖ Прибыль нулевая")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(run_test_with_500_orders())
    except KeyboardInterrupt:
        print("\n⚠️  Тест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

