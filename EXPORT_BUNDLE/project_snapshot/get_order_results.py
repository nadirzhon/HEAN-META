#!/usr/bin/env python3
"""Показать результаты ордеров из системы."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.core.types import OrderStatus
from hean.execution.order_manager import OrderManager
from hean.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


def format_order(order) -> str:
    """Форматировать информацию об ордере."""
    status_emoji = {
        "filled": "✅",
        "placed": "⏳",
        "pending": "⏸️",
        "partially_filled": "🔄",
        "cancelled": "❌",
        "rejected": "🚫",
    }
    emoji = status_emoji.get(str(order.status).lower(), "❓")
    
    lines = [
        f"{emoji} Order ID: {order.order_id}",
        f"   Стратегия: {order.strategy_id}",
        f"   Символ: {order.symbol}",
        f"   Сторона: {order.side}",
        f"   Размер: {order.size:.6f}",
        f"   Статус: {order.status.value if hasattr(order.status, 'value') else order.status}",
    ]
    
    if hasattr(order, 'price') and order.price:
        lines.append(f"   Цена: ${order.price:.2f}")
    
    if hasattr(order, 'filled_size') and order.filled_size:
        fill_pct = (order.filled_size / order.size * 100) if order.size > 0 else 0
        lines.append(f"   Заполнено: {order.filled_size:.6f} ({fill_pct:.1f}%)")
    
    if hasattr(order, 'avg_fill_price') and order.avg_fill_price:
        lines.append(f"   Средняя цена заполнения: ${order.avg_fill_price:.2f}")
    
    if hasattr(order, 'timestamp') and order.timestamp:
        lines.append(f"   Время: {order.timestamp.isoformat()}")
    
    return "\n".join(lines)


def show_order_results() -> None:
    """Показать результаты ордеров."""
    print("=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ОРДЕРОВ")
    print("=" * 70)
    print()
    
    # Создаем OrderManager (в реальной системе он должен быть из TradingSystem)
    # Для демонстрации создадим пустой менеджер
    order_manager = OrderManager()
    
    # Получаем все ордера
    all_orders = list(order_manager._orders.values()) if hasattr(order_manager, '_orders') else []
    
    if not all_orders:
        print("⚠️  Ордеров не найдено")
        print()
        print("Возможные причины:")
        print("- Система еще не выполняла торговлю")
        print("- Ордера хранятся в другом месте")
        print("- Система работает в режиме ожидания")
        print()
        print("💡 Для просмотра результатов торговли используйте:")
        print("   python get_trading_summary.py")
        print()
        return
    
    # Группируем по статусу
    filled_orders = [o for o in all_orders if o.status == OrderStatus.FILLED]
    open_orders = order_manager.get_open_orders()
    cancelled_orders = [o for o in all_orders if o.status == OrderStatus.CANCELLED]
    rejected_orders = [o for o in all_orders if o.status == OrderStatus.REJECTED]
    
    # Статистика
    print("📈 СТАТИСТИКА:")
    print(f"   Всего ордеров: {len(all_orders)}")
    print(f"   ✅ Заполнено: {len(filled_orders)}")
    print(f"   ⏳ Открыто: {len(open_orders)}")
    print(f"   ❌ Отменено: {len(cancelled_orders)}")
    print(f"   🚫 Отклонено: {len(rejected_orders)}")
    print()
    
    # Заполненные ордера
    if filled_orders:
        print("=" * 70)
        print(f"✅ ЗАПОЛНЕННЫЕ ОРДЕРА ({len(filled_orders)})")
        print("=" * 70)
        print()
        
        # Сортируем по времени (если есть)
        try:
            filled_orders_sorted = sorted(
                filled_orders,
                key=lambda o: o.timestamp if hasattr(o, 'timestamp') and o.timestamp else datetime.min,
                reverse=True
            )
        except:
            filled_orders_sorted = filled_orders
        
        for i, order in enumerate(filled_orders_sorted[:20], 1):  # Показываем последние 20
            print(f"{i}. {format_order(order)}")
            print()
        
        if len(filled_orders) > 20:
            print(f"... и еще {len(filled_orders) - 20} ордеров")
            print()
    
    # Открытые ордера
    if open_orders:
        print("=" * 70)
        print(f"⏳ ОТКРЫТЫЕ ОРДЕРА ({len(open_orders)})")
        print("=" * 70)
        print()
        
        for i, order in enumerate(open_orders, 1):
            print(f"{i}. {format_order(order)}")
            print()
    
    # Отмененные ордера
    if cancelled_orders:
        print("=" * 70)
        print(f"❌ ОТМЕНЕННЫЕ ОРДЕРА ({len(cancelled_orders)})")
        print("=" * 70)
        print()
        
        for i, order in enumerate(cancelled_orders[:10], 1):  # Показываем последние 10
            print(f"{i}. {format_order(order)}")
            print()
    
    # Отклоненные ордера
    if rejected_orders:
        print("=" * 70)
        print(f"🚫 ОТКЛОНЕННЫЕ ОРДЕРА ({len(rejected_orders)})")
        print("=" * 70)
        print()
        
        for i, order in enumerate(rejected_orders[:10], 1):  # Показываем последние 10
            print(f"{i}. {format_order(order)}")
            print()
    
    # Группировка по стратегиям
    if all_orders:
        print("=" * 70)
        print("📊 ПО СТРАТЕГИЯМ")
        print("=" * 70)
        print()
        
        from collections import defaultdict
        strategy_stats = defaultdict(lambda: {"total": 0, "filled": 0, "open": 0})
        
        for order in all_orders:
            strategy_id = order.strategy_id
            strategy_stats[strategy_id]["total"] += 1
            if order.status == OrderStatus.FILLED:
                strategy_stats[strategy_id]["filled"] += 1
            elif order.status in (OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED):
                strategy_stats[strategy_id]["open"] += 1
        
        for strategy_id, stats in sorted(strategy_stats.items()):
            print(f"🎯 {strategy_id}:")
            print(f"   Всего: {stats['total']}")
            print(f"   Заполнено: {stats['filled']}")
            print(f"   Открыто: {stats['open']}")
            print()
    
    # Группировка по символам
    if all_orders:
        print("=" * 70)
        print("📈 ПО СИМВОЛАМ")
        print("=" * 70)
        print()
        
        from collections import defaultdict
        symbol_stats = defaultdict(lambda: {"total": 0, "filled": 0})
        
        for order in all_orders:
            symbol = order.symbol
            symbol_stats[symbol]["total"] += 1
            if order.status == OrderStatus.FILLED:
                symbol_stats[symbol]["filled"] += 1
        
        for symbol, stats in sorted(symbol_stats.items()):
            print(f"💹 {symbol}:")
            print(f"   Всего: {stats['total']}")
            print(f"   Заполнено: {stats['filled']}")
            print()
    
    print("=" * 70)
    print("✅ Анализ завершен")
    print("=" * 70)
    print()
    print("💡 Примечание: Для получения полных результатов используйте:")
    print("   python get_trading_summary.py  # Общая статистика торговли")
    print("   python get_local_trading_results.py  # Результаты из логов")
    print()


if __name__ == "__main__":
    try:
        show_order_results()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

