#!/usr/bin/env python3
"""Получить полный отчет о прибыли и ордерах."""

import asyncio
import json
import sys
from datetime import datetime

sys.path.insert(0, "src")

from hean.api.engine_facade import EngineFacade
from hean.config import settings


async def get_trading_report() -> None:
    """Получить полный отчет о торговле."""
    print("=" * 80)
    print("📊 ОТЧЕТ О ПРИБЫЛИ И ОРДЕРАХ")
    print("=" * 80)
    print()
    
    facade = EngineFacade()
    
    # Получить статус
    status = await facade.get_status()
    
    print("=" * 80)
    print("💰 СТАТУС СИСТЕМЫ")
    print("=" * 80)
    print()
    
    print(f"📈 Статус: {status.get('status', 'unknown').upper()}")
    print(f"🔄 Режим торговли: {status.get('trading_mode', 'unknown').upper()}")
    print(f"🌐 Live режим: {'✅ ДА' if status.get('is_live') else '❌ НЕТ (Paper Trading)'}")
    print(f"🔒 Dry Run: {'✅ ДА' if status.get('dry_run') else '❌ НЕТ'}")
    print()
    
    initial_capital = status.get('initial_capital', 0)
    equity = status.get('equity', 0)
    daily_pnl = status.get('daily_pnl', 0)
    
    print(f"💵 Начальный капитал: ${initial_capital:,.2f}")
    print(f"💵 Текущий капитал (Equity): ${equity:,.2f}")
    
    if initial_capital > 0:
        total_return = ((equity - initial_capital) / initial_capital) * 100
        print(f"📊 Общая доходность: {total_return:+.2f}%")
        print(f"💰 Общая прибыль/убыток: ${equity - initial_capital:+,.2f}")
    
    print(f"📅 Дневной PnL: ${daily_pnl:+,.2f}")
    print()
    
    # Получить ордера
    print("=" * 80)
    print("📋 ОРДЕРА")
    print("=" * 80)
    print()
    
    try:
        orders = await facade.get_orders()
        total_orders = len(orders)
        filled_orders = [o for o in orders if o.get('status') == 'filled']
        pending_orders = [o for o in orders if o.get('status') in ['pending', 'open', 'new']]
        cancelled_orders = [o for o in orders if o.get('status') == 'cancelled']
        
        print(f"📊 Общая статистика:")
        print(f"   Всего ордеров: {total_orders}")
        print(f"   ✅ Заполнено: {len(filled_orders)}")
        print(f"   ⏳ В ожидании: {len(pending_orders)}")
        print(f"   ❌ Отменено: {len(cancelled_orders)}")
        print()
        
        if filled_orders:
            # Группировка по стратегиям
            strategy_stats = {}
            for order in filled_orders:
                strategy = order.get('strategy_id', 'unknown')
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'count': 0, 'buy': 0, 'sell': 0, 'total_volume': 0}
                strategy_stats[strategy]['count'] += 1
                if order.get('side') == 'buy':
                    strategy_stats[strategy]['buy'] += 1
                else:
                    strategy_stats[strategy]['sell'] += 1
                strategy_stats[strategy]['total_volume'] += order.get('size', 0) * order.get('price', 0)
            
            print("📈 Статистика по стратегиям:")
            for strategy, stats in strategy_stats.items():
                print(f"   {strategy}:")
                print(f"      Ордеров: {stats['count']} (Buy: {stats['buy']}, Sell: {stats['sell']})")
                print(f"      Объем: ${stats['total_volume']:,.2f}")
            print()
            
            # Последние 10 заполненных ордеров
            print("📝 Последние 10 заполненных ордеров:")
            print()
            print(f"{'Символ':<12} {'Сторона':<8} {'Размер':<15} {'Цена':<15} {'Стратегия':<20} {'Время':<20}")
            print("-" * 100)
            
            for order in filled_orders[-10:]:
                symbol = order.get('symbol', 'N/A')
                side = order.get('side', 'N/A').upper()
                size = order.get('size', 0)
                price = order.get('price', 0)
                strategy = order.get('strategy_id', 'N/A')
                timestamp = order.get('timestamp', '')
                
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            time_str = str(timestamp)
                    except:
                        time_str = str(timestamp)
                else:
                    time_str = 'N/A'
                
                print(f"{symbol:<12} {side:<8} {size:<15.6f} ${price:<14,.2f} {strategy:<20} {time_str:<20}")
            
            print("-" * 100)
            print()
        else:
            print("⚠️  Нет заполненных ордеров")
            print()
        
        if pending_orders:
            print(f"⏳ Ордера в ожидании ({len(pending_orders)}):")
            for order in pending_orders[:5]:
                print(f"   {order.get('symbol')} {order.get('side').upper()} | Size: {order.get('size', 0):.6f} | Status: {order.get('status')}")
            if len(pending_orders) > 5:
                print(f"   ... и еще {len(pending_orders) - 5} ордеров")
            print()
    
    except Exception as e:
        print(f"❌ Ошибка получения ордеров: {e}")
        print()
    
    # Получить позиции
    print("=" * 80)
    print("📊 ОТКРЫТЫЕ ПОЗИЦИИ")
    print("=" * 80)
    print()
    
    try:
        positions = await facade.get_positions()
        
        if positions:
            total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
            total_size = sum(abs(p.get('size', 0)) for p in positions)
            
            print(f"📊 Общая статистика:")
            print(f"   Открытых позиций: {len(positions)}")
            print(f"   Общий размер: {total_size:.6f}")
            print(f"   Общий Unrealized PnL: ${total_unrealized_pnl:+,.2f}")
            print()
            
            print(f"{'Символ':<12} {'Сторона':<8} {'Размер':<15} {'Entry':<15} {'Mark':<15} {'PnL':<15}")
            print("-" * 100)
            
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                side = pos.get('side', 'N/A').upper()
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                mark_price = pos.get('mark_price', 0)
                unrealized_pnl = pos.get('unrealized_pnl', 0)
                
                pnl_sign = "+" if unrealized_pnl > 0 else ""
                print(f"{symbol:<12} {side:<8} {size:<15.6f} ${entry_price:<14,.2f} ${mark_price:<14,.2f} ${pnl_sign}{unrealized_pnl:<14,.2f}")
            
            print("-" * 100)
            print()
        else:
            print("✅ Нет открытых позиций")
            print()
    
    except Exception as e:
        print(f"❌ Ошибка получения позиций: {e}")
        print()
    
    # Итоговая сводка
    print("=" * 80)
    print("📊 ИТОГОВАЯ СВОДКА")
    print("=" * 80)
    print()
    
    print(f"💵 Капитал:")
    print(f"   Начальный: ${initial_capital:,.2f}")
    print(f"   Текущий: ${equity:,.2f}")
    if initial_capital > 0:
        print(f"   Изменение: ${equity - initial_capital:+,.2f} ({((equity - initial_capital) / initial_capital * 100):+.2f}%)")
    print()
    
    try:
        orders = await facade.get_orders()
        filled = [o for o in orders if o.get('status') == 'filled']
        positions = await facade.get_positions()
        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in positions)
        
        print(f"📈 Торговля:")
        print(f"   Заполненных ордеров: {len(filled)}")
        print(f"   Открытых позиций: {len(positions)}")
        if total_unrealized != 0:
            print(f"   Unrealized PnL: ${total_unrealized:+,.2f}")
        print()
    except:
        pass
    
    print("=" * 80)
    print("✅ Отчет завершен")
    print("=" * 80)
    print()
    print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    asyncio.run(get_trading_report())
