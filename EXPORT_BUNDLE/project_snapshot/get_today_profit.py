#!/usr/bin/env python3
"""Показать прибыль за сегодня."""

import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.config import settings
from hean.portfolio.accounting import PortfolioAccounting
from hean.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


def get_profit_from_logs() -> dict | None:
    """Попытаться получить прибыль из логов Docker."""
    import subprocess
    import re
    
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=100", "hean"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        logs = result.stdout
        
        # Ищем последнюю строку с equity
        equity_pattern = r'Equity: \$([\d.]+) \| Daily PnL: \$([\d.-]+)'
        matches = re.findall(equity_pattern, logs)
        
        if matches:
            equity_str, daily_pnl_str = matches[-1]
            return {
                "equity": float(equity_str),
                "daily_pnl": float(daily_pnl_str),
                "from_logs": True
            }
    except:
        pass
    
    return None


def show_today_profit() -> None:
    """Показать прибыль за сегодня."""
    print("=" * 70)
    print("💰 ПРИБЫЛЬ ЗА СЕГОДНЯ")
    print("=" * 70)
    print()
    
    today = date.today()
    print(f"📅 Дата: {today.strftime('%d.%m.%Y')}")
    print()
    
    # Пытаемся получить данные из логов Docker
    log_data = get_profit_from_logs()
    
    if log_data:
        print("📊 ДАННЫЕ ИЗ РАБОТАЮЩЕЙ СИСТЕМЫ:")
        print()
        equity = log_data["equity"]
        daily_pnl = log_data["daily_pnl"]
        
        print(f"   Текущий капитал (Equity): ${equity:,.2f}")
        print()
        
        print("📊 ПРИБЫЛЬ ЗА СЕГОДНЯ:")
        if daily_pnl > 0:
            print(f"   ✅ Дневной PnL: +${daily_pnl:,.2f} ({daily_pnl/equity*100:+.2f}%)")
        elif daily_pnl < 0:
            print(f"   ❌ Дневной PnL: ${daily_pnl:,.2f} ({daily_pnl/equity*100:+.2f}%)")
        else:
            print(f"   ➖ Дневной PnL: ${daily_pnl:,.2f}")
        print()
        
        # Общий PnL
        total_pnl = equity - settings.initial_capital
        total_pnl_pct = (total_pnl / settings.initial_capital * 100) if settings.initial_capital > 0 else 0
        
        print("📈 ОБЩАЯ СТАТИСТИКА:")
        if total_pnl > 0:
            print(f"   ✅ Общий PnL: +${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        elif total_pnl < 0:
            print(f"   ❌ Общий PnL: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        else:
            print(f"   ➖ Общий PnL: ${total_pnl:,.2f}")
        print(f"   Начальный капитал: ${settings.initial_capital:,.2f}")
        print()
        
        print("💡 Для детальной статистики используйте:")
        print("   python3 get_trading_summary.py")
        print()
        return
    
    # Если нет данных из логов, используем локальный учет
    # Создаем учет портфеля
    accounting = PortfolioAccounting(settings.initial_capital)
    
    # Получаем текущий equity
    equity = accounting.get_equity()
    
    # Получаем дневной PnL
    daily_pnl = accounting.get_daily_pnl(equity)
    
    # Получаем drawdown
    drawdown, drawdown_pct = accounting.get_drawdown(equity)
    
    # Получаем реализованный PnL
    realized_pnl = accounting._realized_pnl
    
    # Получаем общий PnL (от начального капитала)
    total_pnl = equity - settings.initial_capital
    total_pnl_pct = (total_pnl / settings.initial_capital * 100) if settings.initial_capital > 0 else 0
    
    # Получаем комиссии
    total_fees = accounting._total_fees
    
    # Статистика по стратегиям
    strategy_metrics = accounting.get_strategy_metrics()
    
    print("💵 ТЕКУЩЕЕ СОСТОЯНИЕ:")
    print(f"   Начальный капитал: ${settings.initial_capital:,.2f}")
    print(f"   Текущий капитал (Equity): ${equity:,.2f}")
    print()
    
    print("📊 ПРИБЫЛЬ ЗА СЕГОДНЯ:")
    if daily_pnl > 0:
        print(f"   ✅ Дневной PnL: +${daily_pnl:,.2f} ({daily_pnl/equity*100:+.2f}%)")
    elif daily_pnl < 0:
        print(f"   ❌ Дневной PnL: ${daily_pnl:,.2f} ({daily_pnl/equity*100:+.2f}%)")
    else:
        print(f"   ➖ Дневной PnL: ${daily_pnl:,.2f}")
    print()
    
    print("📈 ОБЩАЯ СТАТИСТИКА:")
    if total_pnl > 0:
        print(f"   ✅ Общий PnL: +${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    elif total_pnl < 0:
        print(f"   ❌ Общий PnL: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    else:
        print(f"   ➖ Общий PnL: ${total_pnl:,.2f}")
    print(f"   💸 Реализованный PnL: ${realized_pnl:,.2f}")
    print(f"   💳 Всего комиссий: ${total_fees:,.2f}")
    print(f"   📉 Drawdown: ${drawdown:,.2f} ({drawdown_pct:.2f}%)")
    print()
    
    # Статистика по стратегиям
    if strategy_metrics:
        print("=" * 70)
        print("📊 ПО СТРАТЕГИЯМ (СЕГОДНЯ)")
        print("=" * 70)
        print()
        
        total_trades_today = 0
        total_pnl_today = 0
        
        for strategy_id, metrics in sorted(strategy_metrics.items()):
            trades = metrics.get("trades", 0)
            pnl = metrics.get("pnl", 0)
            wins = metrics.get("wins", 0)
            losses = metrics.get("losses", 0)
            
            if trades == 0:
                continue
            
            win_rate = (wins / trades * 100) if trades > 0 else 0
            profit_factor = metrics.get("profit_factor", 0)
            
            total_trades_today += trades
            total_pnl_today += pnl
            
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
        
        if total_trades_today > 0:
            print("=" * 70)
            print("📊 ИТОГО ЗА СЕГОДНЯ:")
            print("=" * 70)
            print()
            print(f"   Всего сделок: {total_trades_today}")
            if total_pnl_today > 0:
                print(f"   ✅ Общий PnL: +${total_pnl_today:,.2f}")
            elif total_pnl_today < 0:
                print(f"   ❌ Общий PnL: ${total_pnl_today:,.2f}")
            else:
                print(f"   ➖ Общий PnL: ${total_pnl_today:,.2f}")
            print()
    else:
        print("⚠️  Сделок за сегодня не было")
        print()
        print("Возможные причины:")
        print("- Система только что запущена")
        print("- Сигналы блокируются системой защиты")
        print("- Недостаточно условий для входа")
        print()
    
    # Показываем открытые позиции
    positions = accounting._positions
    if positions:
        print("=" * 70)
        print(f"📈 ОТКРЫТЫЕ ПОЗИЦИИ ({len(positions)})")
        print("=" * 70)
        print()
        
        total_unrealized = 0
        for pos_id, position in positions.items():
            unrealized = position.unrealized_pnl
            total_unrealized += unrealized
            
            print(f"💹 {position.symbol} ({position.side}):")
            print(f"   Размер: {position.size:.6f}")
            print(f"   Цена входа: ${position.entry_price:,.2f}")
            print(f"   Текущая цена: ${position.current_price:,.2f}")
            if unrealized > 0:
                print(f"   ✅ Нереализованный PnL: +${unrealized:,.2f}")
            elif unrealized < 0:
                print(f"   ❌ Нереализованный PnL: ${unrealized:,.2f}")
            else:
                print(f"   ➖ Нереализованный PnL: ${unrealized:,.2f}")
            print()
        
        if total_unrealized != 0:
            print(f"💵 Общий нереализованный PnL: ", end="")
            if total_unrealized > 0:
                print(f"+${total_unrealized:,.2f}")
            else:
                print(f"${total_unrealized:,.2f}")
            print()
    
    print("=" * 70)
    print("✅ Отчет завершен")
    print("=" * 70)
    print()
    
    # Показываем время
    now = datetime.now()
    print(f"🕐 Время отчета: {now.strftime('%H:%M:%S')}")
    print()


if __name__ == "__main__":
    try:
        show_today_profit()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Примечание: Если система не запущена, используйте:")
        print("   python get_local_trading_results.py  # Результаты из логов")

