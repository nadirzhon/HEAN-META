#!/usr/bin/env python3
"""Get trading results from local system logs and accounting."""

import sys
import re
from datetime import datetime

sys.path.insert(0, "src")

from hean.config import settings

def parse_logs_for_results() -> None:
    """Parse Docker logs for trading results."""
    import subprocess
    
    print("=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТОРГОВЛИ (из логов системы)")
    print("=" * 60)
    print()
    
    # Get recent equity logs
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=500", "hean"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        logs = result.stdout
        
        # Extract equity information
        equity_pattern = r'Equity: \$([\d.]+) \| Daily PnL: \$([\d.-]+) \| Profit: \$([\d.]+)/\$([\d.]+) \(([\d.]+)%\) \| Trades: (\d+) \| Drawdown: ([\d.]+)%'
        equity_matches = re.findall(equity_pattern, logs)
        
        if equity_matches:
            # Get last match (most recent)
            last_match = equity_matches[-1]
            equity, daily_pnl, profit, target, profit_pct, trades, drawdown = last_match
            
            print("💰 ТЕКУЩЕЕ СОСТОЯНИЕ:")
            print(f"   Equity: ${float(equity):,.2f}")
            print(f"   Daily PnL: ${float(daily_pnl):,.2f}")
            print(f"   Profit: ${float(profit):,.2f} / ${float(target):,.2f} ({float(profit_pct):.1f}%)")
            print(f"   Всего сделок: {trades}")
            print(f"   Drawdown: {float(drawdown):.2f}%")
            print()
            
            # Calculate initial capital
            initial_capital = settings.initial_capital
            total_pnl = float(equity) - initial_capital
            total_pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
            
            print("📈 ОБЩАЯ СТАТИСТИКА:")
            print(f"   Начальный капитал: ${initial_capital:,.2f}")
            print(f"   Текущий капитал: ${float(equity):,.2f}")
            print(f"   Общий PnL: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
            print()
        else:
            print("⚠️  Не найдено информации о equity в логах")
            print()
        
        # Extract order information
        order_pattern = r'order|Order|ORDER'
        order_count = len(re.findall(order_pattern, logs, re.IGNORECASE))
        
        filled_pattern = r'filled|Filled|FILLED'
        filled_count = len(re.findall(filled_pattern, logs, re.IGNORECASE))
        
        if order_count > 0:
            print("📋 ОРДЕРА:")
            print(f"   Упоминаний ордеров в логах: {order_count}")
            print(f"   Упоминаний заполнений: {filled_count}")
            print()
        
        # Extract position information
        position_pattern = r'position|Position|POSITION'
        position_count = len(re.findall(position_pattern, logs, re.IGNORECASE))
        
        if position_count > 0:
            print("📊 ПОЗИЦИИ:")
            print(f"   Упоминаний позиций в логах: {position_count}")
            print()
        
        # Extract strategy information
        strategies = {}
        for strategy in ["impulse", "funding", "basis"]:
            pattern = rf'{strategy}.*signal|{strategy}.*order'
            matches = len(re.findall(pattern, logs, re.IGNORECASE))
            if matches > 0:
                strategies[strategy] = matches
        
        if strategies:
            print("🎯 СТРАТЕГИИ:")
            for strategy, count in strategies.items():
                print(f"   {strategy.capitalize()}: {count} упоминаний")
            print()
        
        # Extract errors
        error_pattern = r'ERROR|Error|error'
        errors = re.findall(error_pattern, logs)
        if errors:
            print(f"⚠️  Найдено {len(errors)} ошибок в логах")
            print()
        
        print("=" * 60)
        print("✅ Анализ завершен")
        print("=" * 60)
        print()
        print("💡 Примечание: Для получения полных результатов с Bybit testnet")
        print("   необходимо проверить API ключи и их разрешения.")
        print()
        
    except subprocess.TimeoutExpired:
        print("❌ Таймаут при получении логов")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parse_logs_for_results()

