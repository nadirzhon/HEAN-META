#!/usr/bin/env python3
"""Get trading summary from running system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.config import settings
from hean.portfolio.accounting import PortfolioAccounting

def main():
    acc = PortfolioAccounting(settings.initial_capital)
    
    print("=" * 60)
    print("ИТОГИ ТОРГОВ")
    print("=" * 60)
    print()
    
    equity = acc.get_equity()
    drawdown, drawdown_pct = acc.get_drawdown(equity)
    daily_pnl = acc.get_daily_pnl(equity)
    
    print(f"💰 Начальный капитал: ${acc._initial_capital:.2f}")
    print(f"💵 Текущий капитал (Equity): ${equity:.2f}")
    print(f"📈 Изменение капитала: ${equity - acc._initial_capital:.2f} ({((equity - acc._initial_capital) / acc._initial_capital * 100):.2f}%)")
    print(f"💸 Реализованный PnL: ${acc._realized_pnl:.2f}")
    print(f"📊 Дневной PnL: ${daily_pnl:.2f}")
    print(f"💳 Всего комиссий: ${acc._total_fees:.2f}")
    print(f"📉 Drawdown: ${drawdown:.2f} ({drawdown_pct:.2f}%)")
    print()
    
    total_trades = sum(acc._strategy_trades.values())
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"📍 Открытых позиций: {len(acc._positions)}")
    print()
    
    if total_trades > 0:
        print("=" * 60)
        print("ПО СТРАТЕГИЯМ")
        print("=" * 60)
        print()
        
        for strategy_id in sorted(acc._strategy_trades.keys()):
            trades = acc._strategy_trades.get(strategy_id, 0)
            if trades == 0:
                continue
                
            pnl = acc._strategy_pnl.get(strategy_id, 0)
            wins = acc._strategy_wins.get(strategy_id, 0)
            losses = acc._strategy_losses.get(strategy_id, 0)
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            total_wins = sum(acc._strategy_wins.values())
            total_losses = sum(acc._strategy_losses.values())
            profit_factor = (total_wins / total_losses) if total_losses > 0 else (total_wins if total_wins > 0 else 0)
            
            print(f"📊 {strategy_id}:")
            print(f"   Сделок: {trades}")
            print(f"   PnL: ${pnl:.2f}")
            print(f"   Побед: {wins} ({win_rate:.1f}%)")
            print(f"   Поражений: {losses}")
            if total_losses > 0:
                print(f"   Profit Factor: {profit_factor:.2f}")
            print()
    else:
        print("⚠️  Сделок пока не было")
        print()
        print("Возможные причины:")
        print("- Сигналы блокируются системой защиты")
        print("- Недостаточно условий для входа")
        print("- Система только что запущена")
        print()
    
    # Get strategy metrics
    strategy_metrics = acc.get_strategy_metrics()
    if strategy_metrics:
        print("=" * 60)
        print("ДЕТАЛЬНАЯ СТАТИСТИКА")
        print("=" * 60)
        print()
        for strategy_id, metrics in strategy_metrics.items():
            print(f"📈 {strategy_id}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            print()
    
    print("=" * 60)

if __name__ == "__main__":
    main()

