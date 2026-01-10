#!/usr/bin/env python3
"""Проверка статуса торговли: открытые позиции и прибыль."""

import asyncio
import sys
from typing import Any

try:
    import aiohttp
except ImportError:
    print("ERROR: Missing dependencies. Install with: pip install aiohttp")
    sys.exit(1)


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


async def check_trading_status() -> None:
    """Проверить статус торговли."""
    base_url = "http://localhost:8000"
    api_base = f"{base_url}/api"
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Статус торговли{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    async with aiohttp.ClientSession() as session:
        # 1. Проверить статус движка
        try:
            async with session.get(f"{api_base}/engine/status", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    status = await resp.json()
                    running = status.get("running", False)
                    equity = status.get("equity", 0.0)
                    daily_pnl = status.get("daily_pnl", 0.0)
                    initial_capital = status.get("initial_capital", 0.0)
                    trading_mode = status.get("trading_mode", "unknown")
                    is_live = status.get("is_live", False)
                    
                    print(f"{Colors.BOLD}Статус движка:{Colors.RESET}")
                    if running:
                        print(f"  {Colors.GREEN}✓ Запущен{Colors.RESET}")
                    else:
                        print(f"  {Colors.RED}✗ Остановлен{Colors.RESET}")
                    
                    print(f"  Режим: {trading_mode} ({'LIVE' if is_live else 'PAPER'})")
                    print(f"  Начальный капитал: ${initial_capital:,.2f}")
                    print(f"  Текущий капитал (equity): ${equity:,.2f}")
                    
                    # Рассчитать общую прибыль
                    total_profit = equity - initial_capital
                    profit_pct = (total_profit / initial_capital * 100) if initial_capital > 0 else 0.0
                    
                    if total_profit >= 0:
                        print(f"  {Colors.GREEN}Общая прибыль: ${total_profit:,.2f} ({profit_pct:+.2f}%){Colors.RESET}")
                    else:
                        print(f"  {Colors.RED}Общий убыток: ${total_profit:,.2f} ({profit_pct:.2f}%){Colors.RESET}")
                    
                    if daily_pnl >= 0:
                        print(f"  {Colors.GREEN}Дневная прибыль: ${daily_pnl:,.2f}{Colors.RESET}")
                    else:
                        print(f"  {Colors.RED}Дневной убыток: ${daily_pnl:,.2f}{Colors.RESET}")
                    
                else:
                    print(f"{Colors.RED}✗ Ошибка получения статуса: HTTP {resp.status}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ Ошибка получения статуса: {e}{Colors.RESET}")
        
        print()
        
        # 2. Получить открытые позиции
        try:
            async with session.get(f"{api_base}/orders/positions", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    positions = await resp.json()
                    
                    print(f"{Colors.BOLD}Открытые позиции:{Colors.RESET}")
                    if positions and len(positions) > 0:
                        print(f"  Всего открыто: {len(positions)} позиций\n")
                        
                        total_unrealized_pnl = 0.0
                        total_realized_pnl = 0.0
                        
                        for i, pos in enumerate(positions, 1):
                            symbol = pos.get("symbol", "N/A")
                            side = pos.get("side", "unknown")
                            size = pos.get("size", 0.0)
                            entry_price = pos.get("entry_price", 0.0)
                            unrealized_pnl = pos.get("unrealized_pnl", 0.0)
                            realized_pnl = pos.get("realized_pnl", 0.0)
                            position_id = pos.get("position_id", "N/A")
                            
                            total_unrealized_pnl += unrealized_pnl
                            total_realized_pnl += realized_pnl
                            
                            side_color = Colors.GREEN if side == "long" else Colors.RED
                            
                            print(f"  {i}. {symbol}")
                            print(f"     Направление: {side_color}{side.upper()}{Colors.RESET}")
                            print(f"     Размер: {size:.6f}")
                            print(f"     Цена входа: ${entry_price:,.2f}")
                            
                            if unrealized_pnl >= 0:
                                print(f"     {Colors.GREEN}Нереализованная прибыль: ${unrealized_pnl:,.2f}{Colors.RESET}")
                            else:
                                print(f"     {Colors.RED}Нереализованный убыток: ${unrealized_pnl:,.2f}{Colors.RESET}")
                            
                            if realized_pnl != 0:
                                if realized_pnl >= 0:
                                    print(f"     {Colors.GREEN}Реализованная прибыль: ${realized_pnl:,.2f}{Colors.RESET}")
                                else:
                                    print(f"     {Colors.RED}Реализованный убыток: ${realized_pnl:,.2f}{Colors.RESET}")
                            
                            print(f"     ID: {position_id}")
                            print()
                        
                        # Итого
                        print(f"  {Colors.BOLD}ИТОГО:{Colors.RESET}")
                        if total_unrealized_pnl >= 0:
                            print(f"    {Colors.GREEN}Общая нереализованная прибыль: ${total_unrealized_pnl:,.2f}{Colors.RESET}")
                        else:
                            print(f"    {Colors.RED}Общий нереализованный убыток: ${total_unrealized_pnl:,.2f}{Colors.RESET}")
                        
                        if total_realized_pnl != 0:
                            if total_realized_pnl >= 0:
                                print(f"    {Colors.GREEN}Общая реализованная прибыль: ${total_realized_pnl:,.2f}{Colors.RESET}")
                            else:
                                print(f"    {Colors.RED}Общий реализованный убыток: ${total_realized_pnl:,.2f}{Colors.RESET}")
                        
                        total_pnl = total_unrealized_pnl + total_realized_pnl
                        if total_pnl >= 0:
                            print(f"    {Colors.BOLD}{Colors.GREEN}ОБЩАЯ ПРИБЫЛЬ: ${total_pnl:,.2f}{Colors.RESET}")
                        else:
                            print(f"    {Colors.BOLD}{Colors.RED}ОБЩИЙ УБЫТОК: ${total_pnl:,.2f}{Colors.RESET}")
                    else:
                        print(f"  {Colors.YELLOW}Нет открытых позиций{Colors.RESET}")
                else:
                    print(f"{Colors.RED}✗ Ошибка получения позиций: HTTP {resp.status}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ Ошибка получения позиций: {e}{Colors.RESET}")
        
        print()
        
        # 3. Получить статистику по стратегиям
        try:
            async with session.get(f"{api_base}/analytics/performance", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    performance = await resp.json()
                    strategies = performance.get("strategies", {})
                    
                    if strategies:
                        print(f"{Colors.BOLD}Статистика по стратегиям:{Colors.RESET}\n")
                        
                        for strategy_id, metrics in strategies.items():
                            trades = metrics.get("trades_count", 0)
                            pnl = metrics.get("pnl", 0.0)
                            open_positions = metrics.get("open_positions", 0)
                            profit_factor = metrics.get("profit_factor", 0.0)
                            
                            print(f"  {Colors.BOLD}{strategy_id}:{Colors.RESET}")
                            print(f"    Всего сделок: {trades}")
                            print(f"    Открытых позиций: {open_positions}")
                            
                            if pnl >= 0:
                                print(f"    {Colors.GREEN}PnL: ${pnl:,.2f}{Colors.RESET}")
                            else:
                                print(f"    {Colors.RED}PnL: ${pnl:,.2f}{Colors.RESET}")
                            
                            if profit_factor >= 1.0:
                                print(f"    {Colors.GREEN}Profit Factor: {profit_factor:.2f}{Colors.RESET}")
                            else:
                                print(f"    {Colors.RED}Profit Factor: {profit_factor:.2f}{Colors.RESET}")
                            
                            print()
                    else:
                        print(f"{Colors.YELLOW}Нет данных по стратегиям{Colors.RESET}")
                else:
                    print(f"{Colors.RED}✗ Ошибка получения статистики: HTTP {resp.status}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ Ошибка получения статистики: {e}{Colors.RESET}")
        
        # 4. Получить ордера
        try:
            async with session.get(f"{api_base}/orders?status=all", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    orders = await resp.json()
                    
                    if orders:
                        filled_count = sum(1 for o in orders if o.get("status") == "filled")
                        open_count = sum(1 for o in orders if o.get("status") in ["pending", "placed"])
                        
                        print(f"{Colors.BOLD}Статистика ордеров:{Colors.RESET}")
                        print(f"  Всего ордеров: {len(orders)}")
                        print(f"  Исполнено (filled): {filled_count}")
                        print(f"  Открыто (pending/placed): {open_count}")
                        print()
        except Exception as e:
            pass  # Не критично
    
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        asyncio.run(check_trading_status())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Прервано пользователем{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
