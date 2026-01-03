#!/usr/bin/env python3
"""Получить РЕАЛЬНУЮ прибыль с Bybit API (не из логов!)."""

import asyncio
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "src")

from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


async def get_real_profit() -> None:
    """Получить реальную прибыль с Bybit API."""
    print("=" * 70)
    print("💰 РЕАЛЬНАЯ ПРИБЫЛЬ С BYBIT API")
    print("=" * 70)
    print()
    
    # Проверка режима
    print(f"📊 Режим работы:")
    print(f"   DRY_RUN: {settings.dry_run}")
    print(f"   trading_mode: {settings.trading_mode}")
    print(f"   is_live: {settings.is_live}")
    print(f"   bybit_testnet: {settings.bybit_testnet}")
    print()
    
    if not settings.is_live:
        print("⚠️  ВНИМАНИЕ: Система не в live режиме!")
        print("   Данные могут быть из симуляции, а не реальной торговли.")
        print()
    
    if not settings.bybit_api_key or not settings.bybit_api_secret:
        print("❌ Ошибка: API ключи не настроены")
        return
    
    client = BybitHTTPClient()
    
    try:
        print("🔌 Подключение к Bybit API...")
        await client.connect()
        print("✅ Подключено!")
        print()
        
        # Получить баланс счета
        print("=" * 70)
        print("💰 БАЛАНС СЧЕТА (РЕАЛЬНЫЙ)")
        print("=" * 70)
        print()
        
        account_info = await client.get_account_info()
        accounts = account_info.get("list", [])
        
        if not accounts:
            print("❌ Не удалось получить информацию о счете")
            return
        
        account = accounts[0]
        coins = account.get("coin", [])
        
        total_equity = 0.0
        total_available = 0.0
        total_wallet_balance = 0.0
        
        for coin in coins:
            if coin.get("coin") == "USDT":
                equity_str = coin.get("equity", "0") or "0"
                available_str = coin.get("availableToWithdraw", "0") or "0"
                wallet_balance_str = coin.get("walletBalance", "0") or "0"
                
                total_equity = float(equity_str) if equity_str else 0.0
                total_available = float(available_str) if available_str else 0.0
                total_wallet_balance = float(wallet_balance_str) if wallet_balance_str else 0.0
                
                print(f"💵 USDT:")
                print(f"   Equity (общий капитал): ${total_equity:,.2f}")
                print(f"   Wallet Balance: ${total_wallet_balance:,.2f}")
                print(f"   Available (доступно): ${total_available:,.2f}")
                print()
                break
        
        if total_equity == 0:
            print("⚠️  Баланс USDT: $0.00")
            print("   На счету нет средств для торговли.")
            print()
        
        # Получить открытые позиции
        print("=" * 70)
        print("📊 ОТКРЫТЫЕ ПОЗИЦИИ (РЕАЛЬНЫЕ)")
        print("=" * 70)
        print()
        
        positions = await client.get_positions()
        
        total_unrealized_pnl = 0.0
        active_positions = []
        
        if positions:
            
            for pos in positions:
                size = float(pos.get("size", 0))
                if size == 0:
                    continue
                
                symbol = pos.get("symbol", "")
                side = pos.get("side", "")
                entry_price = float(pos.get("avgPrice", 0))
                mark_price = float(pos.get("markPrice", 0))
                unrealized_pnl = float(pos.get("unrealisedPnl", 0))
                leverage = pos.get("leverage", "1")
                
                total_unrealized_pnl += unrealized_pnl
                active_positions.append({
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "unrealized_pnl": unrealized_pnl,
                })
                
                print(f"📈 {symbol} {side.upper()}:")
                print(f"   Size: {size:.6f}")
                print(f"   Entry: ${entry_price:,.2f}")
                print(f"   Mark Price: ${mark_price:,.2f}")
                if unrealized_pnl > 0:
                    print(f"   ✅ Unrealized PnL: +${unrealized_pnl:,.2f}")
                elif unrealized_pnl < 0:
                    print(f"   ❌ Unrealized PnL: ${unrealized_pnl:,.2f}")
                else:
                    print(f"   ➖ Unrealized PnL: ${unrealized_pnl:,.2f}")
                print(f"   Leverage: {leverage}x")
                print()
            
            if total_unrealized_pnl != 0:
                print(f"💰 Общий Unrealized PnL: ", end="")
                if total_unrealized_pnl > 0:
                    print(f"+${total_unrealized_pnl:,.2f}")
                else:
                    print(f"${total_unrealized_pnl:,.2f}")
                print()
        else:
            print("✅ Нет открытых позиций")
            print()
        
        # Получить закрытые позиции за сегодня
        print("=" * 70)
        print("📈 ЗАКРЫТЫЕ ПОЗИЦИИ ЗА СЕГОДНЯ")
        print("=" * 70)
        print()
        
        today = datetime.utcnow().date()
        today_start = int(datetime.combine(today, datetime.min.time()).timestamp() * 1000)
        
        try:
            params = {
                "category": "linear",
                "limit": 100,
                "startTime": today_start,
            }
            response = await client._request("GET", "/v5/position/closed-pnl", params=params)
            closed_pnl_list = response.get("list", [])
            
            if closed_pnl_list:
                today_pnl = 0.0
                win_count = 0
                loss_count = 0
                
                for pnl_data in closed_pnl_list:
                    closed_pnl = float(pnl_data.get("closedPnl", 0))
                    updated_time = int(pnl_data.get("updatedTime", 0))
                    updated_date = datetime.fromtimestamp(updated_time / 1000).date()
                    
                    # Только сегодняшние сделки
                    if updated_date == today:
                        today_pnl += closed_pnl
                        if closed_pnl > 0:
                            win_count += 1
                        elif closed_pnl < 0:
                            loss_count += 1
                
                print(f"📊 ПРИБЫЛЬ ЗА СЕГОДНЯ ({today.strftime('%d.%m.%Y')}):")
                if today_pnl > 0:
                    print(f"   ✅ Дневной PnL: +${today_pnl:,.2f}")
                elif today_pnl < 0:
                    print(f"   ❌ Дневной PnL: ${today_pnl:,.2f}")
                else:
                    print(f"   ➖ Дневной PnL: ${today_pnl:,.2f}")
                print()
                
                print(f"📈 Статистика за сегодня:")
                print(f"   Прибыльных сделок: {win_count}")
                print(f"   Убыточных сделок: {loss_count}")
                if win_count + loss_count > 0:
                    win_rate = (win_count / (win_count + loss_count)) * 100
                    print(f"   Win Rate: {win_rate:.1f}%")
                print()
            else:
                print("✅ Нет закрытых позиций за сегодня")
                print()
        except Exception as e:
            print(f"⚠️  Не удалось получить закрытые позиции: {e}")
            print()
        
        # Итоговая информация
        print("=" * 70)
        print("📊 ИТОГОВАЯ ИНФОРМАЦИЯ")
        print("=" * 70)
        print()
        
        print(f"💵 Текущий баланс (Equity): ${total_equity:,.2f}")
        print(f"💳 Доступно для вывода: ${total_available:,.2f}")
        
        if active_positions:
            print(f"📈 Открытых позиций: {len(active_positions)}")
        else:
            print(f"📈 Открытых позиций: 0")
        
        print()
        print("=" * 70)
        print("✅ Завершено")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(get_real_profit())

