#!/usr/bin/env python3
"""Get trading results from Bybit testnet."""

import asyncio
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, "src")

from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient


async def get_trading_results() -> None:
    """Get and display trading results from Bybit."""
    print("=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТОРГОВЛИ С BYBIT TESTNET")
    print("=" * 60)
    print()

    # Check configuration
    if not settings.bybit_api_key or not settings.bybit_api_secret:
        print("❌ Ошибка: API ключи не настроены в .env файле")
        return

    print(f"🔑 API Key: {settings.bybit_api_key[:10]}...{settings.bybit_api_key[-5:]}")
    print(f"🌐 Testnet: {'Да' if settings.bybit_testnet else 'Нет'}")
    print(f"📈 Trading Mode: {settings.trading_mode}")
    print()

    client = BybitHTTPClient()

    try:
        # Connect to API
        print("🔌 Подключение к Bybit API...")
        await client.connect()
        print("✅ Подключено успешно!")
        print()

        # Get account info
        print("=" * 60)
        print("💰 ИНФОРМАЦИЯ О СЧЕТЕ")
        print("=" * 60)
        try:
            account_info = await client.get_account_info()
            accounts = account_info.get("list", [])

            if accounts:
                account = accounts[0]
                coins = account.get("coin", [])

                total_equity = 0.0
                total_available = 0.0

                for coin in coins:
                    if coin.get("coin") == "USDT":
                        equity_str = coin.get("equity", "0") or "0"
                        available_str = coin.get("availableToWithdraw", "0") or "0"
                        equity = float(equity_str) if equity_str else 0.0
                        available = float(available_str) if available_str else 0.0
                        total_equity = equity
                        total_available = available

                        print(f"💵 USDT Balance:")
                        print(f"   Equity: ${equity:,.2f}")
                        print(f"   Available: ${available:,.2f}")
                        print()
                        break

                if total_equity == 0:
                    print("⚠️  Баланс USDT: $0.00")
                    print("   (Возможно, нужно пополнить testnet счет)")
                    print()
            else:
                print("⚠️  Информация о счете не найдена")
                print()
        except Exception as e:
            print(f"❌ Ошибка получения информации о счете: {e}")
            print()

        # Get positions
        print("=" * 60)
        print("📊 ОТКРЫТЫЕ ПОЗИЦИИ")
        print("=" * 60)
        try:
            positions = await client.get_positions()

            if positions:
                total_unrealized_pnl = 0.0

                for pos in positions:
                    size = float(pos.get("size", 0))
                    if size == 0:
                        continue  # Skip closed positions

                    symbol = pos.get("symbol", "")
                    side = pos.get("side", "")
                    entry_price = float(pos.get("avgPrice", 0))
                    mark_price = float(pos.get("markPrice", 0))
                    unrealized_pnl = float(pos.get("unrealisedPnl", 0))
                    leverage = pos.get("leverage", "1")

                    total_unrealized_pnl += unrealized_pnl

                    print(f"📈 {symbol} {side.upper()}")
                    print(f"   Size: {size}")
                    print(f"   Entry Price: ${entry_price:,.2f}")
                    print(f"   Mark Price: ${mark_price:,.2f}")
                    print(f"   Unrealized PnL: ${unrealized_pnl:,.2f}")
                    print(f"   Leverage: {leverage}x")
                    print()

                if total_unrealized_pnl != 0:
                    print(f"💰 Total Unrealized PnL: ${total_unrealized_pnl:,.2f}")
                    print()
                else:
                    print("✅ Нет открытых позиций")
                    print()
            else:
                print("✅ Нет открытых позиций")
                print()
        except Exception as e:
            print(f"❌ Ошибка получения позиций: {e}")
            print()

        # Get order history (last 10 orders)
        print("=" * 60)
        print("📋 ИСТОРИЯ ОРДЕРОВ (последние 10)")
        print("=" * 60)
        try:
            # Get order history for BTCUSDT and ETHUSDT
            symbols = ["BTCUSDT", "ETHUSDT"]
            all_orders = []

            for symbol in symbols:
                try:
                    params = {
                        "category": "linear",
                        "symbol": symbol,
                        "limit": 10,
                    }
                    response = await client._request("GET", "/v5/order/history", params=params)
                    orders = response.get("list", [])
                    all_orders.extend(orders)
                except Exception:
                    continue

            if all_orders:
                # Sort by update time
                all_orders.sort(key=lambda x: int(x.get("updatedTime", 0)), reverse=True)
                all_orders = all_orders[:10]  # Last 10

                total_pnl = 0.0
                filled_count = 0

                for order in all_orders:
                    order_id = order.get("orderId", "")
                    symbol = order.get("symbol", "")
                    side = order.get("side", "")
                    order_type = order.get("orderType", "")
                    qty = order.get("qty", "0")
                    price = order.get("price", "0")
                    status = order.get("orderStatus", "")
                    exec_qty = order.get("cumExecQty", "0")
                    exec_price = order.get("avgPrice", "0")
                    exec_value = order.get("cumExecValue", "0")

                    # Parse timestamps
                    create_time = int(order.get("createdTime", 0))
                    update_time = int(order.get("updatedTime", 0))
                    create_dt = datetime.fromtimestamp(create_time / 1000)
                    update_dt = datetime.fromtimestamp(update_time / 1000)

                    print(f"📝 Order: {order_id[:8]}...")
                    print(f"   Symbol: {symbol}")
                    print(f"   Side: {side.upper()} | Type: {order_type}")
                    print(
                        f"   Qty: {qty} | Price: ${float(price):,.2f}"
                        if price != "0"
                        else f"   Qty: {qty}"
                    )
                    print(f"   Status: {status}")
                    print(
                        f"   Executed: {exec_qty} @ ${float(exec_price):,.2f}"
                        if exec_price != "0"
                        else f"   Executed: {exec_qty}"
                    )
                    print(f"   Created: {create_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Updated: {update_dt.strftime('%Y-%m-%d %H:%M:%S')}")

                    if status == "Filled":
                        filled_count += 1

                    print()

                print(f"📊 Статистика:")
                print(f"   Всего ордеров: {len(all_orders)}")
                print(f"   Заполнено: {filled_count}")
                print()
            else:
                print("✅ Нет истории ордеров")
                print()
        except Exception as e:
            print(f"❌ Ошибка получения истории ордеров: {e}")
            print()

        # Get trading statistics
        print("=" * 60)
        print("📈 ТОРГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        try:
            # Get closed PnL
            params = {
                "category": "linear",
                "limit": 100,
            }
            response = await client._request("GET", "/v5/position/closed-pnl", params=params)
            closed_pnl_list = response.get("list", [])

            if closed_pnl_list:
                total_closed_pnl = sum(float(pnl.get("closedPnl", 0)) for pnl in closed_pnl_list)
                win_count = sum(1 for pnl in closed_pnl_list if float(pnl.get("closedPnl", 0)) > 0)
                loss_count = sum(1 for pnl in closed_pnl_list if float(pnl.get("closedPnl", 0)) < 0)

                print(f"💰 Total Closed PnL: ${total_closed_pnl:,.2f}")
                print(f"✅ Прибыльных сделок: {win_count}")
                print(f"❌ Убыточных сделок: {loss_count}")
                if win_count + loss_count > 0:
                    win_rate = (win_count / (win_count + loss_count)) * 100
                    print(f"📊 Win Rate: {win_rate:.1f}%")
                print()
            else:
                print("✅ Нет закрытых позиций")
                print()
        except Exception as e:
            print(f"⚠️  Не удалось получить статистику закрытых позиций: {e}")
            print()

        print("=" * 60)
        print("✅ Завершено")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(get_trading_results())
