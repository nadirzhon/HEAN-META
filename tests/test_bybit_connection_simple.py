#!/usr/bin/env python3
"""
Простой тест подключения к Bybit Testnet
Запустите этот скрипт чтобы проверить доступность API
"""

import urllib.request
import json
import sys

def test_connection():
    """Test Bybit Testnet API connection"""

    print("=" * 70)
    print("🔍 ТЕСТ ПОДКЛЮЧЕНИЯ К BYBIT TESTNET")
    print("=" * 70)
    print()

    # Test URL
    url = "https://api-testnet.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"

    print("📡 Подключаюсь к Bybit Testnet...")
    print(f"   URL: {url}")
    print()

    try:
        # Make request
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            if data.get('retCode') == 0:
                print("✅ ПОДКЛЮЧЕНИЕ УСПЕШНО!")
                print()

                # Extract ticker data
                ticker_list = data.get('result', {}).get('list', [])
                if ticker_list:
                    ticker = ticker_list[0]
                    price = float(ticker.get('lastPrice', 0))
                    volume = float(ticker.get('volume24h', 0))
                    change = float(ticker.get('price24hPcnt', 0)) * 100

                    print("📊 ДАННЫЕ ПОЛУЧЕНЫ:")
                    print(f"   Symbol: {ticker.get('symbol')}")
                    print(f"   Price: ${price:,.2f}")
                    print(f"   24h Change: {change:+.2f}%")
                    print(f"   24h Volume: {volume:,.2f}")
                    print()
                    print("=" * 70)
                    print("🎉 Bybit Testnet API работает отлично!")
                    print("=" * 70)
                    return True
                else:
                    print("⚠️  Нет данных в ответе")
                    return False
            else:
                print(f"❌ ОШИБКА API: {data.get('retMsg')}")
                print(f"   Код: {data.get('retCode')}")
                return False

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP ОШИБКА: {e.code} - {e.reason}")
        print()
        print("Возможные причины:")
        print("  • Нет интернет соединения")
        print("  • api-testnet.bybit.com недоступен")
        print("  • Firewall блокирует запрос")
        return False

    except urllib.error.URLError as e:
        print(f"❌ ОШИБКА ПОДКЛЮЧЕНИЯ: {e.reason}")
        print()
        print("Возможные причины:")
        print("  • Нет интернет соединения")
        print("  • DNS не может разрешить api-testnet.bybit.com")
        print("  • Proxy блокирует запрос")
        return False

    except Exception as e:
        print(f"❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
