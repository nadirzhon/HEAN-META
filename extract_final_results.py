#!/usr/bin/env python3
"""Извлечение результатов бэктеста из лога."""

import re
import sys

log_file = "backtest_30days_output.log"

print("=" * 70)
print("РЕЗУЛЬТАТЫ БЭКТЕСТА НА 30 ДНЕЙ")
print("=" * 70)
print()

try:
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1. Сигналы
    signals = sum(1 for l in lines if "Signal published" in l)
    print(f"📊 СИГНАЛОВ СГЕНЕРИРОВАНО: {signals:,}")
    print()

    # 2. Поиск BACKTEST REPORT
    report_found = False
    for i, line in enumerate(reversed(lines[-50000:])):
        if "BACKTEST REPORT" in line:
            report_found = True
            print("✅ ФИНАЛЬНЫЙ ОТЧЕТ:")
            print()
            start_idx = len(lines) - 50000 + (len(lines[-50000:]) - i - 1)
            for j in range(min(100, len(lines) - start_idx)):
                print(lines[start_idx + j].rstrip())
            break

    if not report_found:
        print("⚠️  Финальный отчет не найден")
        print()
        print("📈 ПРОМЕЖУТОЧНАЯ СТАТИСТИКА:")
        print()

        # Поиск статистики paper broker
        for line in reversed(lines[-500000:]):
            if "Paper broker fill stats" in line:
                try:
                    match = re.search(r"fill stats[^:]*:\s*({[^}]+})", line)
                    if match:
                        stats_str = match.group(1)
                        stats = eval(stats_str)
                        total_fills = stats.get("total_fills", 0)
                        maker_fills = stats.get("maker_fills", 0)
                        taker_fills = stats.get("taker_fills", 0)

                        print(f"✅ ОТКРЫТО ТОРГОВ (ОРДЕРОВ): {total_fills:,}")
                        print(f"   - Maker Fills: {maker_fills:,}")
                        print(f"   - Taker Fills: {taker_fills:,}")
                        print(f"   - Maker Fill Rate: {stats.get('maker_fill_rate_pct', 0):.2f}%")
                        break
                except:
                    pass

        # Поиск информации о прибыли
        print()
        print("💰 ПРИБЫЛЬ:")
        pnl_found = False

        for line in reversed(lines[-500000:]):
            if "total_pnl=" in line:
                match = re.search(r"total_pnl=([\d.-]+)", line)
                if match:
                    pnl = float(match.group(1))
                    print(f"   Total PnL: ${pnl:,.2f}")
                    pnl_found = True
                    break
            elif "Final Equity:" in line:
                match = re.search(r"Final Equity:\s*\$?([\d,]+\.?\d*)", line)
                if match:
                    equity = float(match.group(1).replace(",", ""))
                    initial = 10000.0  # Default
                    # Ищем initial equity
                    for l in lines:
                        if "Initial Equity:" in l:
                            m = re.search(r"Initial Equity:\s*\$?([\d,]+\.?\d*)", l)
                            if m:
                                initial = float(m.group(1).replace(",", ""))
                                break
                    pnl = equity - initial
                    return_pct = (pnl / initial * 100) if initial > 0 else 0
                    print(f"   Initial Equity: ${initial:,.2f}")
                    print(f"   Final Equity: ${equity:,.2f}")
                    print(f"   Total PnL: ${pnl:,.2f}")
                    print(f"   Total Return: {return_pct:.2f}%")
                    pnl_found = True
                    break

        if not pnl_found:
            print("   ⚠️  Данные о прибыли не найдены")
            print("   (Бэктест был остановлен до завершения расчета метрик)")

    print()
    print("=" * 70)

except Exception as e:
    print(f"Ошибка: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
