"""
Пример использования HEAN SYMBIONT X

Демонстрирует как запустить живой организм для торговли
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hean.symbiont_x import HEANSymbiontX


async def main():
    """Главная функция"""

    # Конфигурация SYMBIONT X
    config = {
        # Market data
        'symbols': ['BTCUSDT', 'ETHUSDT'],

        # API credentials (замените на ваши)
        'bybit_api_key': 'YOUR_API_KEY',
        'bybit_api_secret': 'YOUR_API_SECRET',

        # Capital
        'initial_capital': 10000,  # $10,000

        # Evolution parameters
        'population_size': 50,     # 50 стратегий в популяции
        'elite_size': 5,           # Top 5 сохраняются без изменений
        'mutation_rate': 0.1,      # 10% mutation rate
        'crossover_rate': 0.3,     # 30% crossover rate

        # Capital allocation
        'allocation_method': 'survival_weighted',  # Darwinian allocation
        'rebalance_interval_hours': 24,            # Rebalance daily

        # Risk constitution (immutable rules)
        'risk_constitution': {
            'max_position_size_usd': 5000,    # Max $5K per position
            'max_position_size_pct': 15.0,    # Max 15% of capital
            'max_leverage': 3.0,              # Max 3x leverage
            'max_daily_loss_pct': 5.0,        # Max 5% loss per day
            'max_drawdown_pct': 20.0,         # Max 20% drawdown
        }
    }

    # Создаём SYMBIONT X
    print("🧬 Creating HEAN SYMBIONT X...")
    symbiont = HEANSymbiontX(
        config=config,
        storage_path="./symbiont_data"
    )

    # Запускаем
    await symbiont.start()

    # Главный цикл
    print("\n♥️  Entering main loop...")
    try:
        for generation in range(100):  # 100 generations
            print(f"\n{'=' * 60}")
            print(f"GENERATION {generation + 1}")
            print(f"{'=' * 60}")

            # Evolve one generation
            await symbiont.evolve_generation()

            # Show vital signs
            print(symbiont.get_vital_signs())

            # Wait before next generation
            await asyncio.sleep(3600)  # 1 hour per generation

    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted")

    finally:
        # Graceful shutdown
        await symbiont.stop()

        # Final status
        print("\n📊 Final Status:")
        status = symbiont.get_system_status()

        print(f"Uptime: {status['uptime_seconds'] / 3600:.1f} hours")
        print(f"Total Decisions: {status['decision_ledger']['total_decisions']}")
        print(f"Generation: {status['evolution']['generation_number']}")
        print(f"Portfolio Value: ${status['portfolio']['total_capital'] + status['portfolio']['total_pnl']:.2f}")
        print(f"ROI: {status['portfolio']['roi_pct']:.2f}%")

        print("\n👋 Goodbye")


if __name__ == "__main__":
    asyncio.run(main())
