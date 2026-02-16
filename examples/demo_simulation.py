#!/usr/bin/env python3
"""
HEAN SYMBIONT X - Live Demo Simulation

Демонстрация всех компонентов системы в действии
"""

import sys
from pathlib import Path
import time
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.symbiont_x.genome_lab import create_random_genome, MutationEngine
from hean.symbiont_x.regime_brain import MarketRegime, RegimeClassifier
from hean.symbiont_x.capital_allocator import Portfolio
from hean.symbiont_x.decision_ledger import DecisionLedger, Decision, DecisionType
from hean.symbiont_x.nervous_system import EventEnvelope, EventType
from hean.symbiont_x.immune_system import RiskConstitution


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"🧬 {text}")
    print("="*70)


def simulate_market_cycle():
    """Simulate a complete market cycle"""

    print_header("HEAN SYMBIONT X - LIVE SIMULATION")
    print("\n📍 Initializing system components...")
    time.sleep(0.5)

    # === 1. GENOME LAB: Create population ===
    print_header("GENOME LAB: Creating Strategy Population")

    population_size = 10
    population = []

    for i in range(population_size):
        genome = create_random_genome(f"Strategy_{i+1}")
        population.append(genome)
        print(f"  ✅ {genome.name} created with {len(genome.genes)} genes")
        time.sleep(0.1)

    print(f"\n📊 Population: {len(population)} strategies initialized")

    # === 2. NERVOUS SYSTEM: Market events ===
    print_header("NERVOUS SYSTEM: Receiving Market Data")

    # Simulate market data
    btc_price = 50000.0
    events = []

    for i in range(5):
        # Simulate price movement
        price_change = random.uniform(-0.02, 0.02)  # ±2%
        btc_price *= (1 + price_change)

        event = EventEnvelope(
            event_type=EventType.TRADE,
            symbol="BTCUSDT",
            data={
                'price': btc_price,
                'volume': random.uniform(900, 1100),
                'change_pct': price_change * 100
            }
        )
        events.append(event)

        print(f"  📡 Tick {i+1}: BTC = ${btc_price:,.2f} ({price_change*100:+.2f}%)")
        time.sleep(0.2)

    # === 3. REGIME BRAIN: Classify market regime ===
    print_header("REGIME BRAIN: Detecting Market Regime")

    classifier = RegimeClassifier()

    # Simulate regime detection
    avg_volatility = sum(abs(e.data['change_pct']) for e in events) / len(events)

    if avg_volatility > 1.5:
        regime = MarketRegime.HIGH_VOL
    elif btc_price > 50000:
        regime = MarketRegime.TREND_UP
    elif btc_price < 50000:
        regime = MarketRegime.TREND_DOWN
    else:
        regime = MarketRegime.RANGE

    print(f"  🎯 Detected Regime: {regime.name}")
    print(f"  📊 Avg Volatility: {avg_volatility:.2f}%")
    print(f"  💰 Current Price: ${btc_price:,.2f}")

    # === 4. EVOLUTION: Mutate strategies ===
    print_header("GENOME LAB: Evolving Strategies")

    mutation_engine = MutationEngine()

    # Mutate top 3 strategies
    for i in range(3):
        original = population[i]
        mutated = mutation_engine.mutate(original, mutation_rate=0.3)

        print(f"  🧬 {original.name} → {mutated.name} (Generation {mutated.generation})")

        # Show one gene that changed
        for gene_name in original.genes:
            if abs(original.genes[gene_name] - mutated.genes[gene_name]) > 0.01:
                print(f"     Gene '{gene_name}': {original.genes[gene_name]:.2f} → {mutated.genes[gene_name]:.2f}")
                break

        time.sleep(0.2)

    # === 5. CAPITAL ALLOCATOR: Portfolio management ===
    print_header("CAPITAL ALLOCATOR: Managing Portfolio")

    portfolio = Portfolio(
        portfolio_id="demo_portfolio",
        name="SYMBIONT Demo Portfolio",
        total_capital=10000.0
    )

    # Add strategies to portfolio
    for i in range(min(3, len(population))):
        genome = population[i]
        allocation_pct = random.uniform(0.2, 0.4)  # 20-40% each

        portfolio.add_strategy(
            strategy_id=genome.genome_id,
            strategy_name=genome.name,
            allocated_capital=portfolio.total_capital * allocation_pct
        )

        print(f"  💰 {genome.name}: ${portfolio.total_capital * allocation_pct:,.2f} ({allocation_pct*100:.1f}%)")

    print(f"\n📊 Portfolio Value: ${portfolio.total_capital:,.2f}")
    print(f"📊 Allocated: ${portfolio.allocated_capital:,.2f}")
    print(f"📊 Available: ${portfolio.available_capital:,.2f}")

    # === 6. IMMUNE SYSTEM: Risk checks ===
    print_header("IMMUNE SYSTEM: Checking Risk Limits")

    constitution = RiskConstitution(
        max_position_size=1000.0,
        max_daily_loss=500.0,
        max_leverage=2.0
    )

    # Simulate trade validation
    test_trade = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 0.02,
        'price': btc_price
    }

    is_allowed = constitution.check_trade_allowed(test_trade)

    print(f"  📋 Trade Request: BUY 0.02 BTC @ ${btc_price:,.2f}")
    print(f"  ✅ Risk Check: {'PASSED' if is_allowed else 'REJECTED'}")
    print(f"  📊 Position Size: ${test_trade['quantity'] * btc_price:,.2f}")
    print(f"  📊 Max Allowed: ${constitution.max_position_size:,.2f}")

    # === 7. DECISION LEDGER: Record decisions ===
    print_header("DECISION LEDGER: Recording Decisions")

    ledger = DecisionLedger()

    # Record some decisions
    for i in range(3):
        decision = Decision(
            decision_id=f"decision_{i+1}",
            decision_type=random.choice([DecisionType.OPEN_POSITION, DecisionType.NO_ACTION]),
            reason=f"Regime: {regime.name}, Confidence: {random.uniform(0.6, 0.9):.2f}",
            strategy_name=population[i].name,
            symbol="BTCUSDT"
        )

        ledger.record_decision(decision)

        action_emoji = "🟢" if decision.decision_type == DecisionType.OPEN_POSITION else "⏸️"
        print(f"  {action_emoji} {decision.strategy_name}: {decision.decision_type.name}")
        print(f"     Reason: {decision.reason}")
        time.sleep(0.2)

    print(f"\n📊 Total Decisions: {ledger.total_decisions}")

    # === 8. FINAL SUMMARY ===
    print_header("SIMULATION COMPLETE")

    print("\n📊 System Status:")
    print(f"  🧬 Population Size: {len(population)} strategies")
    print(f"  🎯 Market Regime: {regime.name}")
    print(f"  💰 Portfolio Value: ${portfolio.total_capital:,.2f}")
    print(f"  📝 Decisions Recorded: {ledger.total_decisions}")
    print(f"  ✅ All Systems: OPERATIONAL")

    print("\n" + "="*70)
    print("🎉 SYMBIONT X is alive and evolving!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        simulate_market_cycle()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
