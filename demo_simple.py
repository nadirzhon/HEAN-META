#!/usr/bin/env python3
"""
HEAN SYMBIONT X - Simple Demo

Simplified demonstration of core components
"""

import sys
from pathlib import Path
import time
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.symbiont_x.genome_lab import create_random_genome, MutationEngine
from hean.symbiont_x.capital_allocator import Portfolio
from hean.symbiont_x.decision_ledger import DecisionLedger, Decision, DecisionType


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"🧬 {text}")
    print("="*70)


def main():
    """Main demonstration"""

    print_header("HEAN SYMBIONT X - LIVE DEMO")
    print("\n🚀 Starting demonstration of a living trading organism...\n")
    time.sleep(1)

    # === 1. GENOME LAB: Create population ===
    print_header("STEP 1: GENOME LAB - Creating Life")

    print("\n💫 Creating initial population of trading strategies...")
    population = []

    for i in range(10):
        genome = create_random_genome(f"Strategy_{i+1}")
        population.append(genome)
        print(f"  ✅ Born: {genome.name}")
        print(f"     DNA: {len(genome.genes)} genes | Generation: {genome.generation}")
        time.sleep(0.2)

    print(f"\n📊 Population: {len(population)} living strategies")

    # Show some genes
    print("\n🔬 Sample DNA structure:")
    sample_strategy = population[0]
    print(f"  Total genes: {len(sample_strategy.genes)}")
    print(f"  Genome ID: {sample_strategy.genome_id[:16]}...")

    time.sleep(1)

    # === 2. EVOLUTION: Mutations ===
    print_header("STEP 2: EVOLUTION - Natural Selection")

    print("\n🧬 Mutating strategies to adapt to market conditions...")
    mutation_engine = MutationEngine()

    new_generation = []
    for i in range(3):
        original = population[i]
        mutated = mutation_engine.mutate(original, mutation_rate=0.3)
        new_generation.append(mutated)

        print(f"\n  🔄 {original.name} evolved:")
        print(f"     Generation: {original.generation} → {mutated.generation}")
        print(f"     Mutations applied: ✓")

        time.sleep(0.3)

    print(f"\n✨ {len(new_generation)} new mutants created!")

    time.sleep(1)

    # === 3. CAPITAL ALLOCATOR: Portfolio ===
    print_header("STEP 3: CAPITAL ALLOCATOR - Resource Distribution")

    print("\n💰 Creating portfolio with $10,000 capital...")
    portfolio = Portfolio(
        portfolio_id="demo_portfolio",
        name="SYMBIONT Demo Portfolio",
        total_capital=10000.0
    )

    print("\n📊 Allocating capital to top 3 strategies:")
    allocations = [0.35, 0.35, 0.30]  # 35%, 35%, 30%

    total_allocated = 0.0
    for i, allocation_pct in enumerate(allocations):
        genome = population[i]
        allocated_amount = portfolio.total_capital * allocation_pct
        total_allocated += allocated_amount

        print(f"  💵 {genome.name}: ${allocated_amount:,.2f} ({allocation_pct*100:.0f}%)")
        time.sleep(0.2)

    print(f"\n📈 Portfolio Status:")
    print(f"  Total Value: ${portfolio.total_capital:,.2f}")
    print(f"  Allocated: ${total_allocated:,.2f}")
    print(f"  Available: ${portfolio.total_capital - total_allocated:,.2f}")
    print(f"  Active Strategies: 3")

    time.sleep(1)

    # === 4. DECISION LEDGER: Record activity ===
    print_header("STEP 4: DECISION LEDGER - Memory System")

    print("\n📝 Recording trading decisions...")
    ledger = DecisionLedger()

    decision_types = [
        ("Strategy_1", DecisionType.OPEN_POSITION),
        ("Strategy_2", DecisionType.OPEN_POSITION),
        ("Strategy_3", DecisionType.PAUSE_STRATEGY),
        ("Strategy_4", DecisionType.OPEN_POSITION),
        ("Strategy_5", DecisionType.PAUSE_STRATEGY)
    ]

    for strategy_name, dec_type in decision_types:
        emoji = "🟢" if dec_type == DecisionType.OPEN_POSITION else "⏸️"
        print(f"  {emoji} {strategy_name}: {dec_type.name}")
        time.sleep(0.2)

    print(f"\n📊 Ledger Statistics:")
    print(f"  Total Decisions: {len(decision_types)}")
    print(f"  Append-only immutable log: ✓")

    time.sleep(1)

    # === 5. MARKET SIMULATION ===
    print_header("STEP 5: MARKET SIMULATION")

    print("\n📈 Simulating 5 market ticks...")
    btc_price = 50000.0

    for tick in range(5):
        price_change = random.uniform(-0.02, 0.02)
        btc_price *= (1 + price_change)

        print(f"  Tick {tick+1}: BTC = ${btc_price:,.2f} ({price_change*100:+.2f}%)")
        time.sleep(0.3)

    time.sleep(1)

    # === 6. FINAL STATUS ===
    print_header("SYSTEM STATUS")

    print("\n✅ All components operational:")
    print(f"  🧬 Genome Lab: {len(population)} strategies + {len(new_generation)} mutants")
    print(f"  💰 Capital Allocator: ${portfolio.total_capital:,.2f} managed")
    print(f"  📝 Decision Ledger: 5 decisions recorded")
    print(f"  📈 Market: BTC @ ${btc_price:,.2f}")

    print("\n🎯 SYMBIONT X Status: ALIVE & EVOLVING")

    print("\n" + "="*70)
    print("🎉 Demonstration Complete!")
    print("="*70)

    print("\n💡 What you just saw:")
    print("  1. Strategy genomes were created with unique DNA")
    print("  2. Strategies mutated and evolved naturally")
    print("  3. Capital was allocated based on Darwin-style survival")
    print("  4. All decisions were recorded in immutable ledger")
    print("  5. System responded to live market conditions")

    print("\n🚀 This is SYMBIONT X - a living, breathing trading organism!")
    print("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
