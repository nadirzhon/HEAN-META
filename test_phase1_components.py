#!/usr/bin/env python3
"""Quick test script for Phase 1 components (RL Risk, Oracle, Allocator).

Tests each component in isolation to verify functionality.

Usage:
    python3 test_phase1_components.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger(__name__)


async def test_rl_risk_manager():
    """Test RL Risk Manager."""
    logger.info("=" * 60)
    logger.info("Testing RL Risk Manager")
    logger.info("=" * 60)

    try:
        from hean.risk.rl_risk_manager import RLRiskManager

        bus = EventBus()
        manager = RLRiskManager(
            bus=bus,
            model_path=None,  # Test rule-based fallback
            adjustment_interval=60,
            enabled=True,
        )

        await manager.start()
        logger.info("✅ RL Risk Manager started successfully")

        # Simulate some events
        await bus.publish(Event(
            event_type=EventType.EQUITY_UPDATE,
            data={"equity": 10000.0}
        ))

        await bus.publish(Event(
            event_type=EventType.PHYSICS_UPDATE,
            data={
                "symbol": "BTCUSDT",
                "temperature": 0.6,
                "entropy": 0.5,
                "phase": "markup",
            }
        ))

        await bus.publish(Event(
            event_type=EventType.REGIME_UPDATE,
            data={
                "symbol": "BTCUSDT",
                "volatility": 0.03,
            }
        ))

        # Wait a bit for processing
        await asyncio.sleep(1)

        # Get risk parameters
        params = manager.get_risk_parameters()
        logger.info(f"✅ Risk parameters: {params}")

        await manager.stop()
        logger.info("✅ RL Risk Manager test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ RL Risk Manager test FAILED: {e}", exc_info=True)
        return False


async def test_dynamic_oracle():
    """Test Dynamic Oracle Weighting."""
    logger.info("=" * 60)
    logger.info("Testing Dynamic Oracle Weighting")
    logger.info("=" * 60)

    try:
        from hean.core.intelligence.dynamic_oracle import DynamicOracleWeighting

        bus = EventBus()
        oracle = DynamicOracleWeighting(bus=bus)

        await oracle.start()
        logger.info("✅ Dynamic Oracle started successfully")

        # Simulate physics updates
        await bus.publish(Event(
            event_type=EventType.PHYSICS_UPDATE,
            data={
                "symbol": "BTCUSDT",
                "temperature": 0.4,
                "entropy": 0.6,
                "phase": "markup",
            }
        ))

        await bus.publish(Event(
            event_type=EventType.CONTEXT_UPDATE,
            data={"type": "oracle_predictions", "symbol": "BTCUSDT"}
        ))

        # Wait for weight updates
        await asyncio.sleep(2)

        # Get current weights
        weights = oracle.get_weights()
        logger.info(f"✅ Current weights: {weights}")

        # Test signal fusion
        fused = oracle.fuse_signals(
            tcn_signal=0.75,
            finbert_signal=0.50,
            brain_signal=0.60,
            min_confidence=0.6,
        )
        logger.info(f"✅ Fused signal: {fused}")

        await oracle.stop()
        logger.info("✅ Dynamic Oracle test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Dynamic Oracle test FAILED: {e}", exc_info=True)
        return False


async def test_strategy_allocator():
    """Test Strategy Allocator."""
    logger.info("=" * 60)
    logger.info("Testing Strategy Allocator")
    logger.info("=" * 60)

    try:
        from hean.strategies.manager import StrategyAllocator

        bus = EventBus()
        allocator = StrategyAllocator(
            bus=bus,
            initial_capital=10000.0,
            rebalance_interval=5,  # Quick rebalance for testing
        )

        await allocator.start()
        logger.info("✅ Strategy Allocator started successfully")

        # Register some strategies
        allocator.register_strategy("impulse_engine")
        allocator.register_strategy("funding_harvester")
        allocator.register_strategy("basis_arbitrage")
        logger.info("✅ Registered 3 strategies")

        # Simulate some position closures
        from hean.core.types import Position
        from datetime import datetime

        for i in range(3):
            position = Position(
                position_id=f"pos_{i}",
                symbol="BTCUSDT",
                side="long",
                size=0.01,
                entry_price=50000.0,
                current_price=50100.0,
                realized_pnl=10.0 * (i + 1),  # Increasing PnL
                strategy_id="impulse_engine",
            )

            await bus.publish(Event(
                event_type=EventType.POSITION_CLOSED,
                data={"position": position}
            ))

        # Update equity
        await bus.publish(Event(
            event_type=EventType.EQUITY_UPDATE,
            data={"equity": 10060.0}
        ))

        # Wait for rebalancing
        await asyncio.sleep(6)

        # Get allocations
        allocations = allocator.get_all_allocations()
        logger.info(f"✅ Capital allocations: {allocations}")

        # Get performance
        perf = allocator.get_performance("impulse_engine")
        if perf:
            logger.info(
                f"✅ ImpulseEngine performance: "
                f"WR={perf.get_win_rate():.2f} PF={perf.get_profit_factor():.2f}"
            )

        await allocator.stop()
        logger.info("✅ Strategy Allocator test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy Allocator test FAILED: {e}", exc_info=True)
        return False


async def test_component_registry():
    """Test Component Registry."""
    logger.info("=" * 60)
    logger.info("Testing Component Registry")
    logger.info("=" * 60)

    try:
        from hean.core.system.component_registry import ComponentRegistry

        bus = EventBus()
        registry = ComponentRegistry(bus=bus)

        # Initialize all components
        results = await registry.initialize_all(initial_capital=10000.0)
        logger.info(f"✅ Initialization results: {results}")

        # Start all components
        await registry.start_all()
        logger.info("✅ All components started")

        # Register strategies
        registry.register_strategies(["impulse_engine", "funding_harvester"])
        logger.info("✅ Strategies registered")

        # Wait a bit
        await asyncio.sleep(1)

        # Get status
        status = registry.get_status()
        logger.info(f"✅ Component status: {status}")

        # Test utility methods
        risk_params = registry.get_rl_risk_parameters()
        logger.info(f"✅ RL risk params: {risk_params}")

        oracle_weights = registry.get_oracle_weights()
        logger.info(f"✅ Oracle weights: {oracle_weights}")

        allocation = registry.get_strategy_allocation("impulse_engine")
        logger.info(f"✅ ImpulseEngine allocation: ${allocation:.2f}")

        # Stop all
        await registry.stop_all()
        logger.info("✅ Component Registry test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Component Registry test FAILED: {e}", exc_info=True)
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Phase 1 Component Tests")
    logger.info("")

    results = {
        "RL Risk Manager": await test_rl_risk_manager(),
        "Dynamic Oracle": await test_dynamic_oracle(),
        "Strategy Allocator": await test_strategy_allocator(),
        "Component Registry": await test_component_registry(),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{component}: {status}")

    all_passed = all(results.values())

    logger.info("")
    if all_passed:
        logger.info("🎉 All tests PASSED! Components are ready for integration.")
        return 0
    else:
        logger.error("⚠️  Some tests FAILED. Check logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
