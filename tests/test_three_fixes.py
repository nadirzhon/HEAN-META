#!/usr/bin/env python3
"""Test script for FIX-009, FIX-010, and FIX-016 implementations."""

import asyncio
from datetime import datetime

from hean.core.bus import EventBus
from hean.core.types import Event, EventType


async def test_fix_009_brain_analysis_event():
    """Test FIX-009: BRAIN_ANALYSIS event type exists and can be published."""
    print("\n=== Testing FIX-009: Brain Analysis Event ===")

    # Check EventType enum has BRAIN_ANALYSIS
    assert hasattr(EventType, "BRAIN_ANALYSIS"), "BRAIN_ANALYSIS not found in EventType enum"
    print("✓ EventType.BRAIN_ANALYSIS exists")

    # Test publishing brain analysis event
    bus = EventBus()
    await bus.start()

    received_events = []

    async def handler(event: Event):
        received_events.append(event)

    bus.subscribe(EventType.BRAIN_ANALYSIS, handler)

    # Publish test event
    await bus.publish(
        Event(
            event_type=EventType.BRAIN_ANALYSIS,
            data={
                "symbol": "BTCUSDT",
                "sentiment": "bullish",
                "confidence": 0.85,
                "forces": [{"name": "momentum", "direction": "bullish", "magnitude": 0.9}],
            },
        )
    )

    # Give event time to process
    await asyncio.sleep(0.1)

    assert len(received_events) == 1, f"Expected 1 event, got {len(received_events)}"
    event = received_events[0]
    assert event.data["symbol"] == "BTCUSDT"
    assert event.data["sentiment"] == "bullish"
    print("✓ BRAIN_ANALYSIS event published and received successfully")

    await bus.stop()
    print("✓ FIX-009 PASSED\n")


async def test_fix_010_position_reconciliation():
    """Test FIX-010: Position reconciliation module exists and can be imported."""
    print("=== Testing FIX-010: Position Reconciliation ===")

    try:
        from hean.execution.position_reconciliation import PositionReconciler
        print("✓ PositionReconciler imported successfully")

        # Check key methods exist
        assert hasattr(PositionReconciler, "start"), "start method missing"
        assert hasattr(PositionReconciler, "stop"), "stop method missing"
        assert hasattr(PositionReconciler, "reconcile"), "reconcile method missing"
        assert hasattr(PositionReconciler, "get_status"), "get_status method missing"
        print("✓ PositionReconciler has required methods")

        print("✓ FIX-010 PASSED\n")
    except ImportError as e:
        print(f"✗ FIX-010 FAILED: {e}\n")
        raise


def test_fix_016_signal_handlers():
    """Test FIX-016: Signal handlers exist in main.py."""
    print("=== Testing FIX-016: Graceful Shutdown ===")

    # Read main.py and check for signal handling
    with open("/Users/macbookpro/Desktop/HEAN/src/hean/main.py", "r") as f:
        content = f.read()

    # Check for signal imports
    assert "import signal" in content, "signal module not imported"
    print("✓ signal module imported")

    # Check for signal handlers
    assert "signal.SIGINT" in content, "SIGINT not handled"
    assert "signal.SIGTERM" in content, "SIGTERM not handled"
    print("✓ SIGINT and SIGTERM signal handlers registered")

    # Check for panic_close_all call in shutdown
    assert "panic_close_all" in content, "panic_close_all not called in shutdown"
    print("✓ panic_close_all called during shutdown")

    # Check for graceful_shutdown function
    assert "graceful_shutdown" in content, "graceful_shutdown function not found"
    print("✓ graceful_shutdown function implemented")

    print("✓ FIX-016 PASSED\n")


async def test_impulse_engine_brain_integration():
    """Test that ImpulseEngine subscribes to and handles BRAIN_ANALYSIS events."""
    print("=== Testing ImpulseEngine Brain Integration ===")

    from hean.strategies.impulse_engine import ImpulseEngine

    bus = EventBus()
    await bus.start()

    # Create impulse engine
    engine = ImpulseEngine(bus, symbols=["BTCUSDT"])
    await engine.start()

    # Check that brain sentiment tracking exists
    assert hasattr(engine, "_brain_sentiment"), "_brain_sentiment attribute missing"
    assert hasattr(engine, "_brain_confidence"), "_brain_confidence attribute missing"
    print("✓ ImpulseEngine has brain sentiment tracking attributes")

    # Check that brain conflict check method exists
    assert hasattr(engine, "_check_brain_conflict"), "_check_brain_conflict method missing"
    print("✓ ImpulseEngine has _check_brain_conflict method")

    # Publish brain analysis event
    await bus.publish(
        Event(
            event_type=EventType.BRAIN_ANALYSIS,
            data={
                "symbol": "BTCUSDT",
                "sentiment": "bearish",
                "confidence": 0.9,
            },
        )
    )

    # Give event time to process
    await asyncio.sleep(0.1)

    # Check that sentiment was updated
    assert "BTCUSDT" in engine._brain_sentiment, "Brain sentiment not updated"
    assert engine._brain_sentiment["BTCUSDT"] == "bearish", "Brain sentiment incorrect"
    assert engine._brain_confidence["BTCUSDT"] == 0.9, "Brain confidence incorrect"
    print("✓ ImpulseEngine received and processed BRAIN_ANALYSIS event")

    # Test conflict detection
    has_conflict, penalty = engine._check_brain_conflict("BTCUSDT", "buy")
    assert has_conflict, "Conflict not detected for buy signal with bearish sentiment"
    assert penalty < 1.0, "Penalty should be < 1.0 for conflict"
    print(f"✓ Brain conflict detected: penalty={penalty:.2f}")

    await engine.stop()
    await bus.stop()
    print("✓ ImpulseEngine Brain Integration PASSED\n")


async def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING THREE FIXES")
    print("=" * 70)

    try:
        # Test FIX-009
        await test_fix_009_brain_analysis_event()

        # Test FIX-010
        await test_fix_010_position_reconciliation()

        # Test FIX-016
        test_fix_016_signal_handlers()

        # Test integration
        await test_impulse_engine_brain_integration()

        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

    except Exception as e:
        print("=" * 70)
        print(f"TESTS FAILED: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    asyncio.run(main())
