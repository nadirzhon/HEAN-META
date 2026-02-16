"""Tests for ArchonReconciler."""

import asyncio

from hean.core.bus import EventBus


class MockAccounting:
    """Mock accounting for testing."""

    def __init__(self):
        self.positions = []

    def get_positions(self):
        return self.positions

    def get_equity(self):
        return 1000.0


class MockOrderManager:
    """Mock order manager for testing."""

    def get_open_orders(self):
        return []


class MockBybitHTTP:
    """Mock Bybit HTTP client for testing."""

    def __init__(self):
        self.positions = []
        self.balance = 1000.0
        self.orders = []
        self.fail_positions = False
        self.fail_balance = False
        self.fail_orders = False

    async def get_positions(self):
        if self.fail_positions:
            raise Exception("Failed to fetch positions")
        return self.positions

    async def get_wallet_balance(self):
        if self.fail_balance:
            raise Exception("Failed to fetch balance")
        return {"totalEquity": self.balance}

    async def get_open_orders(self):
        if self.fail_orders:
            raise Exception("Failed to fetch orders")
        return self.orders


async def test_reconciler_start_stop():
    """Test reconciler starts and stops cleanly."""
    from hean.archon.reconciler import ArchonReconciler

    bus = EventBus()
    await bus.start()

    accounting = MockAccounting()
    order_manager = MockOrderManager()
    bybit_http = MockBybitHTTP()

    reconciler = ArchonReconciler(
        bus=bus,
        accounting=accounting,
        order_manager=order_manager,
        bybit_http=bybit_http,
        interval_sec=1,
    )

    await reconciler.start()
    assert reconciler._running is True

    await asyncio.sleep(0.1)  # Let loops start

    await reconciler.stop()
    assert reconciler._running is False

    await bus.stop()


async def test_reconciler_no_discrepancies():
    """Test reconciler when local and exchange match."""
    from hean.archon.reconciler import ArchonReconciler

    bus = EventBus()
    await bus.start()

    accounting = MockAccounting()
    order_manager = MockOrderManager()
    bybit_http = MockBybitHTTP()

    reconciler = ArchonReconciler(
        bus=bus,
        accounting=accounting,
        order_manager=order_manager,
        bybit_http=bybit_http,
        interval_sec=1,
    )

    await reconciler.start()
    await asyncio.sleep(0.2)
    await reconciler.stop()

    status = reconciler.get_status()
    assert status["discrepancy_count"] == 0

    await bus.stop()


async def test_reconciler_exchange_error_handling():
    """Test reconciler handles exchange errors gracefully."""
    from hean.archon.reconciler import ArchonReconciler

    bus = EventBus()
    await bus.start()

    accounting = MockAccounting()
    order_manager = MockOrderManager()
    bybit_http = MockBybitHTTP()
    bybit_http.fail_positions = True
    bybit_http.fail_balance = True
    bybit_http.fail_orders = True

    reconciler = ArchonReconciler(
        bus=bus,
        accounting=accounting,
        order_manager=order_manager,
        bybit_http=bybit_http,
        interval_sec=1,
    )

    # Should not crash
    await reconciler.start()
    await asyncio.sleep(0.2)
    await reconciler.stop()

    # Still running despite errors
    status = reconciler.get_status()
    assert status["running"] is False  # Stopped

    await bus.stop()


async def test_reconciler_get_status():
    """Test reconciler status dict structure."""
    from hean.archon.reconciler import ArchonReconciler

    bus = EventBus()
    await bus.start()

    accounting = MockAccounting()
    order_manager = MockOrderManager()
    bybit_http = MockBybitHTTP()

    reconciler = ArchonReconciler(
        bus=bus,
        accounting=accounting,
        order_manager=order_manager,
        bybit_http=bybit_http,
        interval_sec=1,
    )

    status = reconciler.get_status()
    assert "running" in status
    assert "reconciliation_count" in status
    assert "discrepancy_count" in status
    assert "recent_discrepancies" in status
    assert "intervals" in status

    await bus.stop()
