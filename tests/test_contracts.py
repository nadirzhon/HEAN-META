"""Tests for core contracts and market context builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest

from hean.core.bus import Event, EventBus, EventType
from hean.core.context import ContextBuilder, MarketContext
from hean.core.contracts import AlphaModule, Diagnostics, IncomeStream, PriceFeed
from hean.core.regime import Regime, RegimeDetector
from hean.core.types import Signal, Tick
from hean.portfolio.decision_memory import DecisionMemory


class DummyPriceFeed(PriceFeed):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self._symbols = ["BTCUSDT", "ETHUSDT"]

    async def start(self, bus: Any | None = None) -> None:  # type: ignore[override]
        self.started = True

    async def stop(self) -> None:  # type: ignore[override]
        self.stopped = True

    def symbols(self) -> Sequence[str]:  # type: ignore[override]
        return list(self._symbols)


class DummyAlpha(AlphaModule):
    def __init__(self) -> None:
        self._id = "dummy_alpha"
        self._seen_ticks: list[Tick] = []
        self._seen_regimes: list[Regime] = []
        self._pending: list[Signal] = []

    @property  # type: ignore[override]
    def id(self) -> str:
        return self._id

    async def on_tick(self, tick: Tick) -> None:  # type: ignore[override]
        self._seen_ticks.append(tick)

    async def on_regime(self, symbol: str, regime: Regime) -> None:  # type: ignore[override]
        self._seen_regimes.append(regime)

    async def emit_signals(self) -> Sequence[Signal]:  # type: ignore[override]
        signals = list(self._pending)
        self._pending.clear()
        return signals


class DummyIncomeStream(IncomeStream):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.updated = 0

    async def start(self) -> None:  # type: ignore[override]
        self.started = True

    async def stop(self) -> None:  # type: ignore[override]
        self.stopped = True

    async def update(self) -> None:  # type: ignore[override]
        self.updated += 1

    def report(self) -> Mapping[str, Any]:  # type: ignore[override]
        return {
            "started": self.started,
            "stopped": self.stopped,
            "update_count": self.updated,
        }


class DummyDiagnostics(Diagnostics):
    def explain(
        self,
        metrics: Mapping[str, Any] | None,
        no_trade: Mapping[str, Any] | None,
        memory: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:  # type: ignore[override]
        return {
            "has_metrics": bool(metrics),
            "has_no_trade": bool(no_trade),
            "has_memory": bool(memory),
        }


@pytest.mark.asyncio
async def test_price_feed_contract_can_be_implemented() -> None:
    feed = DummyPriceFeed()

    assert isinstance(feed, PriceFeed)
    assert not feed.started

    # Simulate lifecycle
    await feed.start(None)
    await feed.stop()

    assert feed.started is True
    assert feed.stopped is True
    assert "BTCUSDT" in feed.symbols()


@pytest.mark.asyncio
async def test_alpha_module_contract_can_be_implemented() -> None:
    alpha = DummyAlpha()
    tick = Tick(
        symbol="BTCUSDT",
        price=50_000.0,
        timestamp=datetime.utcnow(),
        bid=49_995.0,
        ask=50_005.0,
    )

    assert isinstance(alpha, AlphaModule)
    assert alpha.id == "dummy_alpha"

    await alpha.on_tick(tick)
    await alpha.on_regime("BTCUSDT", Regime.NORMAL)

    assert len(alpha._seen_ticks) == 1
    assert len(alpha._seen_regimes) == 1

    # No signals by default
    signals = await alpha.emit_signals()
    assert isinstance(signals, Sequence)
    assert list(signals) == []


@pytest.mark.asyncio
async def test_income_stream_contract_can_be_implemented() -> None:
    stream = DummyIncomeStream()

    assert isinstance(stream, IncomeStream)
    assert not stream.started

    await stream.start()
    await stream.update()
    await stream.stop()

    report = stream.report()
    assert report["started"] is True
    assert report["stopped"] is True
    assert report["update_count"] == 1


def test_diagnostics_contract_can_be_implemented() -> None:
    diag = DummyDiagnostics()

    assert isinstance(diag, Diagnostics)

    out = diag.explain({"m": 1}, {"n": 2}, {"k": 3})
    assert out["has_metrics"] is True
    assert out["has_no_trade"] is True
    assert out["has_memory"] is True


@pytest.mark.asyncio
async def test_market_context_dataclass_and_builder_buckets() -> None:
    """ContextBuilder should produce a MarketContext that can be bucketed
    by DecisionMemory without errors and with sensible bucket prefixes.
    """

    assert is_dataclass(MarketContext)

    bus = EventBus()
    detector = RegimeDetector(bus)
    builder = ContextBuilder(regime_detector=detector)

    # Start the bus + detector so that get_regime/get_volatility behave normally.
    await bus.start()
    await detector.start()

    # Feed a small price history so the detector has some data.
    base_price = 50_000.0
    now = datetime.utcnow()
    symbol = "BTCUSDT"
    for i in range(60):
        price = base_price * (1 + 0.001 * i)
        tick = Tick(
            symbol=symbol,
            price=price,
            timestamp=now + timedelta(seconds=i),
            bid=price * 0.9999,
            ask=price * 1.0001,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))

    # Build a fresh tick and context snapshot.
    tick = Tick(
        symbol=symbol,
        price=base_price * 1.05,
        timestamp=now.replace(hour=13, minute=30, second=0, microsecond=0),
        bid=base_price * 1.0495,
        ask=base_price * 1.0505,
    )

    ctx = builder.build(tick)

    # Basic shape checks
    assert isinstance(ctx, MarketContext)
    assert ctx.symbol == symbol
    assert ctx.hour_utc == 13
    assert ctx.spread_bps is None or ctx.spread_bps >= 0.0

    # Context must be consumable by DecisionMemory's bucketing helpers.
    dm = DecisionMemory()
    context_key = dm.build_context(
        regime=ctx.regime or Regime.NORMAL,
        spread_bps=ctx.spread_bps,
        volatility=ctx.vol,
        timestamp=tick.timestamp,
    )

    regime_bucket, spread_bucket, vol_bucket, hour_bucket = context_key

    assert regime_bucket.startswith("regime:")
    assert spread_bucket.startswith("spread:")
    assert vol_bucket.startswith("vol:")
    assert hour_bucket.startswith("hour:")

    await detector.stop()
    await bus.stop()


