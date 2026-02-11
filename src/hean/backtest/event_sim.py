"""Event simulator for backtesting - generates synthetic market regimes."""

import asyncio
import random
from datetime import datetime, timedelta
from enum import Enum

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, FundingRate, Tick
from hean.exchange.models import PriceFeed
from hean.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime types."""

    RANGE = "range"
    TREND = "trend"
    IMPULSE = "impulse"
    HIGH_VOL = "high_vol"


class EventSimulator(PriceFeed):
    """Simulates market events for backtesting.

    Implements PriceFeed interface to allow injection into TradingSystem,
    making it interchangeable with BybitPriceFeed for evaluation mode.
    """

    def __init__(
        self,
        bus: EventBus | None,
        symbols: list[str],
        start_date: datetime,
        days: int,
    ) -> None:
        """Initialize the event simulator.

        Args:
            bus: EventBus instance (can be None if injected later via start())
            symbols: List of symbols to simulate
            start_date: Start date for simulation
            days: Number of days to simulate
        """
        self._bus = bus
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = start_date + timedelta(days=days)
        self._current_time = start_date
        self._base_prices: dict[str, float] = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
        }
        self._current_prices: dict[str, float] = self._base_prices.copy()
        self._regime = MarketRegime.RANGE
        self._regime_start = start_date
        self._regime_duration = timedelta(hours=6)  # Regime lasts 6 hours
        self._running = False

    async def start(self, bus: EventBus | None = None) -> None:
        """Start the event simulator.

        Args:
            bus: Optional EventBus to inject. If provided, replaces self._bus.
                 This allows TradingSystem to inject its EventBus after creation.
        """
        if bus is not None:
            self._bus = bus

        if self._bus is None:
            raise RuntimeError("EventBus must be provided either in __init__ or start()")

        self._running = True
        logger.info(f"[EventSimulator] Starting: {self._start_date} to {self._end_date}")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to a symbol (no-op - EventSimulator handles all configured symbols).

        This method is required by PriceFeed interface but is a no-op since
        EventSimulator already publishes to all symbols configured in __init__.
        """
        pass  # EventSimulator already publishes to all configured symbols

    async def stop(self) -> None:
        """Stop the event simulator."""
        logger.info(
            f"[EventSimulator] Stopping (current_time={self._current_time}, end_date={self._end_date})"
        )
        self._running = False
        logger.info("[EventSimulator] Stopped")

    async def run(self) -> None:
        """Run the simulation."""
        logger.info("[EventSimulator] run() called - starting simulation loop")
        tick_interval = timedelta(seconds=1)  # 1 tick per second
        tick_count = 0

        while self._running and self._current_time < self._end_date:
            # Update regime if needed
            if self._current_time - self._regime_start >= self._regime_duration:
                self._regime = random.choice(list(MarketRegime))
                self._regime_start = self._current_time
                self._regime_duration = timedelta(hours=random.uniform(4, 12))  # Random duration
                logger.debug(f"Regime changed to {self._regime}")

            # Generate ticks for each symbol
            for symbol in self._symbols:
                await self._generate_tick(symbol)

            # Generate funding events periodically (every 8 hours)
            if self._current_time.hour % 8 == 0 and self._current_time.minute == 0:
                for symbol in self._symbols:
                    await self._generate_funding(symbol)

            self._current_time += tick_interval
            tick_count += 1
            if tick_count % 1000 == 0:
                logger.debug(
                    f"[EventSimulator] Progress: {self._current_time} / {self._end_date} ({tick_count} ticks)"
                )
            # CRITICAL FIX: Remove sleep for backtesting - it makes 30 days take 7+ hours!
            # For backtesting, we want fast simulation, not real-time pacing.
            # However, we need to yield periodically to allow other tasks to run.
            # Yield every 100 ticks to prevent blocking the event loop completely.
            if tick_count % 100 == 0:
                await asyncio.sleep(0)  # Yield to event loop

        logger.info(
            f"[EventSimulator] Simulation loop completed: current_time={self._current_time}, end_date={self._end_date}, running={self._running}"
        )

    async def _generate_tick(self, symbol: str) -> None:
        """Generate a tick event for a symbol."""
        base_price = self._base_prices.get(symbol, 1000.0)
        current_price = self._current_prices.get(symbol, base_price)

        # Generate price movement based on regime
        if self._regime == MarketRegime.RANGE:
            # Oscillate around base price
            movement = random.gauss(0, 0.001)  # Small random walk
            new_price = current_price * (1 + movement)
            # Mean revert to base
            new_price = new_price * 0.99 + base_price * 0.01

        elif self._regime == MarketRegime.TREND:
            # Trending movement
            trend_direction = random.choice([-1, 1])
            movement = trend_direction * random.gauss(0.002, 0.001)
            new_price = current_price * (1 + movement)

        elif self._regime == MarketRegime.IMPULSE:
            # Impulse movements
            if random.random() < 0.1:  # 10% chance of impulse
                impulse = random.gauss(0, 0.01)  # 1% impulse
                new_price = current_price * (1 + impulse)
            else:
                movement = random.gauss(0, 0.0005)
                new_price = current_price * (1 + movement)

        else:  # HIGH_VOL
            # High volatility
            movement = random.gauss(0, 0.005)  # Higher volatility
            new_price = current_price * (1 + movement)

        # Ensure price doesn't go negative
        new_price = max(new_price, base_price * 0.5)

        self._current_prices[symbol] = new_price

        # Generate bid/ask with spread
        spread_pct = 0.0001  # 0.01% spread
        bid = new_price * (1 - spread_pct / 2)
        ask = new_price * (1 + spread_pct / 2)

        tick = Tick(
            symbol=symbol,
            price=new_price,
            timestamp=self._current_time,
            volume=random.uniform(100, 1000),
            bid=bid,
            ask=ask,
        )

        await self._bus.publish(
            Event(
                event_type=EventType.TICK,
                data={"tick": tick},
            )
        )

    async def _generate_funding(self, symbol: str) -> None:
        """Generate a funding rate event."""
        # Simulate funding rate: typically between -0.01% and 0.01%
        funding_rate = random.gauss(0, 0.0001)

        funding = FundingRate(
            symbol=symbol,
            rate=funding_rate,
            timestamp=self._current_time,
            next_funding_time=self._current_time + timedelta(hours=8),
        )

        await self._bus.publish(
            Event(
                event_type=EventType.FUNDING,
                data={"funding": funding},
            )
        )

    @property
    def current_time(self) -> datetime:
        """Get current simulation time."""
        return self._current_time
