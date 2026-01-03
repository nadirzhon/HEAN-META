"""Synthetic price feed for paper trading."""

import asyncio
import random
from datetime import datetime, timedelta

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, FundingRate, Tick
from hean.exchange.models import PriceFeed
from hean.logging import get_logger

logger = get_logger(__name__)


class SyntheticPriceFeed(PriceFeed):
    """Synthetic price feed that generates realistic market data."""

    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        """Initialize the synthetic price feed."""
        self._bus = bus
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._base_prices: dict[str, float] = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
        }
        self._current_prices: dict[str, float] = self._base_prices.copy()

    async def start(self) -> None:
        """Start the price feed."""
        self._running = True
        for symbol in self._symbols:
            task = asyncio.create_task(self._generate_ticks(symbol))
            self._tasks.append(task)
        task = asyncio.create_task(self._generate_funding())
        self._tasks.append(task)
        logger.info(f"Synthetic price feed started for {self._symbols}")

    async def stop(self) -> None:
        """Stop the price feed."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Synthetic price feed stopped")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to a symbol (already handled in start)."""
        if symbol not in self._symbols:
            self._symbols.append(symbol)
            if self._running:
                task = asyncio.create_task(self._generate_ticks(symbol))
                self._tasks.append(task)

    async def _generate_ticks(self, symbol: str) -> None:
        """Generate tick events for a symbol."""
        while self._running:
            try:
                # Random walk with mean reversion
                current_price = self._current_prices.get(
                    symbol, self._base_prices.get(symbol, 1000.0)
                )
                movement = random.gauss(0, 0.001)  # 0.1% volatility
                new_price = current_price * (1 + movement)

                # Mean revert slightly to base
                base_price = self._base_prices.get(symbol, current_price)
                new_price = new_price * 0.999 + base_price * 0.001

                self._current_prices[symbol] = new_price

                # Generate bid/ask with spread
                spread_pct = 0.0001  # 0.01% spread
                bid = new_price * (1 - spread_pct / 2)
                ask = new_price * (1 + spread_pct / 2)

                tick = Tick(
                    symbol=symbol,
                    price=new_price,
                    timestamp=datetime.utcnow(),
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

                await asyncio.sleep(1.0)  # 1 tick per second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating ticks for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _generate_funding(self) -> None:
        """Generate funding rate events periodically."""
        while self._running:
            try:
                await asyncio.sleep(8 * 3600)  # Every 8 hours

                for symbol in self._symbols:
                    # Simulate funding rate: typically between -0.01% and 0.01%
                    funding_rate = random.gauss(0, 0.0001)

                    funding = FundingRate(
                        symbol=symbol,
                        rate=funding_rate,
                        timestamp=datetime.utcnow(),
                        next_funding_time=datetime.utcnow() + timedelta(hours=8),
                    )

                    await self._bus.publish(
                        Event(
                            event_type=EventType.FUNDING,
                            data={"funding": funding},
                        )
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating funding: {e}", exc_info=True)
                await asyncio.sleep(3600)
