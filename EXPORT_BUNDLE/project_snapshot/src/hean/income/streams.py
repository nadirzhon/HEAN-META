"""Income streams layer (multi-income infrastructure).

Each income stream:
 - Listens to bus events (e.g. CONTEXT_UPDATE, CANDLE, FUNDING, TICK)
 - Emits standard `Signal` objects on the bus
 - Relies on the existing TradingSystem → RiskLimits → ExecutionRouter pipeline
 - Has independent capital / position budgets and can be disabled via config
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.regime import Regime
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamBudget:
    """Simple per-stream risk budget."""

    capital_pct: float
    max_positions: int


class IncomeStream(ABC):
    """Base class for all income streams.

    Implementations MUST:
      - Only emit `Signal` events on the bus
      - Let the normal risk / execution pipeline handle orders and fills
    """

    def __init__(self, stream_id: str, bus: EventBus, budget: StreamBudget) -> None:
        self.stream_id = stream_id
        self._bus = bus
        self._budget = budget
        self._running = False
        self._open_positions: int = 0

    # ------------------------------------------------------------------ Lifecycle
    async def start(self) -> None:
        """Start the stream and subscribe to core events."""
        if self._running:
            return
        self._running = True

        # Default subscriptions:
        # - CONTEXT_UPDATE: high‑level multi-symbol context feed
        # - CANDLE: timeframe candles
        # Implementations can override `_on_context_update` / `_on_candle`.
        self._bus.subscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        self._bus.subscribe(EventType.CANDLE, self._handle_candle)
        logger.info("Income stream %s started", self.stream_id)

    async def stop(self) -> None:
        """Stop the stream and unsubscribe."""
        if not self._running:
            return
        self._running = False
        self._bus.unsubscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        self._bus.unsubscribe(EventType.CANDLE, self._handle_candle)
        logger.info("Income stream %s stopped", self.stream_id)

    # ------------------------------------------------------------------ Event entrypoints
    async def _handle_context_update(self, event: Event) -> None:
        if not self._running:
            return
        await self.on_context_update(event)

    async def _handle_candle(self, event: Event) -> None:
        if not self._running:
            return
        await self.on_candle(event)

    # ------------------------------------------------------------------ Hooks for subclasses
    async def on_context_update(self, event: Event) -> None:
        """Handle CONTEXT_UPDATE events (override in subclass if used)."""
        # Default: no‑op
        return None

    async def on_candle(self, event: Event) -> None:
        """Handle CANDLE events (override in subclass if used)."""
        # Default: no‑op
        return None

    # ------------------------------------------------------------------ Budget helpers
    def can_open_position(self) -> bool:
        """Check if stream can open another position under its budget."""
        return self._open_positions < self._budget.max_positions

    def on_position_opened(self) -> None:
        self._open_positions += 1

    def on_position_closed(self) -> None:
        if self._open_positions > 0:
            self._open_positions -= 1

    # ------------------------------------------------------------------ Signal helper
    async def _emit_signal(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a Signal via the event bus.

        This is the *only* way streams participate in execution.
        """
        if not self._running:
            return

        if not self.can_open_position():
            logger.debug(
                "Income stream %s budget exhausted: open_positions=%d max=%d",
                self.stream_id,
                self._open_positions,
                self._budget.max_positions,
            )
            return

        # Attach basic budget metadata so risk/accounting can attribute PnL.
        md = {
            "income_stream": self.stream_id,
            "capital_pct": self._budget.capital_pct,
        }
        if metadata:
            md.update(metadata)

        signal = Signal(
            strategy_id=self.stream_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=md,
        )

        await self._bus.publish(
            Event(
                event_type=EventType.SIGNAL,
                data={"signal": signal},
            )
        )
        logger.debug(
            "Income stream %s emitted signal: %s %s @ %.4f",
            self.stream_id,
            side,
            symbol,
            entry_price,
        )


class FundingHarvesterStream(IncomeStream):
    """Perpetual funding capture using synthetic funding signal.

    Paper-mode first: we use CONTEXT_UPDATE payloads to infer funding bias.
    Expected `event.data` shape (loosely):
        {
          "symbol": "BTCUSDT",
          "funding_rate": float,
          "regime": Regime | str | None,
          ...
        }
    """

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        budget = StreamBudget(
            capital_pct=settings.stream_funding_capital_pct,
            max_positions=settings.stream_funding_max_positions,
        )
        super().__init__("stream_funding", bus, budget)
        self._symbols = symbols

    async def on_context_update(self, event: Event) -> None:
        ctx = event.data
        symbol: str | None = ctx.get("symbol")
        if symbol is None or symbol not in self._symbols:
            return

        rate = ctx.get("funding_rate")
        if rate is None:
            return

        # Simple deterministic rule: same sign logic as FundingHarvester strategy.
        threshold = 0.0001
        if abs(rate) < threshold:
            return

        side = "sell" if rate > 0 else "buy"
        # Paper-mode deterministic price anchor (these are synthetic anyway).
        price = ctx.get("price") or (50000.0 if "BTC" in symbol else 3000.0)
        stop = price * (0.98 if side == "buy" else 1.02)
        tp = price * (1.01 if side == "buy" else 0.99)

        await self._emit_signal(
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop,
            take_profit=tp,
            metadata={"funding_rate": rate},
        )


class MakerRebateStream(IncomeStream):
    """Post‑only liquidity harvesting.

    Uses CONTEXT_UPDATE to detect stable ranges and emits tiny maker-biased signals.
    This is infrastructure‑level; edge is elsewhere (ExecutionRouter maker-first).
    """

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        budget = StreamBudget(
            capital_pct=settings.stream_maker_rebate_capital_pct,
            max_positions=settings.stream_maker_rebate_max_positions,
        )
        super().__init__("stream_maker_rebate", bus, budget)
        self._symbols = symbols
        self._last_side: dict[str, str] = {}

    async def on_context_update(self, event: Event) -> None:
        ctx = event.data
        symbol: str | None = ctx.get("symbol")
        if symbol is None or symbol not in self._symbols:
            return

        regime = ctx.get("regime")
        if isinstance(regime, Regime):
            is_range = regime == Regime.RANGE
        else:
            is_range = str(regime).lower() == "range"

        if not is_range:
            return

        price = ctx.get("price")
        if price is None:
            price = 50000.0 if "BTC" in symbol else 3000.0

        # Alternate sides to harvest both maker rebates over time.
        prev = self._last_side.get(symbol, "sell")
        side = "buy" if prev == "sell" else "sell"
        self._last_side[symbol] = side

        # Very tight stop/TP; primary objective is fill / rebate, not direction.
        stop = price * (0.999 if side == "buy" else 1.001)
        tp = price * (1.001 if side == "buy" else 0.999)

        await self._emit_signal(
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop,
            take_profit=tp,
            metadata={"reason": "maker_rebate"},
        )


class BasisHedgeStream(IncomeStream):
    """Spot‑perp basis harvesting as an income stream.

    For now this is a thin, paper‑mode wrapper that relies on CONTEXT_UPDATE
    providing a synthetic `basis` field.
    """

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        budget = StreamBudget(
            capital_pct=settings.stream_basis_capital_pct,
            max_positions=settings.stream_basis_max_positions,
        )
        super().__init__("stream_basis_hedge", bus, budget)
        self._symbols = symbols

    async def on_context_update(self, event: Event) -> None:
        ctx = event.data
        symbol: str | None = ctx.get("symbol")
        if symbol is None or symbol not in self._symbols:
            return

        basis = ctx.get("basis")
        if basis is None:
            return

        threshold = 0.002  # 0.2%
        if abs(basis) < threshold:
            return

        price = ctx.get("price")
        if price is None:
            price = 50000.0 if "BTC" in symbol else 3000.0

        # Positive basis: lean short perp; negative basis: lean long perp.
        side = "sell" if basis > 0 else "buy"
        tp = price * (1.001 if side == "buy" else 0.999)
        stop = price * (0.995 if side == "buy" else 1.005)

        await self._emit_signal(
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop,
            take_profit=tp,
            metadata={"basis": basis},
        )


class VolatilityHarvestStream(IncomeStream):
    """Range mean‑reversion / short‑term volatility harvesting."""

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        budget = StreamBudget(
            capital_pct=settings.stream_volatility_capital_pct,
            max_positions=settings.stream_volatility_max_positions,
        )
        super().__init__("stream_volatility", bus, budget)
        self._symbols = symbols

    async def on_context_update(self, event: Event) -> None:
        ctx = event.data
        symbol: str | None = ctx.get("symbol")
        if symbol is None or symbol not in self._symbols:
            return

        # Expect contextual volatility info from upstream (e.g. RegimeDetector / trade_density).
        vol_short = ctx.get("vol_short")
        vol_long = ctx.get("vol_long")
        mean_price = ctx.get("mean_price")
        last_price = ctx.get("price")

        if last_price is None or mean_price is None:
            return

        # Simple mean‑reversion rule: buy below band, sell above band.
        band_width = 0.003  # 0.3% around mean
        lower = mean_price * (1 - band_width)
        upper = mean_price * (1 + band_width)

        if last_price < lower:
            side = "buy"
        elif last_price > upper:
            side = "sell"
        else:
            return

        stop = (
            mean_price * (1 - 2 * band_width)
            if side == "buy"
            else mean_price * (1 + 2 * band_width)
        )
        tp = mean_price

        await self._emit_signal(
            symbol=symbol,
            side=side,
            entry_price=last_price,
            stop_loss=stop,
            take_profit=tp,
            metadata={
                "vol_short": vol_short,
                "vol_long": vol_long,
                "mean_price": mean_price,
                "reason": "volatility_harvest",
            },
        )
