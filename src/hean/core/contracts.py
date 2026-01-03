"""Core contracts and extension points for the HEAN engine.

This module defines *stable* abstract base classes (ABCs) that higher-level
components can depend on without coupling to concrete implementations.

These contracts are intentionally minimal and focused on behaviour rather
than configuration details so that new modules can be added without
breaking existing systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hean.core.regime import Regime
from hean.core.types import Signal, Tick


class PriceFeed(ABC):
    """Abstract interface for market data feeds.

    A `PriceFeed` is responsible for turning an external data source
    (exchange websocket, CSV, synthetic generator, etc.) into a stream
    of :class:`~hean.core.types.Tick` events, typically by publishing
    them to the event bus.

    Implementations SHOULD be side‑effectful (I/O, network, etc.) but the
    contract itself is deliberately small so callers can depend on it
    without caring about transport details.
    """

    @abstractmethod
    async def start(self, bus: Any | None = None) -> None:
        """Begin producing market data into the system.

        The optional ``bus`` argument allows feeds to integrate with the
        existing :class:`~hean.core.bus.EventBus` without importing it
        here, keeping this contract decoupled from the bus implementation.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop producing market data and release any external resources."""

    @abstractmethod
    def symbols(self) -> Iterable[str]:
        """Return the list of symbols this feed is currently providing."""


class AlphaModule(ABC):
    """Abstract interface for alpha‑generating modules.

    An ``AlphaModule`` consumes market/regime context and produces
    trading signals. It is deliberately higher‑level than a single
    strategy implementation so that multiple concrete strategies,
    model ensembles, or feature pipelines can conform to the same
    contract.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identifier for this alpha module."""

    @abstractmethod
    async def on_tick(self, tick: Tick) -> None:
        """Handle a new market tick.

        Implementations typically update internal state and possibly
        prepare signals that will later be emitted via :meth:`emit_signals`.
        """

    @abstractmethod
    async def on_regime(self, symbol: str, regime: Regime) -> None:
        """Handle a regime update for a specific symbol."""

    @abstractmethod
    async def emit_signals(self) -> Sequence[Signal]:
        """Return any signals that should be forwarded to the risk layer.

        Implementations may choose to clear their internal pending queue
        after emission or keep a rolling window – the contract does not
        prescribe a policy.
        """


class IncomeStream(ABC):
    """Abstract interface for strategy‑level income streams.

    An ``IncomeStream`` tracks realised cash‑flow style metrics such as
    funding income, basis carry, or other periodic payments. It can be
    started and stopped independently from the main trading loop.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start tracking and/or harvesting the income stream."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop tracking and release any background resources."""

    @abstractmethod
    async def update(self) -> None:
        """Advance internal state (e.g. on a timer or new data arrival)."""

    @abstractmethod
    def report(self) -> Mapping[str, Any]:
        """Return a snapshot of the current income metrics as a dictionary."""


class Diagnostics(ABC):
    """Abstract interface for producing human‑readable diagnostics.

    A diagnostics module takes in raw telemetry — metrics, no‑trade
    signals, decision memory state, etc. — and produces a structured
    explanation that can be surfaced in UIs, logs, or alerts.
    """

    @abstractmethod
    def explain(
        self,
        metrics: Mapping[str, Any] | None,
        no_trade: Mapping[str, Any] | None,
        memory: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Summarise the current system state into an explanation payload."""
