"""Trade density controller (anti-starvation) for per-strategy relaxation.

This module provides a lightweight controller that derives a per-strategy
relaxation level from trade density, and is intended to control SECONDARY
filters only (volatility and time windows).

Design goals:
    - Stateless wrapper per strategy (identified by strategy_id)
    - Uses global TradeDensityTracker state under the hood
    - Exposes:
        * idle_days: days since last trade (float)
        * relaxation_level: int in [0..3]
    - Relaxation rules (derived from idle_days):
        * 0: idle_days < 3      -> no relaxation
        * 1: 3 <= idle_days < 7 -> mild relaxation
        * 2: 7 <= idle_days < 14 -> medium relaxation
        * 3: idle_days >= 14    -> maximum relaxation

The exact mapping is deliberately simple and can be made configurable later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from hean.core.trade_density import trade_density


@dataclass
class DensityState:
    """Simple DTO for density state as seen by a strategy."""

    idle_days: float
    relaxation_level: int


class DensityController:
    """Per-strategy density controller exposing idle_days and relaxation_level.

    NOTE: This controller *does not* maintain its own trade history. Instead,
    it is a thin wrapper over the global ``trade_density`` tracker so that:

        - Execution layers (e.g., order fills) remain the single source of truth
          for when a "real" trade occurred.
        - Strategies can query a consistently derived relaxation_level that
          controls only SECONDARY filters (volatility/time), never risk or kill
          switches.
    """

    def __init__(self, strategy_id: str) -> None:
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> str:
        """Return the strategy identifier."""
        return self._strategy_id

    def record_trade(self, timestamp: datetime | None = None) -> None:
        """Record a trade for this strategy.

        This is a convenience wrapper around ``trade_density.record_trade``.
        In normal operation the execution layer already records trades, so
        strategies typically do not need to call this directly.
        """
        trade_density.record_trade(self._strategy_id, timestamp)

    def get_idle_days(self, current_time: datetime | None = None) -> float:
        """Return days since last trade for this strategy."""
        return trade_density.get_idle_days(self._strategy_id, current_time)

    def get_relaxation_level(self, current_time: datetime | None = None) -> int:
        """Return relaxation level in [0..3] based on idle_days.

        Level mapping (can be tuned later):
            - 0: idle_days < 3
            - 1: 3 <= idle_days < 7
            - 2: 7 <= idle_days < 14
            - 3: idle_days >= 14
        """
        idle_days = self.get_idle_days(current_time)

        if idle_days >= 14:
            return 3
        if idle_days >= 7:
            return 2
        if idle_days >= 3:
            return 1
        return 0

    def get_state(self, current_time: datetime | None = None) -> DensityState:
        """Return combined density state (idle_days + relaxation_level)."""
        idle_days = self.get_idle_days(current_time)
        level = self.get_relaxation_level(current_time)
        return DensityState(idle_days=idle_days, relaxation_level=level)
