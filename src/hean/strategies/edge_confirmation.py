"""Edge confirmation loop for 2-step impulse entries.

Pattern:
1) First qualifying impulse becomes a *candidate* (no signal emitted).
2) Within a short timeout, a second qualifying impulse must *confirm* via at
   least one of:
   - Spread tightening
   - Volatility expansion
   - Micro pullback then resume in the direction of the trade.

This module is intentionally small and deterministic so it can be unit-tested
with crafted ticks and price histories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hean.config import settings
from hean.core.types import Signal, Tick


@dataclass
class CandidateState:
    """State for a single candidate edge per (strategy, symbol, side)."""

    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    created_at: datetime  # Timestamp of the candidate tick
    expires_at: datetime
    entry_price: float
    spread_bps: float | None
    vol_short: float | None
    vol_long: float | None
    return_pct: float
    price_index: int  # Index of the candidate price in price history
    signal: Signal


class EdgeConfirmationLoop:
    """Per-strategy edge confirmation loop for impulse-style entries."""

    def __init__(
        self,
        timeout_sec: int = 5,
        spread_tightening_ratio: float = 0.7,
        pullback_min_pct: float = 0.0005,  # 0.05%
        pullback_max_pct: float = 0.003,  # 0.3%
    ) -> None:
        """
        Args:
            timeout_sec: Max time between candidate and confirmation.
            spread_tightening_ratio: Required fraction of initial spread
                for spread-tightening confirmation (e.g., 0.7 means 30%
                tighter than the candidate spread).
            pullback_min_pct: Minimum adverse excursion relative to
                candidate entry to count as a "micro" pullback.
            pullback_max_pct: Maximum adverse excursion to still be
                considered a healthy micro pullback (beyond this it is
                treated as invalidation rather than confirmation).
        """
        self._timeout = timedelta(seconds=timeout_sec)
        self._spread_tightening_ratio = spread_tightening_ratio
        self._pullback_min_pct = pullback_min_pct
        self._pullback_max_pct = pullback_max_pct

        # Keyed by (strategy_id, symbol, side)
        self._candidates: dict[tuple[str, str, str], CandidateState] = {}

    def _key(self, signal: Signal) -> tuple[str, str, str]:
        return (signal.strategy_id, signal.symbol, signal.side)

    def clear(self) -> None:
        """Clear all candidates (mainly for tests)."""
        self._candidates.clear()

    def confirm_or_update(
        self,
        signal: Signal,
        tick: Tick,
        context: dict[str, Any],
        prices: list[float],
    ) -> Signal | None:
        """Register or confirm a candidate.

        This method is intended to be called only when the upstream
        impulse logic has already decided that `signal` is tradable
        (i.e., it has passed all filters and edge checks).

        On first call for a (strategy, symbol, side) key, a candidate
        is stored and None is returned.

        On subsequent calls within the timeout, the method checks for
        confirmation via:
            - Spread tightening
            - Volatility expansion
            - Micro pullback then resume

        If any condition is satisfied, the stored candidate is returned
        (the original signal). Otherwise, None is returned and the
        candidate is left in place (or replaced if expired).
        """
        key = self._key(signal)
        now: datetime = tick.timestamp

        spread_bps: float | None = context.get("spread_bps")
        vol_short: float | None = context.get("vol_short")
        vol_long: float | None = context.get("vol_long")
        return_pct: float = float(context.get("return_pct", 0.0))

        existing = self._candidates.get(key)
        if existing is None:
            # First qualifying impulse becomes candidate.
            self._candidates[key] = CandidateState(
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                side=signal.side,
                created_at=now,
                expires_at=now + self._timeout,
                entry_price=signal.entry_price,
                spread_bps=spread_bps,
                vol_short=vol_short,
                vol_long=vol_long,
                return_pct=return_pct,
                price_index=len(prices) - 1 if prices else 0,
                signal=signal,
            )
            return None

        # If candidate is expired, replace with a new candidate.
        if now > existing.expires_at:
            self._candidates[key] = CandidateState(
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                side=signal.side,
                created_at=now,
                expires_at=now + self._timeout,
                entry_price=signal.entry_price,
                spread_bps=spread_bps,
                vol_short=vol_short,
                vol_long=vol_long,
                return_pct=return_pct,
                price_index=len(prices) - 1 if prices else 0,
                signal=signal,
            )
            return None

        # Attempt confirmation using the existing candidate and the
        # current impulse context.
        if self._is_confirmed(existing, tick, spread_bps, vol_short, vol_long, prices):
            # Once confirmed, remove candidate and emit original signal.
            self._candidates.pop(key, None)
            # Annotate metadata for observability.
            if existing.signal.metadata is None:
                existing.signal.metadata = {}
            existing.signal.metadata.setdefault("edge_confirmation", "confirmed")
            return existing.signal

        # Still within timeout, but not confirmed yet – keep candidate.
        return None

    def _is_confirmed(
        self,
        candidate: CandidateState,
        tick: Tick,
        spread_bps: float | None,
        vol_short: float | None,
        vol_long: float | None,
        prices: list[float],
    ) -> bool:
        """Check confirmation conditions."""
        # 0) Hard invalidation: adverse move larger than allowed micro pullback.
        #    In this case we do NOT allow confirmation by any other mechanism.
        if prices and 0 <= candidate.price_index < len(prices):
            sub_prices = prices[candidate.price_index :]
            if len(sub_prices) >= 2:
                entry = candidate.entry_price
                if candidate.side == "buy":
                    min_price = min(sub_prices)
                    if entry > 0:
                        drawdown_pct = (entry - min_price) / entry
                        if drawdown_pct > self._pullback_max_pct:
                            return False
                else:  # sell
                    max_price = max(sub_prices)
                    if entry > 0:
                        runup_pct = (max_price - entry) / entry
                        if runup_pct > self._pullback_max_pct:
                            return False

        # 1) Spread tightening: current spread significantly tighter than candidate
        if (
            candidate.spread_bps is not None
            and spread_bps is not None
            and spread_bps <= candidate.spread_bps * self._spread_tightening_ratio
        ):
            return True

        # 2) Volatility expansion: short/long ratio expands vs candidate and
        #    meets (or beats) the configured impulse expansion ratio.
        candidate_ratio: float | None = None
        if candidate.vol_short is not None and candidate.vol_long not in (None, 0.0):
            candidate_ratio = candidate.vol_short / candidate.vol_long

        current_ratio: float | None = None
        if vol_short is not None and vol_long not in (None, 0.0):
            current_ratio = vol_short / vol_long

        required_ratio = settings.impulse_vol_expansion_ratio
        if (
            current_ratio is not None
            and current_ratio >= required_ratio
            and (candidate_ratio is None or current_ratio > candidate_ratio)
        ):
            return True

        # 3) Micro pullback then resume:
        #    - For a long: price dips modestly below entry, then trades back
        #      to or above entry within the timeout.
        #    - For a short: price pops modestly above entry, then trades back
        #      to or below entry.
        if prices and 0 <= candidate.price_index < len(prices):
            sub_prices = prices[candidate.price_index :]
            if len(sub_prices) >= 2:
                candidate_price = candidate.entry_price
                current_price = prices[-1]

                if candidate.side == "buy":
                    min_price = min(sub_prices)
                    if self._is_micro_pullback_resume_long(
                        candidate_price, min_price, current_price
                    ):
                        return True
                else:  # "sell"
                    max_price = max(sub_prices)
                    if self._is_micro_pullback_resume_short(
                        candidate_price, max_price, current_price
                    ):
                        return True

        return False

    def _is_micro_pullback_resume_long(
        self, entry: float, min_price: float, current_price: float
    ) -> bool:
        """Micro pullback then resume for long side."""
        if entry <= 0:
            return False
        drawdown_pct = (entry - min_price) / entry
        if drawdown_pct <= 0:
            return False
        if drawdown_pct < self._pullback_min_pct:
            return False
        if drawdown_pct > self._pullback_max_pct:
            return False
        # Must have resumed back to (or above) entry.
        if current_price >= entry:
            return True
        return False

    def _is_micro_pullback_resume_short(
        self, entry: float, max_price: float, current_price: float
    ) -> bool:
        """Micro pullback then resume for short side."""
        if entry <= 0:
            return False
        runup_pct = (max_price - entry) / entry
        if runup_pct <= 0:
            return False
        if runup_pct < self._pullback_min_pct:
            return False
        if runup_pct > self._pullback_max_pct:
            return False
        # Must have resumed back to (or below) entry.
        if current_price <= entry:
            return True
        return False
