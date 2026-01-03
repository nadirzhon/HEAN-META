"""Market context primitives and builders.

This module provides a lightweight, serialisable description of the
current market and strategy environment.  The goal is to centralise
how we derive common context features (spread, volatility, time‑of‑day,
liquidity, etc.) from raw market data and regime detectors so that
downstream components (strategies, decision memory, diagnostics) can
depend on a stable contract.

Phase‑0 scope: introduce the data contracts and a minimal `ContextBuilder`
that can be evolved without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from hean.core.regime import Regime, RegimeDetector
from hean.core.types import Tick


@dataclass(slots=True)
class MarketContext:
    """Snapshot of market and meta‑context for a single symbol.

    Attributes:
        symbol: Instrument symbol, e.g. ``"BTCUSDT"``.
        regime: Current market regime for the symbol.
        spread_bps: Bid/ask spread in basis points, or ``None`` if unknown.
        vol: Approximate current volatility (e.g. long‑window stdev of returns).
        vol_short: Optional short‑horizon volatility estimate.
        vol_long: Optional long‑horizon volatility estimate.
        hour_utc: Hour of day in UTC (0‑23) derived from the tick timestamp.
        liquidity_score: Heuristic [0.0, 1.0] proxy for liquidity conditions.
    """

    symbol: str
    regime: Regime | None
    spread_bps: float | None = None
    vol: float | None = None
    vol_short: float | None = None
    vol_long: float | None = None
    hour_utc: int | None = None
    liquidity_score: float | None = None


class ContextBuilder:
    """Utility for constructing :class:`MarketContext` from core signals.

    The builder is intentionally thin: it does not own any state beyond
    references to collaborators (e.g. a :class:`RegimeDetector`) and
    simple configuration.  This makes it cheap to create per‑strategy
    instances and easy to extend in future phases.
    """

    def __init__(self, regime_detector: RegimeDetector | None = None) -> None:
        """Initialise a new context builder.

        Args:
            regime_detector: Optional shared :class:`RegimeDetector`
                instance used to query the current regime and volatility
                for a symbol.  When omitted, those fields will be left
                as ``None`` in the resulting :class:`MarketContext`.
        """

        self._regime_detector = regime_detector

    def build(
        self,
        tick: Tick,
        *,
        regime: Regime | None = None,
        vol_short: float | None = None,
        vol_long: float | None = None,
        liquidity_score: float | None = None,
        timestamp: datetime | None = None,
    ) -> MarketContext:
        """Compute a :class:`MarketContext` from a :class:`Tick`.

        The builder will:

        - derive ``spread_bps`` from ``bid``/``ask`` if available,
        - obtain ``regime`` and long‑window ``vol`` from the attached
          :class:`RegimeDetector` when not explicitly provided,
        - fill in ``hour_utc`` from the tick (or explicit ``timestamp``),
        - provide a conservative default for ``liquidity_score`` when
          none is supplied by a higher‑level component.

        This logic mirrors – but does not change – the heuristics used
        in existing strategies such as :mod:`hean.strategies.impulse_engine`.
        """

        ts = timestamp or tick.timestamp

        # Symbol is always taken directly from the tick.
        symbol = tick.symbol

        # Regime and long‑window volatility come from the detector when present.
        current_regime: Regime | None = regime
        long_vol: float | None = vol_long
        if self._regime_detector is not None:
            if current_regime is None:
                current_regime = self._regime_detector.get_regime(symbol)
            if long_vol is None:
                long_vol = self._regime_detector.get_volatility(symbol)

        # Spread in basis points, if we have a two‑sided quote.
        spread_bps: float | None = None
        if tick.bid is not None and tick.ask is not None and tick.price and tick.price > 0:
            spread = (tick.ask - tick.bid) / tick.price
            spread_bps = max(0.0, float(spread) * 10_000.0)

        # Hour of day (UTC) from timestamp.
        hour_utc: int | None = ts.hour if isinstance(ts, datetime) else None

        # Liquidity score: simple heuristic for now.
        # We keep this intentionally lightweight for phase 0:
        # - if spread is tight (< 10 bps) → score ~1.0
        # - if spread is wide (> 50 bps)  → score ~0.0
        # - otherwise interpolate linearly.
        if liquidity_score is None:
            if spread_bps is None:
                liquidity = None
            else:
                tight = 10.0
                wide = 50.0
                if spread_bps <= tight:
                    liquidity = 1.0
                elif spread_bps >= wide:
                    liquidity = 0.0
                else:
                    # Linear interpolation between tight and wide.
                    liquidity = max(
                        0.0,
                        min(1.0, 1.0 - (spread_bps - tight) / (wide - tight)),
                    )
        else:
            liquidity = max(0.0, min(1.0, float(liquidity)))

        return MarketContext(
            symbol=symbol,
            regime=current_regime,
            spread_bps=spread_bps,
            vol=long_vol,
            vol_short=vol_short,
            vol_long=long_vol if long_vol is not None else vol_long,
            hour_utc=hour_utc,
            liquidity_score=liquidity,
        )
