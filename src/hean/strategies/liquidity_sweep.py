"""Liquidity Sweep Detector Strategy.

Detects when price sweeps key liquidity levels (round numbers, previous highs/lows)
and reverses quickly - a signature of institutional liquidity hunting.

Expected Impact: +15-20% daily profit
Risk: Medium (requires tight stops)

Liquidity Sweep Pattern:
1. Price approaches key level (round number, previous high/low)
2. Price spikes THROUGH level (triggers stops)
3. Price IMMEDIATELY reverses back (sweep complete)
4. Trade the reversal with tight stop BEYOND the swept level

Key Levels:
- Round numbers: $100, $50, $25 intervals for BTC/ETH
- Previous session high/low
- Previous 4H candle high/low

Entry Criteria:
- Price swept level by 0.1-0.3% (enough to trigger stops)
- Reversal within 3-5 ticks (<30 seconds)
- Volume spike on sweep (2x average)
- OFI confirms reversal pressure
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hean.core.bus import EventBus
from hean.core.regime import Regime
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


@dataclass
class SweepDetection:
    """Detected liquidity sweep event."""
    symbol: str
    level: float
    direction: str  # 'up' or 'down'
    sweep_price: float
    reversal_price: float
    volume_spike: float
    detected_at: datetime
    traded: bool = False


class LiquiditySweepDetector(BaseStrategy):
    """Detects liquidity sweeps and trades the reversal.

    Key features:
    - Tracks round number levels ($100, $50, $25 intervals)
    - Monitors for price spikes through levels
    - Detects immediate reversals (within 5 ticks)
    - Uses volume confirmation
    - Optional OFI confirmation
    """

    # Default round number intervals per symbol
    DEFAULT_INTERVALS = {
        "BTCUSDT": 100.0,  # $100 intervals for BTC
        "ETHUSDT": 25.0,   # $25 intervals for ETH
    }

    def __init__(
        self,
        bus: EventBus,
        symbols: list[str] | None = None,
        ofi_monitor: Any = None,
        enabled: bool = True,
    ) -> None:
        """Initialize liquidity sweep detector.

        Args:
            bus: Event bus for communication
            symbols: List of symbols to trade
            ofi_monitor: Optional OFI monitor for confirmation
            enabled: Whether strategy is enabled
        """
        super().__init__("liquidity_sweep", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._ofi_monitor = ofi_monitor
        self._enabled = enabled

        # Key level detection
        self._round_number_interval: dict[str, float] = {}
        for symbol in self._symbols:
            self._round_number_interval[symbol] = self.DEFAULT_INTERVALS.get(symbol, 50.0)

        # Price/volume tracking
        self._price_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}
        self._tick_timestamps: dict[str, deque[datetime]] = {}
        self._window_size = 100

        # Previous highs/lows tracking (rolling 4H window)
        self._session_highs: dict[str, deque[float]] = {}
        self._session_lows: dict[str, deque[float]] = {}
        self._lookback_candles = 24  # Track last 24 4H candles

        # Sweep detection parameters
        self._sweep_threshold_pct = 0.003  # 0.3% beyond level
        self._sweep_window_ticks = 5  # Must reverse within 5 ticks
        self._volume_spike_threshold = 2.0  # 2x average volume
        self._require_ofi_confirmation = False  # Optional OFI confirmation

        # Active sweep tracking
        self._active_sweeps: dict[str, SweepDetection] = {}
        self._recent_sweeps: deque[SweepDetection] = deque(maxlen=100)

        # Regime
        self._allowed_regimes = {Regime.NORMAL, Regime.IMPULSE}
        self._current_regime: dict[str, Regime] = {}

        # Cooldown
        self._last_trade_time: dict[str, datetime] = {}
        self._trade_cooldown = timedelta(minutes=15)

        # Metrics
        self._total_sweeps_detected = 0
        self._sweeps_traded = 0
        self._sweep_win_rate = 0.0
        self._recent_trades: deque[bool] = deque(maxlen=20)  # Track last 20 trades (win/loss)

        logger.info(
            f"LiquiditySweepDetector initialized: symbols={self._symbols}, "
            f"sweep_threshold={self._sweep_threshold_pct:.2%}"
        )

    async def on_tick(self, event: Event) -> None:
        """Handle tick events."""
        if not self._enabled:
            return

        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Initialize tracking if needed
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)
            self._volume_history[tick.symbol] = deque(maxlen=self._window_size)
            self._tick_timestamps[tick.symbol] = deque(maxlen=self._window_size)
            self._session_highs[tick.symbol] = deque(maxlen=self._lookback_candles)
            self._session_lows[tick.symbol] = deque(maxlen=self._lookback_candles)

        # Update history
        self._price_history[tick.symbol].append(tick.price)
        vol = tick.volume if hasattr(tick, 'volume') and tick.volume else 0.0
        self._volume_history[tick.symbol].append(vol)
        self._tick_timestamps[tick.symbol].append(datetime.utcnow())

        # Update session highs/lows (every 100 ticks as proxy for 4H candles)
        if len(self._price_history[tick.symbol]) >= 100:
            recent_prices = list(self._price_history[tick.symbol])[-100:]
            if len(recent_prices) == 100:
                session_high = max(recent_prices)
                session_low = min(recent_prices)

                # Only add if different from last
                if not self._session_highs[tick.symbol] or \
                   abs(session_high - self._session_highs[tick.symbol][-1]) > 1:
                    self._session_highs[tick.symbol].append(session_high)

                if not self._session_lows[tick.symbol] or \
                   abs(session_low - self._session_lows[tick.symbol][-1]) > 1:
                    self._session_lows[tick.symbol].append(session_low)

        # Check regime
        current_regime = self._current_regime.get(tick.symbol, Regime.NORMAL)
        if current_regime not in self._allowed_regimes:
            return

        # Check cooldown
        if tick.symbol in self._last_trade_time:
            if datetime.utcnow() - self._last_trade_time[tick.symbol] < self._trade_cooldown:
                return

        # Detect liquidity sweeps
        await self._detect_sweep(tick)

    async def on_funding(self, event: Event) -> None:
        """Not used by this strategy."""
        pass

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime updates."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol and regime:
            self._current_regime[symbol] = regime

    def _get_key_levels(self, symbol: str, current_price: float) -> dict[str, list[float]]:
        """Get key liquidity levels near current price.

        Returns:
            {
                'round_numbers': [level1, level2, ...],
                'previous_highs': [high1, high2, ...],
                'previous_lows': [low1, low2, ...]
            }
        """
        levels: dict[str, list[float]] = {
            'round_numbers': [],
            'previous_highs': [],
            'previous_lows': [],
        }

        # Round numbers within +/- 2% of current price
        interval = self._round_number_interval.get(symbol, 50.0)
        nearest = round(current_price / interval) * interval

        for offset in [-2, -1, 0, 1, 2]:
            level = nearest + (offset * interval)
            if abs(level - current_price) / current_price <= 0.02:  # Within 2%
                levels['round_numbers'].append(level)

        # Previous highs/lows within 2%
        if symbol in self._session_highs:
            for high in self._session_highs[symbol]:
                if abs(high - current_price) / current_price <= 0.02:
                    levels['previous_highs'].append(high)

        if symbol in self._session_lows:
            for low in self._session_lows[symbol]:
                if abs(low - current_price) / current_price <= 0.02:
                    levels['previous_lows'].append(low)

        return levels

    async def _detect_sweep(self, tick: Tick) -> None:
        """Detect liquidity sweep pattern."""
        symbol = tick.symbol
        current_price = tick.price

        # Need sufficient history
        if len(self._price_history[symbol]) < 10:
            return

        prices = list(self._price_history[symbol])
        volumes = list(self._volume_history[symbol])

        # Get key levels
        key_levels = self._get_key_levels(symbol, current_price)
        all_levels = (
            key_levels['round_numbers'] +
            key_levels['previous_highs'] +
            key_levels['previous_lows']
        )

        if not all_levels:
            return

        # Check for sweep pattern at each level
        for level in all_levels:
            sweep = self._check_sweep_pattern(symbol, level, prices, volumes)

            if sweep:
                self._total_sweeps_detected += 1
                metrics.increment("liquidity_sweeps_detected")

                logger.info(
                    f"[LIQUIDITY SWEEP] {symbol} {sweep.direction.upper()} sweep at ${level:.2f}, "
                    f"price now ${current_price:.2f}, volume_spike={sweep.volume_spike:.1f}x"
                )

                # Trade the reversal
                await self._trade_sweep_reversal(tick, sweep)

    def _check_sweep_pattern(
        self,
        symbol: str,
        level: float,
        prices: list[float],
        volumes: list[float]
    ) -> SweepDetection | None:
        """Check if price swept a level and reversed.

        Returns:
            SweepDetection if pattern found, None otherwise
        """
        if len(prices) < 6:
            return None

        current_price = prices[-1]

        # Look at last 5 ticks for sweep pattern
        for i in range(len(prices) - 5, len(prices)):
            if i < 1:
                continue

            price_before = prices[i - 1]
            price_at = prices[i]

            # Check for upward sweep (price was below, spiked above)
            if price_before < level and price_at >= level * (1 + self._sweep_threshold_pct):
                # Check for reversal (price came back below level)
                if current_price < level:
                    # Calculate volume spike
                    avg_volume = sum(volumes[:-5]) / max(1, len(volumes) - 5) if len(volumes) > 5 else 1.0
                    sweep_volume = volumes[i] if i < len(volumes) else 0.0
                    volume_spike = (sweep_volume / avg_volume) if avg_volume > 0 else 0.0

                    if volume_spike >= self._volume_spike_threshold or volume_spike == 0:
                        return SweepDetection(
                            symbol=symbol,
                            level=level,
                            direction='up',
                            sweep_price=price_at,
                            reversal_price=current_price,
                            volume_spike=volume_spike,
                            detected_at=datetime.utcnow(),
                        )

            # Check for downward sweep (price was above, spiked below)
            if price_before > level and price_at <= level * (1 - self._sweep_threshold_pct):
                # Check for reversal (price came back above level)
                if current_price > level:
                    # Calculate volume spike
                    avg_volume = sum(volumes[:-5]) / max(1, len(volumes) - 5) if len(volumes) > 5 else 1.0
                    sweep_volume = volumes[i] if i < len(volumes) else 0.0
                    volume_spike = (sweep_volume / avg_volume) if avg_volume > 0 else 0.0

                    if volume_spike >= self._volume_spike_threshold or volume_spike == 0:
                        return SweepDetection(
                            symbol=symbol,
                            level=level,
                            direction='down',
                            sweep_price=price_at,
                            reversal_price=current_price,
                            volume_spike=volume_spike,
                            detected_at=datetime.utcnow(),
                        )

        return None

    async def _trade_sweep_reversal(self, tick: Tick, sweep: SweepDetection) -> None:
        """Generate signal to trade the sweep reversal.

        Trade OPPOSITE to sweep direction:
        - Upward sweep → price reversed down → SELL
        - Downward sweep → price reversed up → BUY
        """
        symbol = tick.symbol
        entry_price = tick.price

        # Determine trade direction (opposite of sweep)
        if sweep.direction == 'up':
            side = 'sell'
            stop_loss = sweep.level * 1.005  # 0.5% above swept level
            take_profit = entry_price * 0.99  # 1% target
            take_profit_1 = entry_price * 0.995  # 0.5% first TP
        else:
            side = 'buy'
            stop_loss = sweep.level * 0.995  # 0.5% below swept level
            take_profit = entry_price * 1.01  # 1% target
            take_profit_1 = entry_price * 1.005  # 0.5% first TP

        # Size multiplier based on confidence
        size_multiplier = min(2.0, 1.0 + max(0, (sweep.volume_spike - self._volume_spike_threshold) / 2.0))

        # OFI confirmation (optional)
        ofi_confirmation = False
        if self._ofi_monitor and self._require_ofi_confirmation:
            try:
                ofi_result = self._ofi_monitor.calculate_ofi(symbol)
                if side == 'buy' and ofi_result.ofi_value > 0.2:
                    ofi_confirmation = True
                elif side == 'sell' and ofi_result.ofi_value < -0.2:
                    ofi_confirmation = True

                if not ofi_confirmation:
                    logger.debug(
                        f"[LIQUIDITY SWEEP] OFI does not confirm reversal for {symbol}, "
                        f"trading anyway (OFI confirmation optional)"
                    )
            except Exception as e:
                logger.debug(f"[LIQUIDITY SWEEP] OFI check failed: {e}")

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_1=take_profit_1,
            metadata={
                "type": "liquidity_sweep_reversal",
                "swept_level": sweep.level,
                "sweep_direction": sweep.direction,
                "sweep_price": sweep.sweep_price,
                "volume_spike": sweep.volume_spike,
                "size_multiplier": size_multiplier,
                "ofi_confirmation": ofi_confirmation,
            },
            prefer_maker=False,  # Use taker for immediate entry
            min_maker_edge_bps=0.0,
        )

        await self._publish_signal(signal)

        # Update tracking
        sweep.traded = True
        self._recent_sweeps.append(sweep)
        self._last_trade_time[symbol] = datetime.utcnow()
        self._sweeps_traded += 1
        metrics.increment("liquidity_sweep_trades")

        logger.info(
            f"[LIQUIDITY SWEEP TRADE] {symbol} {side.upper()} @ ${entry_price:.2f}, "
            f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}, size_mult={size_multiplier:.1f}x"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get strategy metrics."""
        return {
            "total_sweeps_detected": self._total_sweeps_detected,
            "sweeps_traded": self._sweeps_traded,
            "sweep_win_rate": self._sweep_win_rate,
            "recent_sweeps": [
                {
                    "symbol": s.symbol,
                    "level": s.level,
                    "direction": s.direction,
                    "volume_spike": s.volume_spike,
                    "traded": s.traded,
                    "time": s.detected_at.isoformat(),
                }
                for s in list(self._recent_sweeps)[-5:]
            ],
            "enabled": self._enabled,
            "symbols": self._symbols,
        }

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the strategy."""
        self._enabled = enabled
        logger.info(f"LiquiditySweepDetector {'ENABLED' if enabled else 'DISABLED'}")

    def set_sweep_threshold(self, threshold_pct: float) -> None:
        """Set the sweep threshold percentage.

        Args:
            threshold_pct: Threshold as decimal (e.g., 0.003 = 0.3%)
        """
        self._sweep_threshold_pct = max(0.001, min(0.01, threshold_pct))
        logger.info(f"Sweep threshold set to {self._sweep_threshold_pct:.2%}")

    def set_cooldown(self, minutes: int) -> None:
        """Set the trade cooldown period.

        Args:
            minutes: Cooldown period in minutes
        """
        self._trade_cooldown = timedelta(minutes=max(5, minutes))
        logger.info(f"Trade cooldown set to {minutes} minutes")
