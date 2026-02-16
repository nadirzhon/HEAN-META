"""Funding harvester strategy - low risk directional bias based on funding.

Enhanced with:
- Multi-symbol opportunity ranking
- Optimal entry timing (1-2 hours before funding)
- Expected profit calculation for position sizing
- Funding rate momentum tracking
- ML-enhanced funding prediction (momentum, volatility, time-of-day features)
- Leverage multiplier for high-confidence predictions
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.market_context import UnifiedMarketContext
from hean.core.regime import Regime
from hean.core.types import Event, FundingRate, Signal, Tick
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


@dataclass
class FundingPrediction:
    """ML-enhanced funding prediction result."""
    predicted_rate: float
    confidence: float  # 0.0 to 1.0
    features: dict[str, float]
    recommended_leverage: float = 1.0


class FundingHarvester(BaseStrategy):
    """Harvests funding by taking small directional positions based on funding rate.

    When funding is positive (longs pay shorts), we take a small short bias.
    When funding is negative (shorts pay longs), we take a small long bias.

    Active in all regimes.
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None, http_client=None) -> None:
        """Initialize the funding harvester.

        Args:
            bus: Event bus for communication
            symbols: List of symbols to trade (default: BTC, ETH)
            http_client: Bybit HTTP client for fetching funding rates (optional)
        """
        super().__init__("funding_harvester", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._last_funding: dict[str, FundingRate] = {}
        self._positions: dict[str, str] = {}  # symbol -> side
        self._http_client = http_client

        # Historical funding tracking (7-day window for each symbol)
        self._historical_funding: dict[str, deque] = {
            symbol: deque(maxlen=56) for symbol in self._symbols  # 8 hrs * 7 days = 56 samples
        }

        # Current market data for prediction
        self._last_tick: dict[str, Tick] = {}
        self._oi_history: dict[str, deque] = {
            symbol: deque(maxlen=10) for symbol in self._symbols
        }

        # Active in all regimes
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL, Regime.IMPULSE}

        # Phase 3: Enhanced Funding Optimization
        # Funding opportunity tracking for multi-symbol ranking
        self._funding_opportunities: dict[str, dict[str, Any]] = {}
        self._optimal_entry_window = (1.0, 2.0)  # 1-2 hours before funding (optimal)
        self._min_funding_threshold = 0.0001  # LOWERED: 0.01% minimum (was 0.02%)
        self._max_concurrent_positions = 2  # Max positions across symbols
        self._active_positions_count = 0

        # Price history for momentum fallback signal
        self._price_history: dict[str, deque] = {
            symbol: deque(maxlen=100) for symbol in self._symbols
        }

        # ML-enhanced prediction settings
        self._ml_predictor_enabled = True
        self._funding_leverage_enabled = True
        self._max_funding_leverage = 3.0  # Max leverage for funding trades
        self._min_confidence_for_leverage = 0.6  # Min confidence for 2x leverage
        self._high_confidence_threshold = 0.75  # Threshold for 3x leverage

        # ML feature tracking
        self._funding_features: dict[str, deque] = {
            symbol: deque(maxlen=24) for symbol in self._symbols  # 24 samples for features
        }

        # Anti-overtrading: Cooldown and signal limits
        from datetime import timedelta
        self._last_signal_time: dict[str, datetime] = {}  # Per-symbol last signal time
        self._signal_cooldown = timedelta(hours=4)  # Min 4 hours between signals per symbol
        self._daily_signals: int = 0
        self._max_daily_signals: int = 6  # Max 6 signals per day (conservative)
        self._daily_reset_time: datetime | None = None

        # Unified context from ContextAggregator
        self._unified_context: dict[str, UnifiedMarketContext] = {}

        # Enhanced metrics
        self._funding_collected = 0.0
        self._funding_trades = 0
        self._best_opportunities: list[dict[str, Any]] = []
        self._ml_prediction_accuracy: dict[str, deque] = {
            symbol: deque(maxlen=20) for symbol in self._symbols
        }

    async def on_tick(self, event: Event) -> None:
        """Handle tick events - store for funding prediction and generate fallback signals."""
        tick: Tick = event.data.get("tick")
        if tick and tick.symbol in self._symbols:
            self._last_tick[tick.symbol] = tick
            self._price_history[tick.symbol].append(tick.price)

            # FALLBACK: If no funding data available, generate simple directional signals based on price momentum
            # This allows the strategy to work even without funding API data
            if tick.symbol not in self._last_funding or len(self._historical_funding[tick.symbol]) == 0:
                await self._generate_momentum_fallback_signal(tick)

    async def on_context_ready(self, event: Event) -> None:
        """Handle unified context from ContextAggregator."""
        ctx: UnifiedMarketContext | None = event.data.get("context")
        if ctx is None:
            return
        if ctx.symbol in self._symbols:
            self._unified_context[ctx.symbol] = ctx

    async def on_funding(self, event: Event) -> None:
        """Handle funding rate events."""
        funding: FundingRate = event.data["funding"]

        if funding.symbol not in self._symbols:
            return

        self._last_funding[funding.symbol] = funding

        # Store in historical funding for prediction
        self._historical_funding[funding.symbol].append({
            "rate": funding.rate,
            "timestamp": funding.timestamp,
        })

        # Generate signal based on funding rate
        await self._evaluate_funding(funding)

    def _calculate_expected_profit(self, funding_rate: float, position_size_usd: float) -> float:
        """Calculate expected profit from funding collection.

        Args:
            funding_rate: Funding rate (decimal)
            position_size_usd: Position size in USD

        Returns:
            Expected profit in USD
        """
        # Funding payment = position_size * funding_rate
        # We receive payment if we're on the right side
        return abs(funding_rate) * position_size_usd

    def _rank_funding_opportunities(self) -> list[dict[str, Any]]:
        """Rank funding opportunities across all symbols.

        Returns:
            Sorted list of opportunities (best first)
        """
        opportunities = []

        for symbol in self._symbols:
            if symbol not in self._last_funding:
                continue

            funding = self._last_funding[symbol]
            if symbol not in self._last_tick:
                continue

            tick = self._last_tick[symbol]

            # Calculate time to funding
            time_to_funding_hrs = None
            if funding.next_funding_time:
                time_to_funding = (funding.next_funding_time - datetime.utcnow()).total_seconds()
                time_to_funding_hrs = time_to_funding / 3600
            else:
                continue

            # Check if in optimal window (1-2 hours)
            in_optimal_window = (
                self._optimal_entry_window[0] <= time_to_funding_hrs <= self._optimal_entry_window[1]
            )

            # Predict next funding
            predicted_funding = self.predict_next_funding(symbol)

            # Calculate opportunity score
            funding_strength = abs(funding.rate)
            prediction_alignment = 1.0 if (funding.rate * predicted_funding) > 0 else 0.5
            timing_bonus = 1.5 if in_optimal_window else 1.0

            opportunity_score = funding_strength * prediction_alignment * timing_bonus

            opportunities.append({
                "symbol": symbol,
                "funding_rate": funding.rate,
                "predicted_funding": predicted_funding,
                "time_to_funding_hrs": time_to_funding_hrs,
                "in_optimal_window": in_optimal_window,
                "opportunity_score": opportunity_score,
                "entry_price": tick.price,
                "side": "sell" if funding.rate > 0 else "buy",
            })

        # Sort by opportunity score (highest first)
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)

        self._best_opportunities = opportunities[:3]  # Store top 3
        return opportunities

    async def _evaluate_funding(self, funding: FundingRate) -> None:
        """Evaluate funding rate and generate signal if appropriate."""
        # Phase 3: Enhanced evaluation with multi-symbol ranking

        # Threshold: only act if funding rate is significant
        if abs(funding.rate) < self._min_funding_threshold:
            return

        # Get current market price from tick
        if funding.symbol not in self._last_tick:
            logger.debug(f"No tick data for {funding.symbol}, skipping funding signal")
            return

        # Check position limits
        if self._active_positions_count >= self._max_concurrent_positions:
            logger.debug(
                f"Max concurrent positions reached ({self._max_concurrent_positions}), "
                f"skipping {funding.symbol}"
            )
            return

        # Anti-overtrading: Reset daily counter if new day
        now = datetime.utcnow()
        if self._daily_reset_time is None or now.date() > self._daily_reset_time.date():
            self._daily_signals = 0
            self._daily_reset_time = now

        # Anti-overtrading: Check daily signal limit
        if self._daily_signals >= self._max_daily_signals:
            logger.debug(
                f"Daily signal limit reached ({self._max_daily_signals}), "
                f"skipping {funding.symbol}"
            )
            return

        # Anti-overtrading: Check per-symbol cooldown
        if funding.symbol in self._last_signal_time:
            time_since = now - self._last_signal_time[funding.symbol]
            if time_since < self._signal_cooldown:
                logger.debug(
                    f"Signal blocked: {funding.symbol} in cooldown "
                    f"(last signal {time_since.total_seconds() / 3600:.1f}h ago, "
                    f"cooldown {self._signal_cooldown.total_seconds() / 3600:.0f}h)"
                )
                return

        # Rank all opportunities
        opportunities = self._rank_funding_opportunities()

        # Find this symbol in opportunities
        symbol_opportunity = next(
            (o for o in opportunities if o["symbol"] == funding.symbol), None
        )

        if not symbol_opportunity:
            return

        # Only take top opportunities
        top_symbols = [o["symbol"] for o in opportunities[:self._max_concurrent_positions]]
        if funding.symbol not in top_symbols:
            logger.debug(
                f"{funding.symbol} not in top {self._max_concurrent_positions} opportunities, skipping"
            )
            return

        # Check timing - optimal window is 1-2 hours before funding
        if not symbol_opportunity["in_optimal_window"]:
            time_hrs = symbol_opportunity["time_to_funding_hrs"]
            # Allow entry in extended window (0.5-4 hours) with reduced size
            if time_hrs < 0.5 or time_hrs > 4.0:
                logger.debug(
                    f"Funding timing not optimal for {funding.symbol}: {time_hrs:.1f}h until funding"
                )
                return

        tick = self._last_tick[funding.symbol]
        entry_price = tick.price

        # Check if we already have a position
        if funding.symbol in self._positions:
            current_side = self._positions[funding.symbol]
            if current_side == symbol_opportunity["side"]:
                return  # Already positioned correctly

        # Use ML-enhanced prediction
        ml_prediction = self.predict_next_funding_ml(funding.symbol)
        predicted_funding = ml_prediction.predicted_rate
        ml_confidence = ml_prediction.confidence
        recommended_leverage = ml_prediction.recommended_leverage

        # Calculate confidence and size multiplier
        funding_strength = abs(funding.rate) / 0.0005  # Normalize by 0.05% funding
        prediction_alignment = 1.0 if (funding.rate * predicted_funding) > 0 else 0.5
        confidence = min(1.0, funding_strength * prediction_alignment * ml_confidence)

        # Size multiplier based on opportunity quality
        size_multiplier = 0.5  # Base for funding trades
        if symbol_opportunity["in_optimal_window"]:
            size_multiplier *= 1.5  # Bonus for optimal timing
        if funding_strength > 1.0:  # Strong funding (> 0.05%)
            size_multiplier *= 1.2

        # Apply ML-recommended leverage
        leverage_multiplier = 1.0
        if self._funding_leverage_enabled and ml_confidence >= self._min_confidence_for_leverage:
            leverage_multiplier = recommended_leverage
            size_multiplier *= leverage_multiplier
            logger.info(
                f"[FUNDING LEVERAGE] {funding.symbol} using {leverage_multiplier:.1f}x leverage "
                f"(ML confidence={ml_confidence:.2f})"
            )

        side = symbol_opportunity["side"]
        time_to_funding_hrs = symbol_opportunity["time_to_funding_hrs"]

        # Apply unified context from ContextAggregator
        if funding.symbol in self._unified_context:
            ctx = self._unified_context[funding.symbol]
            # If brain says bearish with high confidence, skip long funding harvests
            if side == "buy" and ctx.brain.sentiment == "bearish" and ctx.brain.confidence > 0.7:
                logger.info(
                    f"[FUNDING CONTEXT] Skipping {funding.symbol} long: "
                    f"brain bearish (conf={ctx.brain.confidence:.2f})"
                )
                return
            # Apply context size multiplier
            size_multiplier *= ctx.size_multiplier
            # If context signals conflict, halve position
            if ctx.should_reduce_size:
                size_multiplier *= 0.5
                logger.debug(f"[FUNDING CONTEXT] {funding.symbol} size halved: context conflict")

        # Calculate expected profit for logging (includes leverage)
        assumed_position_size = 100 * leverage_multiplier  # Leveraged size
        expected_profit = self._calculate_expected_profit(funding.rate, assumed_position_size)

        # Generate signal with enhanced metadata
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=funding.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=entry_price * (0.985 if side == "buy" else 1.015),  # 1.5% stop (tighter)
            take_profit=entry_price * (1.008 if side == "buy" else 0.992),  # 0.8% target
            take_profit_1=entry_price * (1.003 if side == "buy" else 0.997),  # 0.3% first TP
            metadata={
                "type": "ml_enhanced_funding_harvest",
                "funding_rate": funding.rate,
                "predicted_funding": predicted_funding,
                "ml_confidence": ml_confidence,
                "confidence": confidence,
                "time_to_funding_hrs": time_to_funding_hrs,
                "in_optimal_window": symbol_opportunity["in_optimal_window"],
                "opportunity_rank": top_symbols.index(funding.symbol) + 1,
                "expected_profit_per_100usd": expected_profit,
                "size_multiplier": size_multiplier,
                "leverage": leverage_multiplier,
                "ml_features": ml_prediction.features,
            },
            prefer_maker=True,  # Use maker for funding trades
            min_maker_edge_bps=1.0,
        )

        await self._publish_signal(signal)
        self._positions[funding.symbol] = side
        self._active_positions_count += 1
        self._funding_trades += 1

        # Anti-overtrading: Update signal tracking
        self._last_signal_time[funding.symbol] = datetime.utcnow()
        self._daily_signals += 1

        metrics.increment("funding_harvester_trades")

        logger.info(
            f"[ENHANCED FUNDING] {funding.symbol} {side.upper()} @ ${entry_price:.2f} "
            f"(funding={funding.rate:.4%}, pred={predicted_funding:.4%}, "
            f"time={time_to_funding_hrs:.1f}h, rank=#{top_symbols.index(funding.symbol) + 1})"
        )

    def predict_next_funding(self, symbol: str) -> float:
        """
        Predict next funding rate based on historical data and market conditions.

        Uses simple model:
        - Historical funding mean (momentum)
        - Recent price momentum
        - Open interest changes (if available)

        Args:
            symbol: Trading symbol

        Returns:
            Predicted funding rate (decimal)
        """
        # Get historical funding data
        if symbol not in self._historical_funding or len(self._historical_funding[symbol]) < 3:
            # Not enough history, use current funding
            if symbol in self._last_funding:
                return self._last_funding[symbol].rate
            return 0.0

        history = list(self._historical_funding[symbol])

        # Calculate exponentially weighted moving average (recent data weighted more)
        weights = [0.5 ** (len(history) - i - 1) for i in range(len(history))]
        total_weight = sum(weights)
        weighted_avg = sum(h["rate"] * w for h, w in zip(history, weights, strict=False)) / total_weight

        # Calculate momentum: recent vs older funding
        recent_avg = sum(h["rate"] for h in history[-3:]) / 3
        older_avg = sum(h["rate"] for h in history[:3]) / 3 if len(history) >= 6 else recent_avg
        momentum = recent_avg - older_avg

        # Price momentum adjustment (if we have recent ticks)
        price_adjustment = 0.0
        if symbol in self._last_tick and len(self._historical_funding[symbol]) > 0:
            # If price is rising, funding tends to increase (more longs)
            # This is a simple heuristic
            price_adjustment = 0.0  # Placeholder for now

        # Combine factors
        predicted = weighted_avg + (momentum * 0.3) + price_adjustment

        return predicted

    def predict_next_funding_ml(self, symbol: str) -> FundingPrediction:
        """ML-enhanced funding prediction with multiple features.

        Features used:
        1. Historical funding momentum (short-term vs long-term)
        2. Funding rate volatility (higher vol = lower confidence)
        3. Time-of-day effects (funding varies by session)
        4. Funding trend persistence

        Returns:
            FundingPrediction with predicted rate, confidence, and recommended leverage
        """
        features: dict[str, float] = {}

        # Base prediction from existing method
        base_prediction = self.predict_next_funding(symbol)

        if not self._ml_predictor_enabled:
            return FundingPrediction(
                predicted_rate=base_prediction,
                confidence=0.5,
                features=features,
                recommended_leverage=1.0,
            )

        # Get historical data
        if symbol not in self._historical_funding or len(self._historical_funding[symbol]) < 3:
            return FundingPrediction(
                predicted_rate=base_prediction,
                confidence=0.5,
                features=features,
                recommended_leverage=1.0,
            )

        history = list(self._historical_funding[symbol])

        # FEATURE 1: Funding Momentum (short vs long)
        if len(history) >= 6:
            recent_3 = [h['rate'] for h in history[-3:]]
            older_3 = [h['rate'] for h in history[-6:-3]]
            momentum = (sum(recent_3) / 3) - (sum(older_3) / 3)
            features['funding_momentum'] = momentum
        else:
            features['funding_momentum'] = 0.0

        # FEATURE 2: Funding Volatility (std dev of rates)
        if len(history) >= 5:
            rates = [h['rate'] for h in history[-5:]]
            mean_rate = sum(rates) / len(rates)
            variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
            volatility = variance ** 0.5
            features['funding_volatility'] = volatility
        else:
            features['funding_volatility'] = 0.0

        # FEATURE 3: Time-of-Day Effect
        # Funding payments happen at 00:00, 08:00, 16:00 UTC
        # Market behavior varies by session
        hour = datetime.utcnow().hour
        features['hour_sin'] = math.sin(2 * math.pi * hour / 24)
        features['hour_cos'] = math.cos(2 * math.pi * hour / 24)

        # FEATURE 4: Funding Trend Persistence
        # How many consecutive periods funding stayed same sign?
        persistence = 0
        if len(history) >= 2:
            current_sign = 1 if history[-1]['rate'] > 0 else -1
            for h in reversed(history[:-1]):
                if (h['rate'] > 0 and current_sign > 0) or (h['rate'] < 0 and current_sign < 0):
                    persistence += 1
                else:
                    break
        features['trend_persistence'] = persistence

        # FEATURE 5: Funding rate magnitude
        current_rate = history[-1]['rate'] if history else 0.0
        features['rate_magnitude'] = abs(current_rate)

        # ML MODEL: Simple weighted combination
        # In production, this would be a trained model (GBM, LSTM, etc.)
        # For now, use heuristic weights
        ml_adjustment = 0.0

        # Momentum has strong predictive power (0.6 weight)
        ml_adjustment += features['funding_momentum'] * 0.6

        # High persistence suggests continuation
        if features['trend_persistence'] >= 3:
            ml_adjustment += features['funding_momentum'] * 0.2  # Extra momentum boost

        predicted_rate = base_prediction + ml_adjustment

        # CONFIDENCE CALCULATION
        # Higher volatility = lower confidence
        # Higher persistence = higher confidence
        base_confidence = 0.6
        vol_penalty = min(0.3, features['funding_volatility'] / 0.001 * 0.3)
        persistence_bonus = min(0.2, features['trend_persistence'] * 0.05)

        confidence = max(0.3, min(0.95, base_confidence - vol_penalty + persistence_bonus))
        features['confidence'] = confidence

        # LEVERAGE RECOMMENDATION
        recommended_leverage = 1.0
        if self._funding_leverage_enabled:
            if confidence >= self._high_confidence_threshold:
                recommended_leverage = self._max_funding_leverage  # 3x
            elif confidence >= self._min_confidence_for_leverage:
                # Scale between 1x and 3x based on confidence
                scale = (confidence - self._min_confidence_for_leverage) / (
                    self._high_confidence_threshold - self._min_confidence_for_leverage
                )
                recommended_leverage = 1.0 + (self._max_funding_leverage - 1.0) * scale

        logger.debug(
            f"[FUNDING ML] {symbol}: predicted={predicted_rate:.4%}, "
            f"confidence={confidence:.2f}, leverage={recommended_leverage:.1f}x, "
            f"momentum={features['funding_momentum']:.6f}, vol={features['funding_volatility']:.6f}"
        )

        return FundingPrediction(
            predicted_rate=predicted_rate,
            confidence=confidence,
            features=features,
            recommended_leverage=recommended_leverage,
        )

    async def _generate_momentum_fallback_signal(self, tick: Tick) -> None:
        """Generate fallback signal based on price momentum when funding data is unavailable.

        This allows the strategy to function without funding API by trading simple
        momentum reversals with funding-like position holding times.
        """
        symbol = tick.symbol

        # Need price history
        if len(self._price_history[symbol]) < 20:
            return

        # Anti-overtrading: Check cooldown
        now = datetime.utcnow()
        if symbol in self._last_signal_time:
            time_since = now - self._last_signal_time[symbol]
            if time_since < self._signal_cooldown:
                return

        # Check daily limit
        if self._daily_reset_time is None or now.date() > self._daily_reset_time.date():
            self._daily_signals = 0
            self._daily_reset_time = now

        if self._daily_signals >= self._max_daily_signals:
            return

        # Check position limits
        if self._active_positions_count >= self._max_concurrent_positions:
            return

        # Calculate simple momentum (compare recent vs older prices)
        prices = list(self._price_history[symbol])
        recent_avg = sum(prices[-5:]) / 5
        older_avg = sum(prices[-15:-5]) / 10
        momentum_pct = (recent_avg - older_avg) / older_avg

        # Only trade if momentum is significant (>0.2%)
        if abs(momentum_pct) < 0.002:
            return

        # Trade against momentum (mean reversion, similar to funding harvesting)
        side = "sell" if momentum_pct > 0 else "buy"
        confidence = min(0.6, abs(momentum_pct) * 100)  # 0.4-0.6 confidence

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            entry_price=tick.price,
            stop_loss=tick.price * (0.985 if side == "buy" else 1.015),
            take_profit=tick.price * (1.008 if side == "buy" else 0.992),
            metadata={
                "type": "momentum_fallback",
                "momentum_pct": momentum_pct,
                "confidence": confidence,
                "note": "fallback_signal_no_funding_data",
            },
            prefer_maker=True,
            min_maker_edge_bps=2.0,
        )

        await self._publish_signal(signal)
        self._positions[symbol] = side
        self._active_positions_count += 1
        self._last_signal_time[symbol] = now
        self._daily_signals += 1

        logger.info(
            f"[FUNDING FALLBACK] {symbol} {side.upper()} @ ${tick.price:.2f} "
            f"(momentum={momentum_pct:.2%}, no funding data available)"
        )

    async def fetch_funding_rates(self) -> None:
        """
        Fetch current funding rates from Bybit API.
        This should be called periodically (e.g., every 8 hours).
        """
        if not self._http_client:
            logger.debug("No HTTP client available for funding rate fetch")
            return

        for symbol in self._symbols:
            try:
                funding_data = await self._http_client.get_funding_rate(symbol)
                if funding_data:
                    funding_rate = float(funding_data.get("fundingRate", 0.0))
                    next_funding_str = funding_data.get("nextFundingTime")
                    next_funding_time = None
                    if next_funding_str:
                        # Parse timestamp (Bybit returns milliseconds)
                        next_funding_time = datetime.fromtimestamp(int(next_funding_str) / 1000)

                    funding = FundingRate(
                        symbol=symbol,
                        rate=funding_rate,
                        timestamp=datetime.utcnow(),
                        next_funding_time=next_funding_time,
                    )

                    # Emit funding event
                    await self._bus.publish(Event(
                        event_type="FUNDING",
                        data={"funding": funding},
                    ))
                    logger.info(
                        f"Fetched funding for {symbol}: {funding_rate:.6f} "
                        f"(next: {next_funding_time})"
                    )
            except Exception as e:
                logger.error(f"Error fetching funding rate for {symbol}: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get enhanced funding harvester metrics."""
        return {
            "funding_trades": self._funding_trades,
            "active_positions": self._active_positions_count,
            "best_opportunities": self._best_opportunities,
            "symbols_tracked": len(self._symbols),
            "optimal_entry_window": self._optimal_entry_window,
            "min_funding_threshold": self._min_funding_threshold,
        }

    def close_position(self, symbol: str) -> None:
        """Mark position as closed.        Args:
            symbol: Symbol to close
        """
        if symbol in self._positions:
            del self._positions[symbol]
            self._active_positions_count = max(0, self._active_positions_count - 1)
            logger.info(f"Funding position closed for {symbol}")
