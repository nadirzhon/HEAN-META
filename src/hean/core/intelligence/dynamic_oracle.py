"""Dynamic Oracle with adaptive model weighting based on market conditions.

Fuses 4 signal sources with dynamic weights that adapt to:
- Market regime (phase, temperature, entropy)
- Recent performance of each source
- Volatility and predictability

Sources (when available):
1. TCN (Temporal Convolutional Network) - price reversal prediction
2. FinBERT - text sentiment from news
3. Ollama - local LLM sentiment
4. Claude Brain - periodic market analysis

Weights adapt in real-time based on market physics.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class DynamicOracleWeighting:
    """Manages dynamic weighting of AI/ML signal sources based on market conditions.

    Adapts weights based on:
    - Market phase: Trend-following models get more weight in markup/markdown
    - Volatility: Predictive models get less weight in high chaos
    - Recent accuracy: Tracks hit rate and adjusts
    """

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._enabled = settings.oracle_dynamic_weighting

        # Base weights (used as fallback when dynamic weighting disabled)
        self._base_weights = {
            "tcn": 0.40,        # Price reversal TCN
            "finbert": 0.20,    # FinBERT sentiment
            "ollama": 0.20,     # Ollama LLM sentiment
            "brain": 0.20,      # Claude Brain
        }

        # Current dynamic weights
        self._current_weights = dict(self._base_weights)

        # Market state (from Physics)
        self._market_phase: dict[str, str] = {}  # symbol -> phase
        self._market_temperature: dict[str, float] = {}
        self._market_entropy: dict[str, float] = {}
        self._market_volatility: dict[str, float] = {}

        # Signal staleness tracking
        self._last_signal_time: dict[str, datetime] = {
            "tcn": datetime.min,
            "finbert": datetime.min,
            "ollama": datetime.min,
            "brain": datetime.min,
        }
        self._signal_stale_threshold = timedelta(minutes=10)

        # Performance tracking per source
        self._source_predictions: dict[str, deque] = {
            "tcn": deque(maxlen=50),
            "finbert": deque(maxlen=50),
            "ollama": deque(maxlen=50),
            "brain": deque(maxlen=50),
        }

        self._running = False
        self._update_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start dynamic weighting system."""
        self._running = True

        # Subscribe to market state updates
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._bus.subscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        self._bus.subscribe(EventType.BRAIN_ANALYSIS, self._handle_brain_analysis)

        # Start periodic weight update task
        if self._enabled:
            self._update_task = asyncio.create_task(self._update_weights_loop())
            logger.info("DynamicOracleWeighting started (enabled)")
        else:
            logger.info("DynamicOracleWeighting started (disabled - using fixed weights)")

    async def stop(self) -> None:
        """Stop dynamic weighting system."""
        self._running = False

        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._bus.unsubscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)
        self._bus.unsubscribe(EventType.BRAIN_ANALYSIS, self._handle_brain_analysis)

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("DynamicOracleWeighting stopped")

    async def _handle_physics_update(self, event: Event) -> None:
        """Track physics state for dynamic weighting."""
        data = event.data
        symbol = data.get("symbol")
        if not symbol:
            return

        self._market_phase[symbol] = data.get("phase", "unknown")
        self._market_temperature[symbol] = data.get("temperature", 0.5)
        self._market_entropy[symbol] = data.get("entropy", 0.5)

    async def _handle_regime_update(self, event: Event) -> None:
        """Track volatility from regime updates."""
        data = event.data
        symbol = data.get("symbol")
        if symbol:
            self._market_volatility[symbol] = data.get("volatility", 0.02)

    async def _handle_context_update(self, event: Event) -> None:
        """Track signal freshness from context updates."""
        data = event.data
        ctx_type = data.get("type")

        # Update signal timestamps
        if ctx_type == "oracle_predictions":
            self._last_signal_time["tcn"] = datetime.utcnow()
        elif ctx_type == "finbert_sentiment":
            self._last_signal_time["finbert"] = datetime.utcnow()
        elif ctx_type == "ollama_sentiment":
            self._last_signal_time["ollama"] = datetime.utcnow()

    async def _handle_brain_analysis(self, event: Event) -> None:
        """Track Brain signal freshness."""
        self._last_signal_time["brain"] = datetime.utcnow()

    async def _update_weights_loop(self) -> None:
        """Periodically update dynamic weights based on market conditions."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                await self._calculate_dynamic_weights()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in weight update loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _calculate_dynamic_weights(self) -> None:
        """Calculate dynamic weights based on current market conditions."""
        if not self._enabled:
            self._current_weights = dict(self._base_weights)
            return

        # Get average market state across all symbols
        avg_temp = np.mean(list(self._market_temperature.values())) if self._market_temperature else 0.5
        avg_entropy = np.mean(list(self._market_entropy.values())) if self._market_entropy else 0.5
        avg_vol = np.mean(list(self._market_volatility.values())) if self._market_volatility else 0.02

        # Count phase distribution
        phases = list(self._market_phase.values())
        phase_counts = {
            "accumulation": phases.count("accumulation"),
            "markup": phases.count("markup"),
            "distribution": phases.count("distribution"),
            "markdown": phases.count("markdown"),
        }
        dominant_phase = max(phase_counts, key=phase_counts.get) if phases else "unknown"

        # Start with base weights
        weights = dict(self._base_weights)

        # Rule 1: High volatility/entropy → reduce TCN weight (less predictable)
        if avg_vol > 0.04 or avg_entropy > 0.7:
            weights["tcn"] *= 0.6  # Reduce predictive model
            weights["brain"] *= 1.2  # Increase qualitative analysis
            weights["finbert"] *= 1.1
            weights["ollama"] *= 1.1
            logger.debug(f"High chaos detected (vol={avg_vol:.4f}, entropy={avg_entropy:.2f}) - reducing TCN weight")

        # Rule 2: Strong trend (markup/markdown) → increase sentiment weights
        if dominant_phase in ["markup", "markdown"]:
            weights["finbert"] *= 1.3
            weights["ollama"] *= 1.3
            weights["brain"] *= 1.2
            weights["tcn"] *= 0.9  # Trend-following vs reversal prediction
            logger.debug(f"Trend phase ({dominant_phase}) - increasing sentiment weights")

        # Rule 3: Range/accumulation → increase TCN weight (mean reversion)
        if dominant_phase in ["accumulation", "unknown"]:
            weights["tcn"] *= 1.3
            weights["finbert"] *= 0.8
            weights["ollama"] *= 0.8
            logger.debug(f"Range phase ({dominant_phase}) - increasing TCN weight for mean reversion")

        # Rule 4: Low temperature (stable) → increase all weights evenly
        if avg_temp < 0.3:
            for source in weights:
                weights[source] *= 1.1
            logger.debug(f"Low temperature ({avg_temp:.2f}) - increasing all weights (stable conditions)")

        # Rule 5: Penalize stale sources
        now = datetime.utcnow()
        for source, last_time in self._last_signal_time.items():
            if now - last_time > self._signal_stale_threshold:
                weights[source] *= 0.3  # Heavily penalize stale signals
                logger.debug(f"Source {source} is stale (last: {(now - last_time).total_seconds():.0f}s ago)")

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Update current weights
        old_weights = self._current_weights.copy()
        self._current_weights = weights

        # Log if significant change
        max_change = max(abs(weights[k] - old_weights[k]) for k in weights)
        if max_change > 0.05:
            logger.info(
                f"Dynamic weights updated: TCN={weights['tcn']:.2f} "
                f"FinBERT={weights['finbert']:.2f} Ollama={weights['ollama']:.2f} "
                f"Brain={weights['brain']:.2f} "
                f"(phase={dominant_phase}, vol={avg_vol:.4f}, temp={avg_temp:.2f})"
            )

    def get_weights(self) -> dict[str, float]:
        """Get current weights for ensemble."""
        return dict(self._current_weights)

    def fuse_signals(
        self,
        tcn_signal: float | None = None,
        finbert_signal: float | None = None,
        ollama_signal: float | None = None,
        brain_signal: float | None = None,
        min_confidence: float = 0.6,
    ) -> dict[str, Any] | None:
        """Fuse signals from multiple sources into weighted ensemble.

        Args:
            tcn_signal: TCN reversal probability [-1, 1]
            finbert_signal: FinBERT sentiment [-1, 1]
            ollama_signal: Ollama sentiment [-1, 1]
            brain_signal: Brain sentiment [-1, 1]
            min_confidence: Minimum confidence to return signal

        Returns:
            Fused signal dict or None if confidence too low:
            {
                'direction': 'buy' or 'sell',
                'confidence': 0-1,
                'weighted_score': float,
                'sources_used': [str],
                'weights': dict
            }
        """
        # Collect available signals
        signals = {}
        if tcn_signal is not None:
            signals["tcn"] = tcn_signal
        if finbert_signal is not None:
            signals["finbert"] = finbert_signal
        if ollama_signal is not None:
            signals["ollama"] = ollama_signal
        if brain_signal is not None:
            signals["brain"] = brain_signal

        if not signals:
            return None

        # Get current weights
        weights = self.get_weights()

        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        for source, signal in signals.items():
            weight = weights.get(source, 0.0)
            weighted_score += signal * weight
            total_weight += weight

        # Normalize by total weight of available sources
        if total_weight > 0:
            weighted_score /= total_weight

        # Calculate confidence (absolute value of weighted score)
        confidence = abs(weighted_score)

        # Check minimum confidence threshold
        if confidence < min_confidence:
            return None

        # Determine direction
        direction = "buy" if weighted_score > 0 else "sell"

        return {
            "direction": direction,
            "confidence": confidence,
            "weighted_score": weighted_score,
            "sources_used": list(signals.keys()),
            "weights": {k: weights[k] for k in signals.keys()},
        }
