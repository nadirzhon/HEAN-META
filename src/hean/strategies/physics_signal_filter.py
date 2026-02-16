"""Physics-Based Signal Filter.

Blocks or modifies trading signals based on market thermodynamics:
- Blocks signals in extreme chaos (high temperature + high entropy)
- Blocks signals that conflict with market phase
- Blocks signals when phase confidence is too low
- Enriches signals with physics metadata for downstream decision-making
"""

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class PhysicsSignalFilter:
    """Filter trading signals based on physics state."""

    def __init__(
        self,
        bus: EventBus,
        enabled: bool = True,
        strict: bool = True,
    ) -> None:
        """Initialize the physics signal filter.

        Args:
            bus: Event bus for subscribing to events
            enabled: Whether physics filtering is enabled
            strict: If True, blocks counter-phase signals; if False, only penalizes
        """
        self._bus = bus
        self._enabled = enabled
        self._strict = strict
        self._physics_cache: dict[str, dict] = {}
        self._running = False

        # Counters for monitoring
        self._stats = {
            "signals_received": 0,
            "signals_passed": 0,
            "signals_blocked": 0,
            "block_reasons": {
                "low_phase_confidence": 0,
                "extreme_chaos": 0,
                "phase_conflict": 0,
                "extreme_temperature": 0,
                "extreme_entropy": 0,
            },
        }

    async def start(self) -> None:
        """Start the physics signal filter."""
        if not self._enabled:
            logger.info("PhysicsSignalFilter disabled (PHYSICS_FILTER_ENABLED=false)")
            return

        self._running = True
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.subscribe(EventType.SIGNAL, self._handle_signal)
        logger.info(
            f"PhysicsSignalFilter started (strict={self._strict})"
        )

    async def stop(self) -> None:
        """Stop the physics signal filter."""
        if not self._enabled:
            return

        self._running = False
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.unsubscribe(EventType.SIGNAL, self._handle_signal)
        logger.info("PhysicsSignalFilter stopped")

    async def _handle_physics_update(self, event: Event) -> None:
        """Cache physics state for each symbol."""
        if not self._running:
            return

        symbol = event.data.get("symbol")
        physics = event.data.get("physics")

        if symbol and physics:
            self._physics_cache[symbol] = physics

    async def _handle_signal(self, event: Event) -> None:
        """Filter signals based on physics state."""
        if not self._running:
            return

        signal = event.data.get("signal")
        if not signal:
            return

        self._stats["signals_received"] += 1

        symbol = signal.symbol
        side = signal.side.lower()

        # Get physics state
        physics = self._physics_cache.get(symbol)
        if not physics:
            # No physics data - pass through with warning
            logger.warning(
                f"[PhysicsFilter] No physics data for {symbol}, passing signal through"
            )
            self._stats["signals_passed"] += 1
            return

        # Extract physics metrics
        temperature = physics.get("temperature", 0.5)
        entropy = physics.get("entropy", 0.5)
        phase = physics.get("phase", "unknown")
        phase_confidence = physics.get("phase_confidence", 0.0)

        # Rule 1: Block if phase confidence too low
        if phase_confidence < 0.5:
            self._block_signal(
                signal,
                "low_phase_confidence",
                f"Phase confidence {phase_confidence:.2f} < 0.5",
            )
            return

        # Rule 2: Block in extreme chaos (high temp + high entropy)
        if temperature > 0.75 and entropy > 0.75:
            self._block_signal(
                signal,
                "extreme_chaos",
                f"Extreme chaos: temp={temperature:.2f}, entropy={entropy:.2f}",
            )
            return

        # Rule 3: Block if signal direction conflicts with phase
        if self._is_phase_conflict(side, phase):
            if self._strict:
                self._block_signal(
                    signal,
                    "phase_conflict",
                    f"{side} signal conflicts with {phase} phase",
                )
                return
            else:
                # Non-strict: reduce confidence but allow
                logger.info(
                    f"[PhysicsFilter] {symbol}: {side} vs {phase} - "
                    f"reducing confidence (strict=false)"
                )
                signal.confidence *= 0.5
                signal.metadata["physics_penalty"] = "phase_conflict"

        # Rule 4: Block in extreme conditions
        if temperature > 0.85:
            self._block_signal(
                signal,
                "extreme_temperature",
                f"Extreme temperature {temperature:.2f} > 0.85",
            )
            return

        if entropy > 0.90:
            self._block_signal(
                signal,
                "extreme_entropy",
                f"Extreme entropy {entropy:.2f} > 0.90",
            )
            return

        # Signal passed all filters - enrich with physics metadata
        signal.metadata.update({
            "physics_temperature": temperature,
            "physics_entropy": entropy,
            "physics_phase": phase,
            "physics_phase_confidence": phase_confidence,
            "physics_filtered": True,
        })

        self._stats["signals_passed"] += 1
        logger.debug(
            f"[PhysicsFilter] {symbol} {side} PASSED: "
            f"phase={phase} (conf={phase_confidence:.2f}), "
            f"temp={temperature:.2f}, entropy={entropy:.2f}"
        )

    def _is_phase_conflict(self, side: str, phase: str) -> bool:
        """Check if signal side conflicts with market phase.

        Conflicts:
        - buy + distribution (selling pressure)
        - buy + markdown (downtrend)
        - sell + accumulation (buying pressure)
        - sell + markup (uptrend)
        """
        phase_lower = phase.lower()

        if side == "buy":
            return phase_lower in ("distribution", "markdown")
        elif side == "sell":
            return phase_lower in ("accumulation", "markup")

        return False

    def _block_signal(self, signal, reason: str, details: str) -> None:
        """Block a signal and publish ORDER_DECISION with rejection reason."""
        self._stats["signals_blocked"] += 1
        self._stats["block_reasons"][reason] += 1

        logger.info(
            f"[PhysicsFilter] BLOCKED {signal.symbol} {signal.side}: "
            f"{reason} - {details}"
        )

        # Publish ORDER_DECISION event with blocking reason
        decision_data = {
            "symbol": signal.symbol,
            "side": signal.side,
            "strategy_id": signal.strategy_id,
            "decision": "reject",
            "reason_code": f"physics_{reason}",
            "reason": details,
            "signal_id": signal.metadata.get("signal_id", "unknown"),
        }

        self._bus.publish(
            Event(
                event_type=EventType.ORDER_DECISION,
                data=decision_data,
            )
        )

    def get_stats(self) -> dict:
        """Get filter statistics."""
        total = self._stats["signals_received"]
        if total == 0:
            pass_rate = 0.0
            block_rate = 0.0
        else:
            pass_rate = self._stats["signals_passed"] / total * 100
            block_rate = self._stats["signals_blocked"] / total * 100

        return {
            **self._stats,
            "pass_rate_pct": pass_rate,
            "block_rate_pct": block_rate,
        }
