"""Cortex Decision Engine — Strategic orchestration for ARCHON.

The Cortex evaluates system health, signal pipeline performance, risk state,
and market conditions to make strategic decisions. It issues advisory directives
to components via EventBus events.
"""

import asyncio
import time
from enum import Enum
from typing import Any

from hean.archon.directives import Directive, DirectiveAck, DirectiveType
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class SystemMode(str, Enum):
    """System operational modes."""

    AGGRESSIVE = "aggressive"  # All strategies active, full sizing
    NORMAL = "normal"  # Standard operation
    DEFENSIVE = "defensive"  # Reduced position sizes, fewer strategies
    EMERGENCY = "emergency"  # Minimal trading, closing positions


class Cortex:
    """Strategic decision engine for ARCHON.

    Evaluates system health, signal pipeline performance, risk state,
    and market conditions. Issues Directives to components.

    IMPORTANT: Directives are ADVISORY — published as events, not forced.
    Components opt-in by subscribing to ARCHON_DIRECTIVE events.
    """

    def __init__(
        self,
        bus: EventBus,
        health_matrix: Any = None,
        signal_pipeline: Any = None,
        interval_sec: int = 30,
    ) -> None:
        """Initialize Cortex decision engine.

        Args:
            bus: EventBus for publishing directives
            health_matrix: HealthMatrix instance for composite health score
            signal_pipeline: SignalPipelineManager for fill rate metrics
            interval_sec: Decision loop interval in seconds
        """
        self._bus = bus
        self._health_matrix = health_matrix
        self._signal_pipeline = signal_pipeline
        self._interval_sec = interval_sec

        self._mode = SystemMode.NORMAL
        self._directives_issued: list[Directive] = []
        self._last_evaluation: dict[str, Any] = {}
        self._running = False
        self._decision_task: asyncio.Task[None] | None = None

        # Metrics for tracking directive outcomes
        self._metrics = {
            "total_directives_issued": 0,
            "mode_changes": 0,
            "pause_directives": 0,
            "resume_directives": 0,
            "last_decision_time": 0.0,
        }

    async def start(self) -> None:
        """Launch decision loop as asyncio.Task."""
        if self._running:
            logger.warning("[Cortex] Already running")
            return

        self._running = True
        self._decision_task = asyncio.create_task(self._decision_loop())
        logger.info(
            f"[Cortex] Started — decision interval {self._interval_sec}s, mode={self._mode.value}"
        )

    async def stop(self) -> None:
        """Stop decision loop."""
        if not self._running:
            return

        self._running = False
        if self._decision_task:
            self._decision_task.cancel()
            try:
                await self._decision_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"[Cortex] Stopped — issued {self._metrics['total_directives_issued']} "
            f"directives, {self._metrics['mode_changes']} mode changes"
        )

    async def _decision_loop(self) -> None:
        """Main loop: sleep → evaluate → issue directives → repeat."""
        while self._running:
            try:
                await asyncio.sleep(self._interval_sec)

                start_time = time.time()
                directives = await self._evaluate()
                elapsed_ms = (time.time() - start_time) * 1000

                # Execute directives
                for directive in directives:
                    await self._execute_directive(directive)

                self._metrics["last_decision_time"] = time.time()
                logger.debug(
                    f"[Cortex] Decision cycle complete in {elapsed_ms:.1f}ms, "
                    f"issued {len(directives)} directives"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Cortex] Decision loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Brief pause before retry

    async def _evaluate(self) -> list[Directive]:
        """One evaluation cycle. Returns list of directives to issue.

        Decision rules:
        1. health_score < 40 → PAUSE_TRADING, mode=EMERGENCY
        2. health_score < 60 → mode=DEFENSIVE
        3. health_score > 80 and mode != NORMAL → mode=NORMAL
        4. health_score > 90 → mode=AGGRESSIVE (if market conditions allow)
        5. signal fill_rate < 30% → log warning, investigate
        6. dead_letters growing fast (>10 in last interval) → log alert
        """
        directives: list[Directive] = []
        evaluation: dict[str, Any] = {
            "timestamp": time.time(),
            "mode": self._mode.value,
            "decisions": [],
        }

        # Get health score
        health_score = 100.0  # Default if no health matrix
        if self._health_matrix:
            try:
                health_score = self._health_matrix.get_composite_score()
                evaluation["health_score"] = health_score
            except Exception as e:
                logger.error(f"[Cortex] Failed to get health score: {e}")

        # Get signal pipeline metrics
        fill_rate = 100.0  # Default if no pipeline
        dead_letter_count = 0
        if self._signal_pipeline:
            try:
                status = self._signal_pipeline.get_status()
                fill_rate = status.get("fill_rate_pct", 100.0)
                dead_letter_count = status.get("dead_letter_count", 0)
                evaluation["fill_rate_pct"] = fill_rate
                evaluation["dead_letter_count"] = dead_letter_count
            except Exception as e:
                logger.error(f"[Cortex] Failed to get pipeline metrics: {e}")

        # Rule 1: Critical health — emergency mode
        if health_score < 40:
            if self._mode != SystemMode.EMERGENCY:
                self._mode = SystemMode.EMERGENCY
                self._metrics["mode_changes"] += 1
                evaluation["decisions"].append("health_score < 40 → EMERGENCY mode + PAUSE_TRADING")
                logger.warning(
                    f"[Cortex] Critical health ({health_score:.1f}/100) → EMERGENCY mode"
                )

                directives.append(
                    Directive(
                        directive_type=DirectiveType.PAUSE_TRADING,
                        target_component="trading_system",
                        params={"reason": "critical_health", "health_score": health_score},
                    )
                )
                self._metrics["pause_directives"] += 1

        # Rule 2: Low health — defensive mode
        elif health_score < 60:
            if self._mode not in (SystemMode.DEFENSIVE, SystemMode.EMERGENCY):
                self._mode = SystemMode.DEFENSIVE
                self._metrics["mode_changes"] += 1
                evaluation["decisions"].append("health_score < 60 → DEFENSIVE mode")
                logger.info(f"[Cortex] Low health ({health_score:.1f}/100) → DEFENSIVE mode")

        # Rule 3: Good health — recover to normal
        elif health_score > 80 and self._mode != SystemMode.NORMAL:
            prev_mode = self._mode
            self._mode = SystemMode.NORMAL
            self._metrics["mode_changes"] += 1
            evaluation["decisions"].append(
                f"health_score > 80 → NORMAL mode (from {prev_mode.value})"
            )
            logger.info(
                f"[Cortex] Health recovered ({health_score:.1f}/100) → "
                f"NORMAL mode (from {prev_mode.value})"
            )

            # If recovering from emergency, issue resume directive
            if prev_mode == SystemMode.EMERGENCY:
                directives.append(
                    Directive(
                        directive_type=DirectiveType.RESUME_TRADING,
                        target_component="trading_system",
                        params={"reason": "health_recovered", "health_score": health_score},
                    )
                )
                self._metrics["resume_directives"] += 1

        # Rule 4: Excellent health — aggressive mode
        elif health_score > 90 and self._mode != SystemMode.AGGRESSIVE:
            # Only switch to aggressive in favorable market conditions
            # For now, enable if health is excellent
            self._mode = SystemMode.AGGRESSIVE
            self._metrics["mode_changes"] += 1
            evaluation["decisions"].append("health_score > 90 → AGGRESSIVE mode")
            logger.info(f"[Cortex] Excellent health ({health_score:.1f}/100) → AGGRESSIVE mode")

        # Rule 5: Low fill rate warning
        if fill_rate < 30.0:
            evaluation["decisions"].append(
                f"fill_rate < 30% ({fill_rate:.1f}%) → investigate pipeline"
            )
            logger.warning(
                f"[Cortex] Low signal fill rate: {fill_rate:.1f}%. "
                f"Check signal pipeline for bottlenecks or blocking."
            )

        # Rule 6: Growing dead letter queue
        if dead_letter_count > 10:
            evaluation["decisions"].append(
                f"dead_letters > 10 ({dead_letter_count}) → investigate failures"
            )
            logger.warning(
                f"[Cortex] Dead letter queue growing: {dead_letter_count} signals. "
                f"Check risk blocking, order rejection, or timeouts."
            )

        self._last_evaluation = evaluation
        return directives

    async def _execute_directive(self, directive: Directive) -> DirectiveAck:
        """Execute a directive by publishing it to EventBus.

        Returns:
            DirectiveAck indicating success/failure
        """
        try:
            # Publish directive as event for opt-in components
            await self._bus.publish(
                Event(
                    event_type=EventType.ARCHON_DIRECTIVE,
                    data={
                        "directive": directive,
                        "directive_type": directive.directive_type.value,
                        "target_component": directive.target_component,
                        "params": directive.params,
                    },
                )
            )

            # Track issued directives
            self._directives_issued.append(directive)
            self._metrics["total_directives_issued"] += 1

            logger.info(
                f"[Cortex] Issued directive: {directive.directive_type.value} → "
                f"{directive.target_component}"
            )

            return DirectiveAck(
                directive_id=directive.directive_id,
                component_id="cortex",
                success=True,
                result={"published": True},
            )

        except Exception as e:
            logger.error(
                f"[Cortex] Failed to execute directive {directive.directive_type.value}: {e}",
                exc_info=True,
            )
            return DirectiveAck(
                directive_id=directive.directive_id,
                component_id="cortex",
                success=False,
                error=str(e),
            )

    def get_status(self) -> dict[str, Any]:
        """Get current Cortex status for API introspection.

        Returns:
            Status dictionary with mode, evaluation, directives
        """
        return {
            "running": self._running,
            "mode": self._mode.value,
            "interval_sec": self._interval_sec,
            "last_evaluation": self._last_evaluation,
            "total_directives_issued": self._metrics["total_directives_issued"],
            "mode_changes": self._metrics["mode_changes"],
            "pause_directives": self._metrics["pause_directives"],
            "resume_directives": self._metrics["resume_directives"],
            "last_decision_time": self._metrics["last_decision_time"],
            "recent_directives": [
                {
                    "directive_type": d.directive_type.value,
                    "target_component": d.target_component,
                    "params": d.params,
                    "issued_at": d.issued_at.isoformat(),
                }
                for d in self._directives_issued[-10:]
            ],
        }
