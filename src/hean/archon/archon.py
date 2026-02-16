"""ARCHON — Central Brain-Orchestrator."""

from typing import Any

from hean.archon.chronicle import Chronicle
from hean.archon.cortex import Cortex
from hean.archon.health_matrix import HealthMatrix
from hean.archon.heartbeat import HeartbeatRegistry
from hean.archon.reconciler import ArchonReconciler
from hean.archon.signal_pipeline_manager import SignalPipelineManager
from hean.config import HEANSettings
from hean.core.bus import EventBus
from hean.logging import get_logger

logger = get_logger(__name__)


class Archon:
    """Central orchestration brain for HEAN Trading System.

    Wraps around existing components without modifying them.
    Adds: signal tracking, health monitoring, reconciliation,
    strategic decisions, and audit trail.
    """

    def __init__(self, bus: EventBus, settings: HEANSettings) -> None:
        """Initialize ARCHON.

        Args:
            bus: EventBus instance
            settings: HEANSettings configuration
        """
        self._bus = bus
        self._settings = settings
        self._running = False

        self.signal_pipeline: SignalPipelineManager | None = None
        self.heartbeat: HeartbeatRegistry | None = None
        self.health_matrix: HealthMatrix | None = None
        self.cortex: Cortex | None = None
        self.reconciler: ArchonReconciler | None = None
        self.chronicle: Chronicle | None = None

    async def start(self, **components: Any) -> None:
        """Start ARCHON sub-systems based on settings.

        Args:
            **components: Optional component instances:
                - accounting: PortfolioAccounting
                - order_manager: OrderManager
                - bybit_http: BybitHTTPClient
        """
        self._running = True
        s = self._settings

        try:
            # Signal Pipeline (if enabled)
            if s.archon_signal_pipeline_enabled:
                try:
                    self.signal_pipeline = SignalPipelineManager(
                        bus=self._bus,
                        max_active=s.archon_max_active_signals,
                        stage_timeout_sec=s.archon_signal_timeout_sec,
                    )
                    await self.signal_pipeline.start()
                    logger.info("[ARCHON] Signal Pipeline started")
                except Exception as e:
                    logger.warning(f"[ARCHON] Signal Pipeline failed to start: {e}")
                    self.signal_pipeline = None

            # Heartbeat Registry
            try:
                self.heartbeat = HeartbeatRegistry(
                    default_interval=s.archon_heartbeat_interval_sec,
                )
                logger.info("[ARCHON] Heartbeat Registry initialized")
            except Exception as e:
                logger.warning(f"[ARCHON] Heartbeat Registry failed: {e}")
                self.heartbeat = None

            # Health Matrix
            try:
                self.health_matrix = HealthMatrix(
                    bus=self._bus,
                    heartbeat=self.heartbeat,
                    signal_pipeline=self.signal_pipeline,
                )
                await self.health_matrix.start()
                logger.info("[ARCHON] Health Matrix started")
            except Exception as e:
                logger.warning(f"[ARCHON] Health Matrix failed to start: {e}")
                self.health_matrix = None

            # Chronicle (if enabled)
            if s.archon_chronicle_enabled:
                try:
                    self.chronicle = Chronicle(
                        bus=self._bus,
                        max_memory=s.archon_chronicle_max_memory,
                    )
                    await self.chronicle.start()
                    logger.info("[ARCHON] Chronicle started")
                except Exception as e:
                    logger.warning(f"[ARCHON] Chronicle failed to start: {e}")
                    self.chronicle = None

            # Reconciler (if enabled)
            if s.archon_reconciliation_enabled:
                accounting = components.get("accounting")
                order_manager = components.get("order_manager")
                bybit_http = components.get("bybit_http")
                if accounting and order_manager and bybit_http:
                    try:
                        self.reconciler = ArchonReconciler(
                            bus=self._bus,
                            accounting=accounting,
                            order_manager=order_manager,
                            bybit_http=bybit_http,
                            interval_sec=s.archon_reconciliation_interval_sec,
                        )
                        await self.reconciler.start()
                        logger.info("[ARCHON] Reconciler started")
                    except Exception as e:
                        logger.warning(f"[ARCHON] Reconciler failed to start: {e}")
                        self.reconciler = None
                else:
                    logger.debug(
                        "[ARCHON] Reconciler disabled — missing required components "
                        f"(accounting={accounting is not None}, "
                        f"order_manager={order_manager is not None}, "
                        f"bybit_http={bybit_http is not None})"
                    )

            # Cortex (if enabled)
            if s.archon_cortex_enabled:
                try:
                    self.cortex = Cortex(
                        bus=self._bus,
                        health_matrix=self.health_matrix,
                        signal_pipeline=self.signal_pipeline,
                        interval_sec=s.archon_cortex_interval_sec,
                    )
                    await self.cortex.start()
                    logger.info("[ARCHON] Cortex started")
                except Exception as e:
                    logger.warning(f"[ARCHON] Cortex failed to start: {e}")
                    self.cortex = None

            logger.info("[ARCHON] Brain-Orchestrator activated")

        except Exception as e:
            logger.error(f"[ARCHON] Failed to start: {e}", exc_info=True)
            # Attempt to stop any started subsystems
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop all sub-systems in reverse order."""
        self._running = False

        # Stop in reverse order of startup
        if self.cortex:
            try:
                await self.cortex.stop()
            except Exception as e:
                logger.error(f"[ARCHON] Error stopping Cortex: {e}", exc_info=True)

        if self.reconciler:
            try:
                await self.reconciler.stop()
            except Exception as e:
                logger.error(f"[ARCHON] Error stopping Reconciler: {e}", exc_info=True)

        if self.chronicle:
            try:
                await self.chronicle.stop()
            except Exception as e:
                logger.error(f"[ARCHON] Error stopping Chronicle: {e}", exc_info=True)

        if self.health_matrix:
            try:
                await self.health_matrix.stop()
            except Exception as e:
                logger.error(f"[ARCHON] Error stopping Health Matrix: {e}", exc_info=True)

        if self.signal_pipeline:
            try:
                await self.signal_pipeline.stop()
            except Exception as e:
                logger.error(f"[ARCHON] Error stopping Signal Pipeline: {e}", exc_info=True)

        logger.info("[ARCHON] Brain-Orchestrator deactivated")

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive ARCHON status."""
        return {
            "running": self._running,
            "signal_pipeline": (
                self.signal_pipeline.get_status() if self.signal_pipeline else None
            ),
            "health": (self.health_matrix.get_composite_score() if self.health_matrix else None),
            "heartbeats": self.heartbeat.get_status() if self.heartbeat else None,
            "cortex": self.cortex.get_status() if self.cortex else None,
            "reconciler_active": self.reconciler is not None,
            "reconciler": self.reconciler.get_status() if self.reconciler else None,
            "chronicle_active": self.chronicle is not None,
        }
