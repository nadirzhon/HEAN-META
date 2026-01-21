"""
Topological Watchdog: Halts trading if market manifold becomes disconnected
Indicates flash-crash, API lag, or market structural collapse.
"""

from typing import Any, Optional
from datetime import datetime, timedelta

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class TopologicalWatchdog:
    """Topological Watchdog that halts trading on disconnected market manifold.
    
    Monitors TDA topology scores continuously. If manifold becomes disconnected
    (indicating flash-crash or API lag), immediately halts all trading activity.
    """
    
    def __init__(self, bus: EventBus) -> None:
        """Initialize the Topological Watchdog.
        
        Args:
            bus: Event bus for publishing halt events
        """
        self._bus = bus
        self._running = False
        self._halt_active = False
        self._disconnected_since: Optional[datetime] = None
        self._disconnection_threshold_seconds = 2.0  # Halt if disconnected > 2 seconds
        self._last_topology_score = 1.0
        
        # FastWarden for topology monitoring
        self._fast_warden: Optional[Any] = None
        try:
            import graph_engine_py  # type: ignore
            self._fast_warden = graph_engine_py.FastWarden()
            logger.info("Topological Watchdog initialized with FastWarden")
        except ImportError:
            logger.warning("FastWarden not available. Watchdog will use fallback monitoring.")
    
    async def start(self) -> None:
        """Start the Topological Watchdog."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._monitor_topology)
        logger.info("Topological Watchdog started - monitoring market manifold connectivity")
    
    async def stop(self) -> None:
        """Stop the Topological Watchdog."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._monitor_topology)
        
        # Clear halt if active
        if self._halt_active:
            await self._clear_halt()
        
        logger.info("Topological Watchdog stopped")
    
    async def _monitor_topology(self, event: Event) -> None:
        """Monitor market topology and detect disconnection.
        
        Continuously checks if market manifold is disconnected.
        If disconnected for > threshold, triggers trading halt.
        """
        if not self._running:
            return
        
        # Check if market manifold is disconnected
        is_disconnected = False
        
        if self._fast_warden:
            try:
                is_disconnected = self._fast_warden.is_market_disconnected()
                self._last_topology_score = self._fast_warden.get_market_topology_score()
            except Exception as e:
                logger.warning(f"Failed to check topology from FastWarden: {e}")
                # Assume connected on error
                is_disconnected = False
        else:
            # Fallback: check topology score
            # If score drops below 0.2, consider disconnected
            is_disconnected = self._last_topology_score < 0.2
        
        now = datetime.utcnow()
        
        if is_disconnected:
            # Market manifold is disconnected
            if not self._halt_active:
                # First detection: record timestamp
                if self._disconnected_since is None:
                    self._disconnected_since = now
                    logger.warning(
                        f"Market manifold disconnection detected: "
                        f"topology_score={self._last_topology_score:.3f}"
                    )
                else:
                    # Check if disconnected for > threshold
                    disconnected_duration = (now - self._disconnected_since).total_seconds()
                    if disconnected_duration >= self._disconnection_threshold_seconds:
                        # Trigger trading halt
                        await self._trigger_halt(
                            reason=f"Market manifold disconnected for {disconnected_duration:.1f}s, "
                                   f"topology_score={self._last_topology_score:.3f}"
                        )
            else:
                # Already halted, keep monitoring
                disconnected_duration = (now - self._disconnected_since).total_seconds() if self._disconnected_since else 0.0
                logger.debug(
                    f"Trading halted: manifold disconnected for {disconnected_duration:.1f}s, "
                    f"topology_score={self._last_topology_score:.3f}"
                )
        else:
            # Market manifold is connected
            if self._halt_active:
                # Manifold reconnected: clear halt
                if self._disconnected_since:
                    disconnected_duration = (now - self._disconnected_since).total_seconds()
                    logger.info(
                        f"Market manifold reconnected after {disconnected_duration:.1f}s, "
                        f"topology_score={self._last_topology_score:.3f}"
                    )
                await self._clear_halt()
            
            # Reset disconnection timestamp
            self._disconnected_since = None
    
    async def _trigger_halt(self, reason: str) -> None:
        """Trigger trading halt due to topological disconnection.
        
        Args:
            reason: Reason for halt
        """
        if self._halt_active:
            return  # Already halted
        
        self._halt_active = True
        
        logger.critical(
            f"🚨 TOPOLOGICAL WATCHDOG: TRADING HALTED - {reason}"
        )
        
        # Publish STOP_TRADING event
        from hean.core.types import EventType
        await self._bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,
                data={
                    "reason": f"Topological Watchdog: {reason}",
                    "topology_score": self._last_topology_score,
                    "watchdog": True,
                },
            )
        )
        
        logger.critical(
            f"Trading halted by Topological Watchdog. Market manifold disconnected. "
            f"Reason: {reason}"
        )
    
    async def _clear_halt(self) -> None:
        """Clear trading halt after manifold reconnection."""
        if not self._halt_active:
            return
        
        self._halt_active = False
        self._disconnected_since = None
        
        logger.info(
            "✅ TOPOLOGICAL WATCHDOG: Trading halt cleared. Market manifold reconnected."
        )
        
        # Note: We don't automatically restart trading here
        # The main system should handle restart after manual review
    
    def is_halted(self) -> bool:
        """Check if trading is currently halted by watchdog.
        
        Returns:
            True if halted, False otherwise
        """
        return self._halt_active
    
    def get_topology_status(self) -> dict[str, any]:
        """Get current topology monitoring status.
        
        Returns:
            Status dictionary
        """
        disconnected_duration = 0.0
        if self._disconnected_since:
            disconnected_duration = (datetime.utcnow() - self._disconnected_since).total_seconds()
        
        return {
            "halt_active": self._halt_active,
            "topology_score": self._last_topology_score,
            "is_disconnected": self._disconnected_since is not None,
            "disconnected_duration_seconds": disconnected_duration,
            "disconnection_threshold_seconds": self._disconnection_threshold_seconds,
        }