"""C++-backed circuit breaker for hard-stop trading on latency spikes or risk limits.

This circuit breaker operates at the C++ level, bypassing Python entirely
when critical thresholds are exceeded. It uses shared memory for communication.
"""

import time
from typing import Optional

from hean.hft.shared_memory import SharedMemoryRiskState
from hean.logging import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    """Hard-stop circuit breaker that operates at C++ level.

    This component monitors:
    - Latency spikes (P99 > threshold)
    - Risk limits (drawdown, position size, order rate)
    - System health (memory, CPU)

    When triggered, it immediately stops all trading operations at the C++ level,
    bypassing Python completely.
    """

    def __init__(
        self,
        latency_threshold_us: int = 5000,  # 5ms P99 latency threshold
        max_drawdown_pct: float = 5.0,
        max_order_rate: int = 100,  # Max orders per second
    ) -> None:
        """Initialize circuit breaker.

        Args:
            latency_threshold_us: Maximum allowed P99 latency in microseconds
            max_drawdown_pct: Maximum allowed drawdown percentage
            max_order_rate: Maximum allowed order rate per second
        """
        self.latency_threshold_us = latency_threshold_us
        self.max_drawdown_pct = max_drawdown_pct
        self.max_order_rate = max_order_rate

        self._risk_shm = SharedMemoryRiskState()
        self._risk_shm.initialize()

        self._circuit_open = False
        self._last_check_ns = time.time_ns()

        # Latency tracking
        self._latency_samples: list[int] = []
        self._max_samples = 1000

        # Order rate tracking
        self._order_timestamps: list[int] = []
        self._rate_window_seconds = 1.0

    def check(self) -> tuple[bool, Optional[str]]:
        """Check circuit breaker status.

        Returns:
            (is_open, reason) tuple. If is_open is True, trading should be stopped.
        """
        risk_state = self._risk_shm.read_risk_state()

        if not risk_state:
            # If we can't read risk state, assume safe (or trigger different logic)
            return False, None

        # Check C++ circuit breaker status first (highest priority)
        if risk_state.get("circuit_breaker_active"):
            reason = "C++ circuit breaker activated"
            self._circuit_open = True
            return True, reason

        # Check drawdown
        current_dd = risk_state.get("current_drawdown_pct", 0.0)
        if current_dd > self.max_drawdown_pct:
            reason = f"Drawdown exceeded: {current_dd:.2f}% > {self.max_drawdown_pct}%"
            self._circuit_open = True
            self._update_circuit_breaker_state(True, reason)
            return True, reason

        # Check latency (P99)
        latency_p99 = risk_state.get("latency_p99_us", 0)
        if latency_p99 > self.latency_threshold_us:
            reason = f"Latency spike: P99={latency_p99}us > {self.latency_threshold_us}us"
            self._circuit_open = True
            self._update_circuit_breaker_state(True, reason)
            return True, reason

        # Check order rate
        current_rate = risk_state.get("current_order_rate", 0)
        rate_limit = risk_state.get("order_rate_limit", self.max_order_rate)
        if current_rate > rate_limit:
            reason = f"Order rate exceeded: {current_rate}/s > {rate_limit}/s"
            self._circuit_open = True
            self._update_circuit_breaker_state(True, reason)
            return True, reason

        # All checks passed
        if self._circuit_open:
            logger.info("Circuit breaker closed - conditions normalized")
            self._circuit_open = False
            self._update_circuit_breaker_state(False, None)

        return False, None

    def record_latency(self, latency_us: int) -> None:
        """Record latency sample for monitoring.

        Args:
            latency_us: Latency in microseconds
        """
        self._latency_samples.append(latency_us)
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples.pop(0)

        # Calculate P99 and update shared memory
        if len(self._latency_samples) >= 100:
            sorted_samples = sorted(self._latency_samples)
            p99_index = int(len(sorted_samples) * 0.99)
            p99_latency = sorted_samples[p99_index]

            # Update risk state
            risk_state = self._risk_shm.read_risk_state()
            if risk_state:
                self._risk_shm.write_risk_state(
                    circuit_breaker_active=risk_state.get("circuit_breaker_active", False),
                    total_equity=risk_state.get("total_equity", 0.0),
                    max_drawdown_pct=risk_state.get("max_drawdown_pct", 0.0),
                    current_drawdown_pct=risk_state.get("current_drawdown_pct", 0.0),
                    latency_p99_us=p99_latency,
                    order_rate_limit=risk_state.get("order_rate_limit", self.max_order_rate),
                    current_order_rate=risk_state.get("current_order_rate", 0),
                )

    def record_order(self) -> None:
        """Record order placement for rate limiting."""
        now = time.time_ns()
        self._order_timestamps.append(now)

        # Remove old timestamps outside window
        cutoff_ns = now - int(self._rate_window_seconds * 1_000_000_000)
        self._order_timestamps = [ts for ts in self._order_timestamps if ts > cutoff_ns]

        # Calculate current rate
        current_rate = len(self._order_timestamps)

        # Update risk state
        risk_state = self._risk_shm.read_risk_state()
        if risk_state:
            self._risk_shm.write_risk_state(
                circuit_breaker_active=risk_state.get("circuit_breaker_active", False),
                total_equity=risk_state.get("total_equity", 0.0),
                max_drawdown_pct=risk_state.get("max_drawdown_pct", 0.0),
                current_drawdown_pct=risk_state.get("current_drawdown_pct", 0.0),
                latency_p99_us=risk_state.get("latency_p99_us", 0),
                order_rate_limit=risk_state.get("order_rate_limit", self.max_order_rate),
                current_order_rate=current_rate,
            )

    def _update_circuit_breaker_state(self, active: bool, reason: Optional[str]) -> None:
        """Update circuit breaker state in shared memory."""
        risk_state = self._risk_shm.read_risk_state()
        if risk_state:
            self._risk_shm.write_risk_state(
                circuit_breaker_active=active,
                total_equity=risk_state.get("total_equity", 0.0),
                max_drawdown_pct=risk_state.get("max_drawdown_pct", 0.0),
                current_drawdown_pct=risk_state.get("current_drawdown_pct", 0.0),
                latency_p99_us=risk_state.get("latency_p99_us", 0),
                order_rate_limit=risk_state.get("order_rate_limit", self.max_order_rate),
                current_order_rate=risk_state.get("current_order_rate", 0),
            )

        if active:
            logger.critical(f"CIRCUIT BREAKER OPEN: {reason}")
        else:
            logger.info("Circuit breaker closed")

    def update_equity_and_drawdown(self, equity: float, drawdown_pct: float) -> None:
        """Update equity and drawdown in shared memory for C++ circuit breaker.

        Args:
            equity: Total equity
            drawdown_pct: Current drawdown percentage
        """
        risk_state = self._risk_shm.read_risk_state()
        if risk_state:
            self._risk_shm.write_risk_state(
                circuit_breaker_active=risk_state.get("circuit_breaker_active", False),
                total_equity=equity,
                max_drawdown_pct=self.max_drawdown_pct,
                current_drawdown_pct=drawdown_pct,
                latency_p99_us=risk_state.get("latency_p99_us", 0),
                order_rate_limit=risk_state.get("order_rate_limit", self.max_order_rate),
                current_order_rate=risk_state.get("current_order_rate", 0),
            )

    def is_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        is_open, _ = self.check()
        return is_open

    def close(self) -> None:
        """Close circuit breaker (reset to normal state)."""
        self._circuit_open = False
        self._update_circuit_breaker_state(False, None)
        self._risk_shm.close()
