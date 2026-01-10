"""Zero-copy shared memory for high-speed market data and risk enforcement.

This module provides a Python interface to C++ shared memory backend for
ultra-low latency data exchange between components.
"""

import mmap
import os
import struct
from ctypes import Structure, c_double, c_int64, c_uint64
from typing import Optional

from hean.logging import get_logger

logger = get_logger(__name__)

# Memory-mapped file for market data (zero-copy)
MARKET_DATA_SHM_NAME = "/hean_market_data"
RISK_STATE_SHM_NAME = "/hean_risk_state"

# Market data structure size (symbol: 16 bytes, price: 8 bytes, timestamp: 8 bytes, etc.)
MARKET_DATA_ENTRY_SIZE = 64  # bytes per entry
MAX_MARKET_DATA_ENTRIES = 10000

# Risk state structure size
RISK_STATE_SIZE = 256  # bytes


class MarketDataEntry(Structure):
    """Market data entry in shared memory (C-compatible)."""

    _fields_ = [
        ("symbol_hash", c_uint64),  # Hash of symbol for fast lookup
        ("price", c_double),
        ("bid", c_double),
        ("ask", c_double),
        ("timestamp_ns", c_int64),  # Nanoseconds since epoch
        ("volume", c_double),
        ("latency_us", c_uint64),  # Latency in microseconds
    ]


class RiskState(Structure):
    """Risk state in shared memory (C-compatible)."""

    _fields_ = [
        ("circuit_breaker_active", c_uint64),  # 0 = off, 1 = on
        ("total_equity", c_double),
        ("max_drawdown_pct", c_double),
        ("current_drawdown_pct", c_double),
        ("latency_p99_us", c_uint64),  # P99 latency in microseconds
        ("order_rate_limit", c_uint64),  # Max orders per second
        ("current_order_rate", c_uint64),
        ("last_update_ns", c_int64),
    ]


class SharedMemoryMarketData:
    """Zero-copy shared memory interface for market data."""

    def __init__(self, shm_name: str = MARKET_DATA_SHM_NAME) -> None:
        """Initialize shared memory for market data.

        Args:
            shm_name: Name of shared memory region
        """
        self.shm_name = shm_name
        self.shm_size = MARKET_DATA_ENTRY_SIZE * MAX_MARKET_DATA_ENTRIES
        self._mmap: Optional[mmap.mmap] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize shared memory region.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create memory-mapped file (will be created by C++ component)
            # For now, we'll create it here as a fallback
            if not os.path.exists(f"/dev/shm{self.shm_name}"):
                # Create file-backed shared memory
                fd = os.open(f"/tmp{self.shm_name}", os.O_CREAT | os.O_RDWR, 0o666)
                os.ftruncate(fd, self.shm_size)
                self._mmap = mmap.mmap(fd, self.shm_size, access=mmap.ACCESS_WRITE)
                os.close(fd)
            else:
                fd = os.open(f"/dev/shm{self.shm_name}", os.O_RDWR)
                self._mmap = mmap.mmap(fd, self.shm_size, access=mmap.ACCESS_WRITE)
                os.close(fd)

            self._initialized = True
            logger.info(f"Shared memory initialized: {self.shm_name} ({self.shm_size} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize shared memory: {e}")
            return False

    def write_tick(
        self,
        symbol: str,
        price: float,
        bid: float,
        ask: float,
        volume: float,
        timestamp_ns: int,
    ) -> bool:
        """Write tick data to shared memory (zero-copy).

        Args:
            symbol: Trading symbol
            price: Current price
            bid: Best bid
            ask: Best ask
            volume: Volume
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            True if successful
        """
        if not self._initialized or not self._mmap:
            return False

        try:
            # Find first available slot (circular buffer)
            # For now, use simple hash-based slot selection
            symbol_hash = hash(symbol) % MAX_MARKET_DATA_ENTRIES
            offset = symbol_hash * MARKET_DATA_ENTRY_SIZE

            entry = MarketDataEntry(
                symbol_hash=symbol_hash,
                price=price,
                bid=bid,
                ask=ask,
                timestamp_ns=timestamp_ns,
                volume=volume,
                latency_us=0,  # Will be set by C++ component
            )

            # Write to shared memory
            self._mmap.seek(offset)
            self._mmap.write(entry)

            return True
        except Exception as e:
            logger.error(f"Failed to write tick to shared memory: {e}")
            return False

    def read_tick(self, symbol: str) -> Optional[dict]:
        """Read tick data from shared memory (zero-copy).

        Args:
            symbol: Trading symbol

        Returns:
            Tick data dict or None if not found
        """
        if not self._initialized or not self._mmap:
            return None

        try:
            symbol_hash = hash(symbol) % MAX_MARKET_DATA_ENTRIES
            offset = symbol_hash * MARKET_DATA_ENTRY_SIZE

            self._mmap.seek(offset)
            data = self._mmap.read(MARKET_DATA_ENTRY_SIZE)

            if len(data) < MARKET_DATA_ENTRY_SIZE:
                return None

            entry = MarketDataEntry.from_buffer_copy(data)

            # Validate timestamp (should be recent, within last second)
            import time

            current_ns = time.time_ns()
            age_ns = current_ns - entry.timestamp_ns
            if age_ns > 1_000_000_000:  # Older than 1 second
                return None

            return {
                "price": entry.price,
                "bid": entry.bid,
                "ask": entry.ask,
                "volume": entry.volume,
                "timestamp_ns": entry.timestamp_ns,
                "latency_us": entry.latency_us,
            }
        except Exception as e:
            logger.error(f"Failed to read tick from shared memory: {e}")
            return None

    def close(self) -> None:
        """Close shared memory region."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        self._initialized = False


class SharedMemoryRiskState:
    """Zero-copy shared memory interface for risk state."""

    def __init__(self, shm_name: str = RISK_STATE_SHM_NAME) -> None:
        """Initialize shared memory for risk state.

        Args:
            shm_name: Name of shared memory region
        """
        self.shm_name = shm_name
        self.shm_size = RISK_STATE_SIZE
        self._mmap: Optional[mmap.mmap] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize shared memory region.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(f"/dev/shm{self.shm_name}"):
                fd = os.open(f"/tmp{self.shm_name}", os.O_CREAT | os.O_RDWR, 0o666)
                os.ftruncate(fd, self.shm_size)
                self._mmap = mmap.mmap(fd, self.shm_size, access=mmap.ACCESS_WRITE)
                os.close(fd)
            else:
                fd = os.open(f"/dev/shm{self.shm_name}", os.O_RDWR)
                self._mmap = mmap.mmap(fd, self.shm_size, access=mmap.ACCESS_WRITE)
                os.close(fd)

            self._initialized = True
            logger.info(f"Risk state shared memory initialized: {self.shm_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize risk state shared memory: {e}")
            return False

    def read_risk_state(self) -> Optional[dict]:
        """Read risk state from shared memory.

        Returns:
            Risk state dict or None if error
        """
        if not self._initialized or not self._mmap:
            return None

        try:
            self._mmap.seek(0)
            data = self._mmap.read(RISK_STATE_SIZE)

            if len(data) < RISK_STATE_SIZE:
                return None

            state = RiskState.from_buffer_copy(data)

            return {
                "circuit_breaker_active": bool(state.circuit_breaker_active),
                "total_equity": state.total_equity,
                "max_drawdown_pct": state.max_drawdown_pct,
                "current_drawdown_pct": state.current_drawdown_pct,
                "latency_p99_us": state.latency_p99_us,
                "order_rate_limit": state.order_rate_limit,
                "current_order_rate": state.current_order_rate,
                "last_update_ns": state.last_update_ns,
            }
        except Exception as e:
            logger.error(f"Failed to read risk state: {e}")
            return None

    def write_risk_state(
        self,
        circuit_breaker_active: bool,
        total_equity: float,
        max_drawdown_pct: float,
        current_drawdown_pct: float,
        latency_p99_us: int,
        order_rate_limit: int,
        current_order_rate: int,
    ) -> bool:
        """Write risk state to shared memory.

        Args:
            circuit_breaker_active: Whether circuit breaker is active
            total_equity: Total equity
            max_drawdown_pct: Max drawdown percentage
            current_drawdown_pct: Current drawdown percentage
            latency_p99_us: P99 latency in microseconds
            order_rate_limit: Max orders per second
            current_order_rate: Current order rate

        Returns:
            True if successful
        """
        if not self._initialized or not self._mmap:
            return False

        try:
            import time

            state = RiskState(
                circuit_breaker_active=1 if circuit_breaker_active else 0,
                total_equity=total_equity,
                max_drawdown_pct=max_drawdown_pct,
                current_drawdown_pct=current_drawdown_pct,
                latency_p99_us=latency_p99_us,
                order_rate_limit=order_rate_limit,
                current_order_rate=current_order_rate,
                last_update_ns=time.time_ns(),
            )

            self._mmap.seek(0)
            self._mmap.write(state)

            return True
        except Exception as e:
            logger.error(f"Failed to write risk state: {e}")
            return False

    def close(self) -> None:
        """Close shared memory region."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        self._initialized = False
