"""
Phase 16: Zero-Copy Shared Memory Bridge
Python side reads raw bytes from Memory-Mapped Files (mmap) to avoid serialization overhead.
"""

import mmap
import struct
from collections.abc import Generator
from ctypes import Structure, c_char, c_double, c_int64, c_uint8, c_uint32
from typing import NamedTuple

try:
    import posix_ipc
    POSIX_IPC_AVAILABLE = True
except ImportError:
    POSIX_IPC_AVAILABLE = False
    posix_ipc = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from hean.logging import get_logger

logger = get_logger(__name__)


class TickData(NamedTuple):
    """Tick data structure matching C++ TickData."""
    symbol: str
    price: float
    bid: float
    ask: float
    timestamp_ns: int
    sequence_id: int
    valid: bool


class SharedMemoryBridge:
    """
    Zero-copy shared memory bridge for reading market data from C++ Feed Handler.

    Uses posix_ipc (or fallback to mmap) to read raw bytes from shared memory
    without any serialization overhead. Data is read directly from memory-mapped
    ring buffer.
    """

    # Structure matching C++ TickData (must match exactly)
    class TickDataStruct(Structure):
        _fields_ = [
            ("symbol", c_char * 16),
            ("price", c_double),
            ("bid", c_double),
            ("ask", c_double),
            ("timestamp_ns", c_int64),
            ("sequence_id", c_uint32),
            ("valid", c_uint8),
            ("padding", c_char * 7),
        ]

    RING_SIZE = 1024
    TICK_SIZE = struct.calcsize("16s d d d q I B 7x")  # Size of TickData in bytes

    def __init__(self, shm_name: str = "hean_feed_ring", mutex_name: str = "hean_feed_mutex"):
        """Initialize shared memory bridge.

        Args:
            shm_name: Shared memory object name (must match C++ side)
            mutex_name: Named mutex name (must match C++ side)
        """
        self._shm_name = shm_name
        self._mutex_name = mutex_name
        self._shm: posix_ipc.SharedMemory | None = None
        self._mmap: mmap.mmap | None = None
        self._mutex: posix_ipc.Semaphore | None = None
        self._read_index = 0
        self._last_sequence_id = 0
        self._initialized = False

    def connect(self) -> bool:
        """Connect to shared memory created by C++ Feed Handler.

        Returns:
            True if connection successful, False otherwise
        """
        if not POSIX_IPC_AVAILABLE:
            logger.error(
                "posix_ipc not available. Install with: pip install posix_ipc\n"
                "On macOS: brew install posix_ipc (or use conda)"
            )
            return False

        try:
            # Open existing shared memory (created by C++ side)
            self._shm = posix_ipc.SharedMemory(self._shm_name, flags=posix_ipc.O_CREAT, size=0)  # type: ignore

            # Map shared memory
            self._mmap = mmap.mmap(self._shm.fd, self._shm.size, access=mmap.ACCESS_READ)  # type: ignore

            # Open mutex (for coordination, though we're mostly lock-free)
            try:
                self._mutex = posix_ipc.Semaphore(self._mutex_name, flags=posix_ipc.O_CREAT, initial_value=1)  # type: ignore
            except posix_ipc.ExistentialError:  # type: ignore
                # Mutex might not exist yet, that's okay for read-only access
                logger.warning(f"Mutex {self._mutex_name} not found, continuing without lock")

            self._initialized = True
            logger.info(f"Connected to shared memory: {self._shm_name} ({self._shm.size} bytes)")  # type: ignore
            return True

        except posix_ipc.ExistentialError:  # type: ignore
            logger.error(f"Shared memory {self._shm_name} does not exist. Is C++ Feed Handler running?")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to shared memory: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from shared memory."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None

        if self._shm:
            if hasattr(self._shm, 'close_fd'):
                self._shm.close_fd()
            self._shm = None

        if self._mutex:
            try:
                if hasattr(self._mutex, 'close'):
                    self._mutex.close()
            except Exception:
                pass
            self._mutex = None

        self._initialized = False
        logger.info("Disconnected from shared memory")

    def _read_atomic_uint64(self, offset: int) -> int:
        """Read atomic uint64 from shared memory (read_index/write_index)."""
        if not self._mmap:
            return 0

        # Read 8 bytes (uint64_t) at offset
        self._mmap.seek(offset)
        data = self._mmap.read(8)
        if len(data) == 8:
            return struct.unpack('Q', data)[0]  # 'Q' = unsigned long long (uint64)
        return 0

    def _read_tick_at_index(self, index: int) -> TickData | None:
        """Read tick data from ring buffer at given index.

        Args:
            index: Index in ring buffer (0 to RING_SIZE-1)

        Returns:
            TickData if valid, None otherwise
        """
        if not self._mmap:
            return None

        # Calculate offset: skip metadata (3 * uint64 + uint32 = 28 bytes), then tick index
        metadata_size = 28  # write_index, read_index, dropped_ticks (3 * 8 + 4 = 28)
        offset = metadata_size + (index % self.RING_SIZE) * self.TICK_SIZE

        try:
            self._mmap.seek(offset)
            data = self._mmap.read(self.TICK_SIZE)

            if len(data) != self.TICK_SIZE:
                return None

            # Unpack using struct (faster than ctypes for read-only access)
            # Format: "16s d d d q I B 7x" (symbol, price, bid, ask, timestamp_ns, sequence_id, valid, padding)
            unpacked = struct.unpack("16s d d d q I B 7x", data)

            symbol_bytes, price, bid, ask, timestamp_ns, sequence_id, valid = unpacked

            # Decode symbol (remove null terminator)
            symbol = symbol_bytes.split(b'\x00')[0].decode('utf-8', errors='ignore')

            if valid == 0:
                return None

            return TickData(
                symbol=symbol,
                price=price,
                bid=bid,
                ask=ask,
                timestamp_ns=timestamp_ns,
                sequence_id=sequence_id,
                valid=bool(valid)
            )

        except Exception as e:
            logger.debug(f"Failed to read tick at index {index}: {e}")
            return None

    def read_ticks(self, max_ticks: int = 100) -> Generator[TickData, None, None]:
        """
        Read new ticks from shared memory ring buffer (zero-copy).

        Yields:
            TickData: New tick data
        """
        if not self._initialized or not self._mmap:
            return

        # Read current write_index from shared memory
        write_index_offset = 0  # First 8 bytes are write_index
        write_index = self._read_atomic_uint64(write_index_offset)

        # Read dropped ticks counter (for monitoring)
        dropped_offset = 16  # After write_index (8) and read_index (8)
        self._mmap.seek(dropped_offset)
        dropped_data = self._mmap.read(4)
        dropped_ticks = struct.unpack('I', dropped_data)[0] if len(dropped_data) == 4 else 0

        if dropped_ticks > 0 and self._last_sequence_id > 0:
            logger.warning(f"Dropped ticks detected: {dropped_ticks}")

        # Read ticks from last read position to current write position
        ticks_read = 0
        while self._read_index < write_index and ticks_read < max_ticks:
            tick = self._read_tick_at_index(self._read_index)
            if tick:
                # Check for sequence gaps (missed ticks)
                if self._last_sequence_id > 0 and tick.sequence_id != self._last_sequence_id + 1:
                    gap = tick.sequence_id - self._last_sequence_id - 1
                    if gap > 0:
                        logger.warning(f"Sequence gap detected: missed {gap} ticks (expected {self._last_sequence_id + 1}, got {tick.sequence_id})")

                self._last_sequence_id = tick.sequence_id
                yield tick
                ticks_read += 1

            self._read_index += 1

        # Update read_index in shared memory (so C++ side knows we've consumed)
        # This is done atomically, but we use the mutex for safety
        if self._mutex and POSIX_IPC_AVAILABLE:
            try:
                self._mutex.acquire(timeout=0.001)  # Non-blocking, 1ms timeout  # type: ignore
                try:
                    read_index_offset = 8  # After write_index (8 bytes)
                    self._mmap.seek(read_index_offset)
                    self._mmap.write(struct.pack('Q', self._read_index))
                finally:
                    self._mutex.release()  # type: ignore
            except posix_ipc.BusyError:  # type: ignore
                # Mutex busy, skip update this iteration (will update next time)
                pass
            except Exception as e:
                logger.debug(f"Failed to update read_index: {e}")

    def get_stats(self) -> dict:
        """Get statistics about shared memory bridge.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized or not self._mmap:
            return {}

        write_index = self._read_atomic_uint64(0)

        self._mmap.seek(16)  # dropped_ticks offset
        dropped_data = self._mmap.read(4)
        dropped_ticks = struct.unpack('I', dropped_data)[0] if len(dropped_data) == 4 else 0

        return {
            "write_index": write_index,
            "read_index": self._read_index,
            "pending_ticks": max(0, write_index - self._read_index),
            "dropped_ticks": dropped_ticks,
            "last_sequence_id": self._last_sequence_id,
            "buffer_utilization": min(100.0, ((write_index - self._read_index) / self.RING_SIZE) * 100.0),
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
