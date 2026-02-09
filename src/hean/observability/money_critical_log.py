"""Money-Critical Log - Append-only audit trail for all money-affecting events.

This module provides an immutable audit log for all events that could affect money:
- SIGNAL: Trading signals that could lead to orders
- ORDER_REQUEST: Order placement attempts
- ORDER_FILLED: Order execution confirmations
- ORDER_CANCELLED: Order cancellations
- ORDER_REJECTED: Order rejections
- POSITION_OPENED: New position opened
- POSITION_CLOSED: Position closed (realized PnL)
- PNL_UPDATE: PnL changes
- RISK_ALERT: Risk limit violations

The log is:
- Append-only: Cannot be modified after writing
- Persistent: Survives restarts
- Structured: JSON-lines format for easy parsing
- Time-ordered: Events are logged with nanosecond timestamps
- Correlation-aware: Tracks event chains via correlation IDs
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


# Events that affect money and must be logged
MONEY_CRITICAL_EVENTS = frozenset({
    EventType.SIGNAL,
    EventType.ORDER_REQUEST,
    EventType.ORDER_FILLED,
    EventType.ORDER_CANCELLED,
    EventType.ORDER_REJECTED,
    EventType.POSITION_OPENED,
    EventType.POSITION_CLOSED,
    EventType.PNL_UPDATE,
    EventType.RISK_ALERT,
    EventType.FUNDING_UPDATE,
})


class LogEntryType(str, Enum):
    """Type of log entry."""

    EVENT = "event"  # Regular event
    CHECKPOINT = "checkpoint"  # System checkpoint
    RECOVERY = "recovery"  # System recovered from failure


@dataclass
class LogEntry:
    """A single entry in the money-critical log."""

    # Required fields
    sequence_number: int
    timestamp_ns: int
    entry_type: LogEntryType
    event_type: str

    # Event data
    data: dict[str, Any]

    # Tracing
    correlation_id: str | None = None
    causation_id: str | None = None  # ID of the event that caused this one

    # Metadata
    symbol: str | None = None
    strategy_id: str | None = None
    order_id: str | None = None

    # Integrity
    prev_hash: str | None = None
    entry_hash: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Compute entry hash after initialization."""
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of entry for integrity verification."""
        content = json.dumps({
            "sequence_number": self.sequence_number,
            "timestamp_ns": self.timestamp_ns,
            "entry_type": self.entry_type.value if isinstance(self.entry_type, LogEntryType) else self.entry_type,
            "event_type": self.event_type,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "seq": self.sequence_number,
            "ts": self.timestamp_ns,
            "type": self.entry_type.value if isinstance(self.entry_type, LogEntryType) else self.entry_type,
            "event": self.event_type,
            "data": self.data,
            "corr_id": self.correlation_id,
            "cause_id": self.causation_id,
            "symbol": self.symbol,
            "strategy": self.strategy_id,
            "order_id": self.order_id,
            "prev_hash": self.prev_hash,
            "hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        entry = cls(
            sequence_number=d["seq"],
            timestamp_ns=d["ts"],
            entry_type=LogEntryType(d["type"]),
            event_type=d["event"],
            data=d["data"],
            correlation_id=d.get("corr_id"),
            causation_id=d.get("cause_id"),
            symbol=d.get("symbol"),
            strategy_id=d.get("strategy"),
            order_id=d.get("order_id"),
            prev_hash=d.get("prev_hash"),
        )
        # Verify hash
        if entry.entry_hash != d.get("hash"):
            logger.warning(
                f"Hash mismatch for entry {entry.sequence_number}: "
                f"computed={entry.entry_hash}, stored={d.get('hash')}"
            )
        return entry

    def to_json(self) -> str:
        """Convert to JSON line."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


@dataclass
class EventChain:
    """A chain of related events for debugging."""

    correlation_id: str
    entries: list[LogEntry]

    @property
    def start_time(self) -> datetime:
        """Get start time of chain."""
        if self.entries:
            return datetime.utcfromtimestamp(self.entries[0].timestamp_ns / 1e9)
        return datetime.utcnow()

    @property
    def end_time(self) -> datetime:
        """Get end time of chain."""
        if self.entries:
            return datetime.utcfromtimestamp(self.entries[-1].timestamp_ns / 1e9)
        return datetime.utcnow()

    @property
    def duration_ms(self) -> float:
        """Get duration of chain in milliseconds."""
        if len(self.entries) >= 2:
            return (self.entries[-1].timestamp_ns - self.entries[0].timestamp_ns) / 1e6
        return 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get chain summary."""
        return {
            "correlation_id": self.correlation_id,
            "entry_count": len(self.entries),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "event_types": [e.event_type for e in self.entries],
        }


class MoneyCriticalLog:
    """Append-only log for money-critical events.

    Features:
    - Append-only: Cannot modify or delete entries
    - Hash chain: Each entry contains hash of previous for integrity
    - Correlation tracking: Links related events together
    - Persistent: Writes to JSON-lines file
    - In-memory index: Fast lookup by correlation ID
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        max_in_memory: int = 10000,
        flush_interval: int = 100,
    ) -> None:
        """Initialize money-critical log.

        Args:
            log_dir: Directory for log files (None for in-memory only)
            max_in_memory: Maximum entries to keep in memory
            flush_interval: Entries between file flushes
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.max_in_memory = max_in_memory
        self.flush_interval = flush_interval

        # In-memory buffer
        self._entries: deque[LogEntry] = deque(maxlen=max_in_memory)

        # Correlation index: correlation_id -> list of sequence numbers
        self._correlation_index: dict[str, list[int]] = {}

        # State
        self._sequence_number = 0
        self._last_hash: str | None = None
        self._entries_since_flush = 0

        # File handle for persistent logging
        self._file: Any = None
        self._current_file_path: Path | None = None

        # Statistics
        self._stats = {
            "total_entries": 0,
            "events_by_type": {},
            "chains_started": 0,
            "integrity_violations": 0,
        }

        # Initialize log file if directory provided
        if self.log_dir:
            self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize log file for current session."""
        if not self.log_dir:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create date-based log file
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        self._current_file_path = self.log_dir / f"money_critical_{date_str}.jsonl"

        # Load existing entries if file exists
        if self._current_file_path.exists():
            self._load_existing_entries()

        # Open file for appending
        self._file = open(self._current_file_path, "a", encoding="utf-8")
        logger.info(f"Money-critical log initialized: {self._current_file_path}")

    def _load_existing_entries(self) -> None:
        """Load existing entries from file to restore state."""
        if not self._current_file_path or not self._current_file_path.exists():
            return

        try:
            with open(self._current_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = LogEntry.from_dict(data)
                        self._entries.append(entry)
                        self._sequence_number = max(self._sequence_number, entry.sequence_number)
                        self._last_hash = entry.entry_hash

                        # Update correlation index
                        if entry.correlation_id:
                            if entry.correlation_id not in self._correlation_index:
                                self._correlation_index[entry.correlation_id] = []
                            self._correlation_index[entry.correlation_id].append(entry.sequence_number)

                        self._stats["total_entries"] += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse log entry: {e}")

            logger.info(
                f"Loaded {len(self._entries)} existing entries, "
                f"last sequence: {self._sequence_number}"
            )
        except Exception as e:
            logger.error(f"Failed to load existing log: {e}")

    def log_event(
        self,
        event: Event,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> LogEntry | None:
        """Log a money-critical event.

        Args:
            event: The event to log
            correlation_id: ID linking related events
            causation_id: ID of event that caused this one

        Returns:
            LogEntry if event was logged, None if not money-critical
        """
        # Only log money-critical events
        if event.event_type not in MONEY_CRITICAL_EVENTS:
            return None

        # Extract metadata from event data
        data = event.data or {}
        symbol = data.get("symbol")
        strategy_id = data.get("strategy_id")
        order_id = data.get("order_id")

        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = data.get("correlation_id")

        # Create entry
        self._sequence_number += 1
        entry = LogEntry(
            sequence_number=self._sequence_number,
            timestamp_ns=time.time_ns(),
            entry_type=LogEntryType.EVENT,
            event_type=event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type),
            data=data,
            correlation_id=correlation_id,
            causation_id=causation_id,
            symbol=symbol,
            strategy_id=strategy_id,
            order_id=order_id,
            prev_hash=self._last_hash,
        )

        # Update state
        self._last_hash = entry.entry_hash
        self._entries.append(entry)

        # Update correlation index
        if correlation_id:
            if correlation_id not in self._correlation_index:
                self._correlation_index[correlation_id] = []
                self._stats["chains_started"] += 1
            self._correlation_index[correlation_id].append(entry.sequence_number)

        # Update stats
        self._stats["total_entries"] += 1
        event_type_str = entry.event_type
        self._stats["events_by_type"][event_type_str] = (
            self._stats["events_by_type"].get(event_type_str, 0) + 1
        )

        # Write to file
        self._write_entry(entry)

        return entry

    def log_checkpoint(self, description: str, metadata: dict[str, Any] | None = None) -> LogEntry:
        """Log a system checkpoint for debugging.

        Args:
            description: Checkpoint description
            metadata: Additional metadata

        Returns:
            LogEntry for the checkpoint
        """
        self._sequence_number += 1
        entry = LogEntry(
            sequence_number=self._sequence_number,
            timestamp_ns=time.time_ns(),
            entry_type=LogEntryType.CHECKPOINT,
            event_type="CHECKPOINT",
            data={"description": description, **(metadata or {})},
            prev_hash=self._last_hash,
        )

        self._last_hash = entry.entry_hash
        self._entries.append(entry)
        self._stats["total_entries"] += 1

        self._write_entry(entry)
        return entry

    def _write_entry(self, entry: LogEntry) -> None:
        """Write entry to log file."""
        if self._file:
            self._file.write(entry.to_json() + "\n")
            self._entries_since_flush += 1

            if self._entries_since_flush >= self.flush_interval:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._entries_since_flush = 0

    def get_chain(self, correlation_id: str) -> EventChain | None:
        """Get all events in an event chain.

        Args:
            correlation_id: The correlation ID to look up

        Returns:
            EventChain if found, None otherwise
        """
        if correlation_id not in self._correlation_index:
            return None

        seq_numbers = self._correlation_index[correlation_id]
        entries = [
            e for e in self._entries
            if e.sequence_number in seq_numbers
        ]

        return EventChain(
            correlation_id=correlation_id,
            entries=sorted(entries, key=lambda e: e.sequence_number),
        )

    def get_recent_entries(self, limit: int = 100) -> list[LogEntry]:
        """Get recent log entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent entries
        """
        return list(self._entries)[-limit:]

    def get_entries_by_type(self, event_type: str, limit: int = 100) -> list[LogEntry]:
        """Get entries filtered by event type.

        Args:
            event_type: Event type to filter by
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        return [
            e for e in list(self._entries)[-limit * 10:]  # Check more for filtering
            if e.event_type == event_type
        ][:limit]

    def get_entries_by_symbol(self, symbol: str, limit: int = 100) -> list[LogEntry]:
        """Get entries filtered by symbol.

        Args:
            symbol: Symbol to filter by
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        return [
            e for e in list(self._entries)[-limit * 10:]
            if e.symbol == symbol
        ][:limit]

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify log integrity using hash chain.

        Returns:
            (is_valid, list of violations)
        """
        violations = []
        prev_hash = None

        for entry in self._entries:
            # Check hash chain
            if entry.prev_hash != prev_hash:
                violations.append(
                    f"Hash chain broken at seq {entry.sequence_number}: "
                    f"expected prev_hash={prev_hash}, got {entry.prev_hash}"
                )

            # Verify entry hash
            computed_hash = entry._compute_hash()
            if entry.entry_hash != computed_hash:
                violations.append(
                    f"Entry hash mismatch at seq {entry.sequence_number}: "
                    f"stored={entry.entry_hash}, computed={computed_hash}"
                )

            prev_hash = entry.entry_hash

        if violations:
            self._stats["integrity_violations"] = len(violations)
            logger.error(f"Log integrity verification failed: {len(violations)} violations")

        return len(violations) == 0, violations

    def get_stats(self) -> dict[str, Any]:
        """Get log statistics.

        Returns:
            Dictionary with log statistics
        """
        return {
            **self._stats,
            "in_memory_entries": len(self._entries),
            "active_chains": len(self._correlation_index),
            "current_sequence": self._sequence_number,
            "log_file": str(self._current_file_path) if self._current_file_path else None,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get log summary for API response.

        Returns:
            Summary dictionary
        """
        recent = self.get_recent_entries(10)

        return {
            "stats": self.get_stats(),
            "recent_entries": [e.to_dict() for e in recent],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def replay_chain(self, correlation_id: str) -> Iterator[LogEntry]:
        """Replay events in a chain for debugging.

        Args:
            correlation_id: Chain to replay

        Yields:
            LogEntry in order
        """
        chain = self.get_chain(correlation_id)
        if chain:
            for entry in chain.entries:
                yield entry

    def close(self) -> None:
        """Close log file and flush remaining entries."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            logger.info("Money-critical log closed")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


# Global instance
_money_log: MoneyCriticalLog | None = None


def get_money_log(log_dir: str | Path | None = None) -> MoneyCriticalLog:
    """Get or create global money-critical log.

    Args:
        log_dir: Directory for log files

    Returns:
        MoneyCriticalLog instance
    """
    global _money_log
    if _money_log is None:
        _money_log = MoneyCriticalLog(log_dir=log_dir)
    return _money_log


def log_money_event(
    event: Event,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> LogEntry | None:
    """Log a money-critical event to the global log.

    Convenience function for logging events without managing the log instance.

    Args:
        event: Event to log
        correlation_id: Optional correlation ID
        causation_id: Optional causation ID

    Returns:
        LogEntry if logged, None if not money-critical
    """
    return get_money_log().log_event(event, correlation_id, causation_id)
