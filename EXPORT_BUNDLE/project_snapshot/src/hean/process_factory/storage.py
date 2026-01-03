"""Storage interface and SQLite implementation for Process Factory."""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from hean.process_factory.schemas import (
    BybitEnvironmentSnapshot,
    DailyCapitalPlan,
    ProcessPortfolioEntry,
    ProcessRun,
)

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore


class Storage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    async def save_snapshot(self, snapshot: BybitEnvironmentSnapshot) -> None:
        """Save environment snapshot."""

    @abstractmethod
    async def load_latest_snapshot(self) -> BybitEnvironmentSnapshot | None:
        """Load latest environment snapshot."""

    @abstractmethod
    async def save_run(self, run: ProcessRun) -> None:
        """Save process run."""

    @abstractmethod
    async def list_runs(
        self,
        process_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProcessRun]:
        """List process runs."""

    @abstractmethod
    async def save_portfolio(self, entries: list[ProcessPortfolioEntry]) -> None:
        """Save process portfolio."""

    @abstractmethod
    async def load_portfolio(self) -> list[ProcessPortfolioEntry]:
        """Load process portfolio."""

    @abstractmethod
    async def save_capital_plan(self, plan: DailyCapitalPlan) -> None:
        """Save daily capital plan."""

    @abstractmethod
    async def load_latest_capital_plan(self) -> DailyCapitalPlan | None:
        """Load latest capital plan."""

    @abstractmethod
    async def close(self) -> None:
        """Close storage connection."""


class SQLiteStorage(Storage):
    """SQLite implementation of storage."""

    def __init__(self, db_path: str | Path = "process_factory.db") -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._use_async = aiosqlite is not None

    async def _get_connection(self) -> sqlite3.Connection | Any:
        """Get database connection (sync or async)."""
        if self._use_async:
            if self._conn is None:
                self._conn = await aiosqlite.connect(str(self.db_path))  # type: ignore
                await self._init_schema(self._conn)  # type: ignore
            return self._conn
        else:
            if self._conn is None:
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._init_schema_sync(self._conn)
            return self._conn

    def _init_schema_sync(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema (sync version)."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                process_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                metrics TEXT,
                logs_ref TEXT,
                inputs TEXT,
                outputs TEXT,
                error TEXT,
                capital_allocated_usd REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                process_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0,
                runs_count INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                pnl_sum REAL NOT NULL DEFAULT 0,
                max_dd REAL NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                fail_rate REAL NOT NULL DEFAULT 0,
                time_efficiency REAL NOT NULL DEFAULT 0,
                last_run_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS capital_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_runs_process_id ON runs(process_id);
            CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp);
        """
        )
        conn.commit()

    async def _init_schema(self, conn: Any) -> None:
        """Initialize database schema (async version)."""
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                process_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                metrics TEXT,
                logs_ref TEXT,
                inputs TEXT,
                outputs TEXT,
                error TEXT,
                capital_allocated_usd REAL NOT NULL DEFAULT 0,
                daily_run_key TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS daily_run_keys (
                daily_run_key TEXT PRIMARY KEY,
                process_id TEXT NOT NULL,
                date TEXT NOT NULL,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                process_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0,
                runs_count INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                pnl_sum REAL NOT NULL DEFAULT 0,
                max_dd REAL NOT NULL DEFAULT 0,
                avg_roi REAL NOT NULL DEFAULT 0,
                fail_rate REAL NOT NULL DEFAULT 0,
                time_efficiency REAL NOT NULL DEFAULT 0,
                last_run_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS capital_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_runs_process_id ON runs(process_id);
            CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp);
        """
        )
        await conn.commit()

    async def save_snapshot(self, snapshot: BybitEnvironmentSnapshot) -> None:
        """Save environment snapshot."""
        conn = await self._get_connection()
        data_json = json.dumps(snapshot.model_dump(mode="json"), default=str)
        if self._use_async:
            await conn.execute(  # type: ignore
                "INSERT INTO snapshots (timestamp, data) VALUES (?, ?)",
                (snapshot.timestamp.isoformat(), data_json),
            )
            await conn.commit()  # type: ignore
        else:
            conn.execute(
                "INSERT INTO snapshots (timestamp, data) VALUES (?, ?)",
                (snapshot.timestamp.isoformat(), data_json),
            )
            conn.commit()

    async def load_latest_snapshot(self) -> BybitEnvironmentSnapshot | None:
        """Load latest environment snapshot."""
        conn = await self._get_connection()
        if self._use_async:
            cursor = await conn.execute(  # type: ignore
                "SELECT data FROM snapshots ORDER BY timestamp DESC LIMIT 1"
            )
            row = await cursor.fetchone()  # type: ignore
        else:
            cursor = conn.execute(
                "SELECT data FROM snapshots ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return BybitEnvironmentSnapshot(**data)

    async def save_run(self, run: ProcessRun, daily_run_key: str | None = None) -> None:
        """Save process run.

        Args:
            run: Process run to save
            daily_run_key: Optional daily run key for idempotency
        """
        conn = await self._get_connection()
        data = {
            "run_id": run.run_id,
            "process_id": run.process_id,
            "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "status": run.status.value,
            "metrics": json.dumps(run.metrics, default=str),
            "logs_ref": run.logs_ref,
            "inputs": json.dumps(run.inputs, default=str),
            "outputs": json.dumps(run.outputs, default=str),
            "error": run.error,
            "capital_allocated_usd": run.capital_allocated_usd,
            "daily_run_key": daily_run_key,
        }
        if self._use_async:
            await conn.execute(  # type: ignore
                """INSERT OR REPLACE INTO runs 
                   (run_id, process_id, started_at, finished_at, status, metrics, logs_ref, 
                    inputs, outputs, error, capital_allocated_usd, daily_run_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["run_id"],
                    data["process_id"],
                    data["started_at"],
                    data["finished_at"],
                    data["status"],
                    data["metrics"],
                    data["logs_ref"],
                    data["inputs"],
                    data["outputs"],
                    data["error"],
                    data["capital_allocated_usd"],
                    data["daily_run_key"],
                )
            )
            # Save daily run key if provided
            if daily_run_key:
                date_str = run.started_at.date().isoformat()
                await conn.execute(  # type: ignore
                    """INSERT OR REPLACE INTO daily_run_keys 
                       (daily_run_key, process_id, date, run_id)
                       VALUES (?, ?, ?, ?)""",
                    (daily_run_key, run.process_id, date_str, run.run_id),
                )
            await conn.commit()  # type: ignore
        else:
            conn.execute(
                """INSERT OR REPLACE INTO runs 
                   (run_id, process_id, started_at, finished_at, status, metrics, logs_ref, 
                    inputs, outputs, error, capital_allocated_usd, daily_run_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["run_id"],
                    data["process_id"],
                    data["started_at"],
                    data["finished_at"],
                    data["status"],
                    data["metrics"],
                    data["logs_ref"],
                    data["inputs"],
                    data["outputs"],
                    data["error"],
                    data["capital_allocated_usd"],
                    data["daily_run_key"],
                ),
            )
            # Save daily run key if provided
            if daily_run_key:
                date_str = run.started_at.date().isoformat()
                conn.execute(
                    """INSERT OR REPLACE INTO daily_run_keys 
                       (daily_run_key, process_id, date, run_id)
                       VALUES (?, ?, ?, ?)""",
                    (daily_run_key, run.process_id, date_str, run.run_id),
                )
            conn.commit()

    async def check_daily_run_key(
        self, daily_run_key: str
    ) -> tuple[bool, str | None]:
        """Check if a daily run key already exists.

        Args:
            daily_run_key: Daily run key to check

        Returns:
            Tuple of (exists, run_id if exists)
        """
        conn = await self._get_connection()
        if self._use_async:
            cursor = await conn.execute(  # type: ignore
                "SELECT run_id FROM daily_run_keys WHERE daily_run_key = ?",
                (daily_run_key,),
            )
            row = await cursor.fetchone()  # type: ignore
        else:
            cursor = conn.execute(
                "SELECT run_id FROM daily_run_keys WHERE daily_run_key = ?",
                (daily_run_key,),
            )
            row = cursor.fetchone()
        if row:
            return True, row[0]
        return False, None

    async def list_runs(
        self,
        process_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProcessRun]:
        """List process runs."""
        conn = await self._get_connection()
        if process_id:
            if self._use_async:
                cursor = await conn.execute(  # type: ignore
                    "SELECT * FROM runs WHERE process_id = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (process_id, limit, offset),
                )
                rows = await cursor.fetchall()  # type: ignore
            else:
                cursor = conn.execute(
                    "SELECT * FROM runs WHERE process_id = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (process_id, limit, offset),
                )
                rows = cursor.fetchall()
        else:
            if self._use_async:
                cursor = await conn.execute(  # type: ignore
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
                rows = await cursor.fetchall()  # type: ignore
            else:
                cursor = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
                rows = cursor.fetchall()

        runs = []
        for row in rows:
            row_dict = dict(row) if self._use_async else dict(row)
            run_dict = {
                "run_id": row_dict["run_id"],
                "process_id": row_dict["process_id"],
                "started_at": datetime.fromisoformat(row_dict["started_at"]),
                "finished_at": (
                    datetime.fromisoformat(row_dict["finished_at"])
                    if row_dict["finished_at"]
                    else None
                ),
                "status": row_dict["status"],
                "metrics": json.loads(row_dict["metrics"] or "{}"),
                "logs_ref": row_dict["logs_ref"],
                "inputs": json.loads(row_dict["inputs"] or "{}"),
                "outputs": json.loads(row_dict["outputs"] or "{}"),
                "error": row_dict["error"],
                "capital_allocated_usd": row_dict["capital_allocated_usd"],
            }
            # Create ProcessRun with proper enum
            from hean.process_factory.schemas import ProcessRunStatus

            run_dict["status"] = ProcessRunStatus(run_dict["status"])
            runs.append(ProcessRun(**run_dict))
        return runs

    async def save_portfolio(self, entries: list[ProcessPortfolioEntry]) -> None:
        """Save process portfolio."""
        conn = await self._get_connection()
        for entry in entries:
            data = {
                "process_id": entry.process_id,
                "state": entry.state.value,
                "weight": entry.weight,
                "runs_count": entry.runs_count,
                "wins": entry.wins,
                "losses": entry.losses,
                "pnl_sum": entry.pnl_sum,
                "max_dd": entry.max_dd,
                "avg_roi": entry.avg_roi,
                "fail_rate": entry.fail_rate,
                "time_efficiency": entry.time_efficiency,
                "last_run_at": entry.last_run_at.isoformat() if entry.last_run_at else None,
            }
            if self._use_async:
                await conn.execute(  # type: ignore
                    """INSERT OR REPLACE INTO portfolio 
                       (process_id, state, weight, runs_count, wins, losses, pnl_sum, max_dd,
                        avg_roi, fail_rate, time_efficiency, last_run_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        data["process_id"],
                        data["state"],
                        data["weight"],
                        data["runs_count"],
                        data["wins"],
                        data["losses"],
                        data["pnl_sum"],
                        data["max_dd"],
                        data["avg_roi"],
                        data["fail_rate"],
                        data["time_efficiency"],
                        data["last_run_at"],
                    ),
                )
                await conn.commit()  # type: ignore
            else:
                conn.execute(
                    """INSERT OR REPLACE INTO portfolio 
                       (process_id, state, weight, runs_count, wins, losses, pnl_sum, max_dd,
                        avg_roi, fail_rate, time_efficiency, last_run_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        data["process_id"],
                        data["state"],
                        data["weight"],
                        data["runs_count"],
                        data["wins"],
                        data["losses"],
                        data["pnl_sum"],
                        data["max_dd"],
                        data["avg_roi"],
                        data["fail_rate"],
                        data["time_efficiency"],
                        data["last_run_at"],
                    ),
                )
                conn.commit()

    async def load_portfolio(self) -> list[ProcessPortfolioEntry]:
        """Load process portfolio."""
        conn = await self._get_connection()
        if self._use_async:
            cursor = await conn.execute("SELECT * FROM portfolio")  # type: ignore
            rows = await cursor.fetchall()  # type: ignore
        else:
            cursor = conn.execute("SELECT * FROM portfolio")
            rows = cursor.fetchall()

        entries = []
        for row in rows:
            row_dict = dict(row) if self._use_async else dict(row)
            from hean.process_factory.schemas import ProcessPortfolioState

            entry_dict = {
                "process_id": row_dict["process_id"],
                "state": ProcessPortfolioState(row_dict["state"]),
                "weight": row_dict["weight"],
                "runs_count": row_dict["runs_count"],
                "wins": row_dict["wins"],
                "losses": row_dict["losses"],
                "pnl_sum": row_dict["pnl_sum"],
                "max_dd": row_dict["max_dd"],
                "avg_roi": row_dict["avg_roi"],
                "fail_rate": row_dict["fail_rate"],
                "time_efficiency": row_dict["time_efficiency"],
                "last_run_at": (
                    datetime.fromisoformat(row_dict["last_run_at"])
                    if row_dict["last_run_at"]
                    else None
                ),
            }
            entries.append(ProcessPortfolioEntry(**entry_dict))
        return entries

    async def save_capital_plan(self, plan: DailyCapitalPlan) -> None:
        """Save daily capital plan."""
        conn = await self._get_connection()
        data_json = json.dumps(plan.model_dump(mode="json"), default=str)
        if self._use_async:
            await conn.execute(  # type: ignore
                "INSERT INTO capital_plans (date, data) VALUES (?, ?)",
                (plan.date.isoformat(), data_json),
            )
            await conn.commit()  # type: ignore
        else:
            conn.execute(
                "INSERT INTO capital_plans (date, data) VALUES (?, ?)",
                (plan.date.isoformat(), data_json),
            )
            conn.commit()

    async def load_latest_capital_plan(self) -> DailyCapitalPlan | None:
        """Load latest capital plan."""
        conn = await self._get_connection()
        if self._use_async:
            cursor = await conn.execute(  # type: ignore
                "SELECT data FROM capital_plans ORDER BY date DESC LIMIT 1"
            )
            row = await cursor.fetchone()  # type: ignore
        else:
            cursor = conn.execute(
                "SELECT data FROM capital_plans ORDER BY date DESC LIMIT 1"
            )
            row = cursor.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return DailyCapitalPlan(**data)

    async def close(self) -> None:
        """Close storage connection."""
        if self._conn:
            if self._use_async:
                await self._conn.close()  # type: ignore
            else:
                self._conn.close()
            self._conn = None

