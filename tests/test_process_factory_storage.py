"""Tests for Process Factory storage."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from hean.process_factory.schemas import (
    BybitEnvironmentSnapshot,
    DailyCapitalPlan,
    ProcessPortfolioEntry,
    ProcessPortfolioState,
    ProcessRun,
    ProcessRunStatus,
)
from hean.process_factory.storage import SQLiteStorage


@pytest.mark.asyncio
async def test_storage_snapshot():
    """Test saving and loading snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(Path(tmpdir) / "test.db")
        snapshot = BybitEnvironmentSnapshot(
            timestamp=datetime.now(),
            balances={"USDT": 1000.0},
            positions=[],
            open_orders=[],
            funding_rates={"BTCUSDT": 0.0001},
        )
        await storage.save_snapshot(snapshot)
        loaded = await storage.load_latest_snapshot()
        assert loaded is not None
        assert loaded.balances["USDT"] == 1000.0
        assert loaded.funding_rates["BTCUSDT"] == 0.0001
        await storage.close()


@pytest.mark.asyncio
async def test_storage_run():
    """Test saving and loading runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(Path(tmpdir) / "test.db")
        run = ProcessRun(
            run_id="test_run",
            process_id="test_process",
            started_at=datetime.now(),
            status=ProcessRunStatus.COMPLETED,
            metrics={"capital_delta": 10.0},
            capital_allocated_usd=100.0,
        )
        await storage.save_run(run)
        runs = await storage.list_runs(process_id="test_process")
        assert len(runs) == 1
        assert runs[0].run_id == "test_run"
        assert runs[0].metrics["capital_delta"] == 10.0
        await storage.close()


@pytest.mark.asyncio
async def test_storage_portfolio():
    """Test saving and loading portfolio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(Path(tmpdir) / "test.db")
        entries = [
            ProcessPortfolioEntry(
                process_id="test_process",
                state=ProcessPortfolioState.TESTING,
                weight=0.1,
                runs_count=5,
                pnl_sum=50.0,
            )
        ]
        await storage.save_portfolio(entries)
        loaded = await storage.load_portfolio()
        assert len(loaded) == 1
        assert loaded[0].process_id == "test_process"
        assert loaded[0].pnl_sum == 50.0
        await storage.close()


@pytest.mark.asyncio
async def test_storage_capital_plan():
    """Test saving and loading capital plans."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = SQLiteStorage(Path(tmpdir) / "test.db")
        plan = DailyCapitalPlan(
            date=datetime.now(),
            reserve_usd=400.0,
            active_usd=500.0,
            experimental_usd=100.0,
            allocations={"process1": 100.0},
            total_capital_usd=1000.0,
        )
        await storage.save_capital_plan(plan)
        loaded = await storage.load_latest_capital_plan()
        assert loaded is not None
        assert loaded.reserve_usd == 400.0
        assert loaded.allocations["process1"] == 100.0
        await storage.close()

