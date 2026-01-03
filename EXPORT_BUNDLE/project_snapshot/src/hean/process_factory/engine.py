"""Process Engine: orchestrator for planning/running processes."""

import uuid
from datetime import datetime
from typing import Any

from hean.logging import get_logger
from hean.process_factory.integrations.bybit_env import BybitEnvScanner
from hean.process_factory.leverage_engine import LeverageEngine
from hean.process_factory.registry import ProcessRegistry
from hean.process_factory.router import CapitalRouter
from hean.process_factory.sandbox import ProcessSandbox
from hean.process_factory.schemas import (
    BybitEnvironmentSnapshot,
    DailyCapitalPlan,
    Opportunity,
    OpportunitySource,
    ProcessDefinition,
    ProcessPortfolioEntry,
    ProcessPortfolioState,
    ProcessRun,
    ProcessRunStatus,
)
from hean.process_factory.scorer import rank_opportunities
from hean.process_factory.selector import ProcessSelector
from hean.process_factory.storage import Storage

logger = get_logger(__name__)


class ProcessEngine:
    """Orchestrator for process planning and execution."""

    def __init__(
        self,
        storage: Storage,
        registry: ProcessRegistry | None = None,
        router: CapitalRouter | None = None,
        selector: ProcessSelector | None = None,
        sandbox: ProcessSandbox | None = None,
        leverage_engine: LeverageEngine | None = None,
    ) -> None:
        """Initialize process engine.

        Args:
            storage: Storage interface
            registry: Process registry (creates default if not provided)
            router: Capital router (creates default if not provided)
            selector: Process selector (creates default if not provided)
            sandbox: Process sandbox (creates default if not provided)
            leverage_engine: Leverage engine (creates default if not provided)
        """
        self.storage = storage
        self.registry = registry or ProcessRegistry()
        self.router = router or CapitalRouter()
        self.selector = selector or ProcessSelector()
        self.sandbox = sandbox or ProcessSandbox()
        self.leverage_engine = leverage_engine or LeverageEngine()

    async def scan_environment(
        self, http_client: Any | None = None
    ) -> BybitEnvironmentSnapshot:
        """Scan environment and save snapshot.

        Args:
            http_client: Optional Bybit HTTP client

        Returns:
            Environment snapshot
        """
        scanner = BybitEnvScanner(http_client=http_client)
        snapshot = await scanner.scan()
        await self.storage.save_snapshot(snapshot)
        logger.info(f"Environment scan completed: {len(snapshot.balances)} balances, {len(snapshot.positions)} positions")
        return snapshot

    async def plan(
        self, total_capital_usd: float, snapshot: BybitEnvironmentSnapshot | None = None
    ) -> tuple[list[tuple[Opportunity, float]], DailyCapitalPlan]:
        """Plan daily capital allocation.

        Args:
            total_capital_usd: Total available capital
            snapshot: Environment snapshot (loads latest if not provided)

        Returns:
            Tuple of (ranked opportunities, capital plan)
        """
        # Load snapshot if not provided
        if snapshot is None:
            snapshot = await self.storage.load_latest_snapshot()
            if snapshot is None:
                logger.warning("No environment snapshot available, creating empty snapshot")
                snapshot = BybitEnvironmentSnapshot(timestamp=datetime.now())

        # Convert snapshot to opportunities
        opportunities = self._snapshot_to_opportunities(snapshot)

        # Rank opportunities
        ranked = rank_opportunities(opportunities)

        # Load portfolio
        portfolio = await self.storage.load_portfolio()

        # Compute capital plan
        plan = self.router.compute_daily_plan(total_capital_usd, portfolio)

        # Save plan
        await self.storage.save_capital_plan(plan)

        logger.info(f"Planning completed: {len(ranked)} opportunities, plan with {len(plan.allocations)} allocations")
        return ranked, plan

    async def run_process(
        self,
        process_id: str,
        inputs: dict[str, Any],
        mode: str = "sandbox",
        capital_allocated_usd: float = 0.0,
        force: bool = False,
    ) -> ProcessRun:
        """Run a process with structured logging and idempotency.

        Args:
            process_id: Process ID to run
            inputs: Process inputs
            mode: Execution mode ("sandbox" or "live")
            capital_allocated_usd: Capital allocated to this run
            force: Force run even if daily_run_key exists (default False)

        Returns:
            Process run result
            
        Raises:
            ValueError: If snapshot is stale (prevents run)
        """
        process = self.registry.get(process_id)
        if not process:
            raise ValueError(f"Process {process_id} not found")

        # Check snapshot staleness before running (unless forced)
        if not force:
            snapshot = await self.storage.load_latest_snapshot()
            if snapshot is not None and snapshot.is_stale(max_age_hours=24.0):
                staleness_hours = snapshot.staleness_hours or (datetime.now() - snapshot.timestamp).total_seconds() / 3600
                raise ValueError(
                    f"Cannot run process: snapshot is stale ({staleness_hours:.1f} hours old, max 24.0 hours). "
                    f"Please run 'process scan' to create a fresh snapshot."
                )

        # Generate daily run key for idempotency
        from datetime import date

        today = date.today()
        daily_run_key = f"{process_id}_{today.isoformat()}"

        # Check if run already exists for today (unless forced)
        if not force:
            exists, existing_run_id = await self.storage.check_daily_run_key(
                daily_run_key
            )
            if exists:
                logger.info(
                    f"Process run already exists for today (idempotency check)",
                    extra={
                        "process_id": process_id,
                        "daily_run_key": daily_run_key,
                        "existing_run_id": existing_run_id,
                    },
                )
                # Load and return existing run
                runs = await self.storage.list_runs(process_id=process_id, limit=1)
                if runs and runs[0].run_id == existing_run_id:
                    return runs[0]

        # Structured log: run started
        logger.info(
            f"Process run started",
            extra={
                "process_id": process_id,
                "run_mode": mode,
                "capital_allocated_usd": capital_allocated_usd,
                "inputs": inputs,
            },
        )

        if mode == "sandbox":
            run = await self.sandbox.simulate_run(process, inputs, capital_allocated_usd)
        else:
            # Live mode - for now, delegate to sandbox (real implementation would execute)
            logger.warning("Live mode not yet implemented, using sandbox")
            run = await self.sandbox.simulate_run(process, inputs, capital_allocated_usd)

        # Save run with daily run key
        await self.storage.save_run(run, daily_run_key=daily_run_key)

        # Compute attribution for structured logging
        from hean.process_factory.truth_layer import TruthLayer

        truth_layer = TruthLayer()
        attribution = truth_layer.compute_attribution(run)

        # Structured log: run completed with ledger summary
        logger.info(
            f"Process run completed: {run.status.value}",
            extra={
                "process_id": process_id,
                "run_id": run.run_id,
                "mode": mode,
                "status": run.status.value,
                "started_at": run.started_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "capital_allocated_usd": capital_allocated_usd,
                "gross_pnl_usd": attribution.gross_pnl_usd,
                "net_pnl_usd": attribution.net_pnl_usd,
                "total_fees_usd": attribution.total_fees_usd,
                "total_funding_usd": attribution.total_funding_usd,
                "total_rewards_usd": attribution.total_rewards_usd,
                "opportunity_cost_usd": attribution.opportunity_cost_usd,
                "profit_illusion": attribution.profit_illusion,
                "ledger_entry_count": len(attribution.ledger_entries),
            },
        )

        return run

    async def update_portfolio(self) -> list[ProcessPortfolioEntry]:
        """Update portfolio based on recent runs.

        Returns:
            Updated portfolio entries
        """
        portfolio = await self.storage.load_portfolio()
        all_processes = self.registry.list_processes()

        # Create portfolio entries for all registered processes
        portfolio_dict = {entry.process_id: entry for entry in portfolio}
        for process in all_processes:
            if process.id not in portfolio_dict:
                portfolio_dict[process.id] = ProcessPortfolioEntry(
                    process_id=process.id,
                    state=ProcessPortfolioState.NEW,
                    weight=0.0,
                )

        # Update each entry based on runs
        updated_entries = []
        for process_id, entry in portfolio_dict.items():
            runs = await self.storage.list_runs(process_id=process_id, limit=100)
            entry = self.selector.update_portfolio_entry(entry, runs)
            old_state = entry.state
            new_state = self.selector.evaluate_process(entry)
            entry.state = new_state
            entry.weight = self.selector.compute_weight(entry)
            updated_entries.append(entry)

            # Structured log: state changes and kill/scale decisions
            if old_state != new_state:
                logger.info(
                    f"Process state changed: {old_state.value} -> {new_state.value}",
                    extra={
                        "process_id": process_id,
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "runs_count": entry.runs_count,
                        "pnl_sum": entry.pnl_sum,
                        "fail_rate": entry.fail_rate,
                    },
                )
            if new_state == ProcessPortfolioState.KILLED:
                logger.warning(
                    f"Process killed",
                    extra={
                        "process_id": process_id,
                        "reason": "evaluation",
                        "runs_count": entry.runs_count,
                        "pnl_sum": entry.pnl_sum,
                        "fail_rate": entry.fail_rate,
                        "max_dd": entry.max_dd,
                    },
                )

        # Save updated portfolio
        await self.storage.save_portfolio(updated_entries)

        logger.info(f"Portfolio updated: {len(updated_entries)} processes")
        return updated_entries

    def _snapshot_to_opportunities(
        self, snapshot: BybitEnvironmentSnapshot
    ) -> list[Opportunity]:
        """Convert environment snapshot to opportunities.

        Args:
            snapshot: Environment snapshot

        Returns:
            List of opportunities
        """
        opportunities: list[Opportunity] = []

        # Trading opportunities from funding rates
        for symbol, rate in snapshot.funding_rates.items():
            if abs(rate) > 0.0001:  # Significant funding rate
                expected_profit = abs(rate) * 1000  # Simplified estimate
                opportunities.append(
                    Opportunity(
                        id=f"funding_{symbol}",
                        source=OpportunitySource.TRADING,
                        expected_profit_usd=expected_profit,
                        time_hours=8.0,  # Funding rate period
                        risk_factor=2.0,
                        complexity=2,
                        confidence=0.6,
                        metadata={"symbol": symbol, "rate": rate},
                    )
                )

        # Earn opportunities (if available)
        if snapshot.earn_availability.get("status") != "UNKNOWN":
            opportunities.append(
                Opportunity(
                    id="earn_general",
                    source=OpportunitySource.EARN,
                    expected_profit_usd=10.0,  # Placeholder
                    time_hours=24.0,
                    risk_factor=1.0,
                    complexity=1,
                    confidence=0.5,
                )
            )

        # Campaign opportunities
        if snapshot.campaign_availability.get("status") != "UNKNOWN":
            opportunities.append(
                Opportunity(
                    id="campaign_general",
                    source=OpportunitySource.CAMPAIGN,
                    expected_profit_usd=50.0,  # Placeholder
                    time_hours=2.0,
                    risk_factor=1.5,
                    complexity=3,
                    confidence=0.4,
                )
            )

        return opportunities

