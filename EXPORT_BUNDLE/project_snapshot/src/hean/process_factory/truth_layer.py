"""Attribution Truth Layer: compute REAL net contribution per process.

This module provides accurate profit attribution including:
- Trading fees, funding, slippage proxies
- Lockup opportunity cost for Earn-like processes
- Time-weighted capital usage
- Reward/bonus valuation (tracked as separate ledger entries)
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from hean.process_factory.schemas import ProcessRun, ProcessRunStatus


class LedgerEntryType(str, Enum):
    """Type of ledger entry."""

    PNL = "PNL"  # Gross profit/loss
    FEE = "FEE"  # Trading fees
    FUNDING = "FUNDING"  # Funding payments
    REWARD = "REWARD"  # Rewards/bonuses
    OPPORTUNITY_COST = "OPPORTUNITY_COST"  # Lockup opportunity cost


class LedgerEntry(BaseModel):
    """A single ledger entry for attribution."""

    entry_id: str = Field(..., description="Unique entry identifier")
    run_id: str = Field(..., description="Associated process run ID")
    process_id: str = Field(..., description="Process ID")
    entry_type: LedgerEntryType = Field(..., description="Type of entry")
    amount_usd: float = Field(..., description="Amount in USD (positive or negative)")
    timestamp: datetime = Field(..., description="Entry timestamp")
    description: str = Field(default="", description="Human-readable description")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (symbol, fee_type, etc.)"
    )


class AttributionResult(BaseModel):
    """Attribution result for a process run."""

    run_id: str = Field(..., description="Process run ID")
    process_id: str = Field(..., description="Process ID")
    gross_pnl_usd: float = Field(..., description="Gross PnL (before fees/costs)")
    net_pnl_usd: float = Field(..., description="Net PnL (after all costs)")
    total_fees_usd: float = Field(default=0.0, description="Total fees paid")
    total_funding_usd: float = Field(default=0.0, description="Net funding payments")
    total_rewards_usd: float = Field(default=0.0, description="Total rewards/bonuses")
    opportunity_cost_usd: float = Field(
        default=0.0, description="Opportunity cost (lockup time * risk-free rate)"
    )
    capital_locked_hours: float = Field(
        default=0.0, ge=0, description="Hours capital was locked"
    )
    time_weighted_capital_usd: float = Field(
        default=0.0, description="Time-weighted capital usage (capital * hours)"
    )
    ledger_entries: list[LedgerEntry] = Field(
        default_factory=list, description="All ledger entries for this run"
    )
    profit_illusion: bool = Field(
        default=False,
        description="True if gross positive but net negative (profit illusion)",
    )


class TruthLayer:
    """Computes real net contribution per process."""

    def __init__(
        self,
        risk_free_rate_apr: float = 0.05,
        default_maker_fee_bps: float = 0.5,
        default_taker_fee_bps: float = 3.0,
        slippage_proxy_bps: float = 1.0,
    ) -> None:
        """Initialize truth layer.

        Args:
            risk_free_rate_apr: Risk-free rate for opportunity cost (default 5% APR)
            default_maker_fee_bps: Default maker fee in basis points (default 0.5 bps)
            default_taker_fee_bps: Default taker fee in basis points (default 3.0 bps)
            slippage_proxy_bps: Slippage proxy in basis points (default 1.0 bps)
        """
        self.risk_free_rate_apr = risk_free_rate_apr
        self.default_maker_fee_bps = default_maker_fee_bps
        self.default_taker_fee_bps = default_taker_fee_bps
        self.slippage_proxy_bps = slippage_proxy_bps

    def compute_attribution(self, run: ProcessRun) -> AttributionResult:
        """Compute attribution for a process run.

        Args:
            run: Process run to analyze

        Returns:
            Attribution result with gross vs net contribution
        """
        ledger_entries: list[LedgerEntry] = []

        # Extract gross PnL from metrics
        gross_pnl = run.metrics.get("capital_delta", 0.0)
        capital_allocated = run.capital_allocated_usd
        time_hours = run.metrics.get("time_hours", 0.0)

        # Create PNL ledger entry
        if gross_pnl != 0.0:
            ledger_entries.append(
                LedgerEntry(
                    entry_id=f"{run.run_id}_pnl",
                    run_id=run.run_id,
                    process_id=run.process_id,
                    entry_type=LedgerEntryType.PNL,
                    amount_usd=gross_pnl,
                    timestamp=run.started_at,
                    description="Gross PnL from process execution",
                    metadata={"source": "metrics.capital_delta"},
                )
            )

        # Compute fees
        total_fees = self._compute_fees(run, ledger_entries)

        # Compute funding
        total_funding = self._compute_funding(run, ledger_entries)

        # Compute rewards
        total_rewards = self._compute_rewards(run, ledger_entries)

        # Compute opportunity cost (for locked capital)
        opportunity_cost = self._compute_opportunity_cost(
            run, capital_allocated, time_hours, ledger_entries
        )

        # Compute time-weighted capital
        time_weighted_capital = capital_allocated * time_hours if time_hours > 0 else 0.0

        # Net PnL = gross - fees - opportunity_cost + funding + rewards
        net_pnl = gross_pnl - total_fees - opportunity_cost + total_funding + total_rewards

        # Check for profit illusion
        profit_illusion = gross_pnl > 0 and net_pnl < 0

        return AttributionResult(
            run_id=run.run_id,
            process_id=run.process_id,
            gross_pnl_usd=gross_pnl,
            net_pnl_usd=net_pnl,
            total_fees_usd=total_fees,
            total_funding_usd=total_funding,
            total_rewards_usd=total_rewards,
            opportunity_cost_usd=opportunity_cost,
            capital_locked_hours=time_hours,
            time_weighted_capital_usd=time_weighted_capital,
            ledger_entries=ledger_entries,
            profit_illusion=profit_illusion,
        )

    def _compute_fees(
        self, run: ProcessRun, ledger_entries: list[LedgerEntry]
    ) -> float:
        """Compute trading fees for a run.

        Args:
            run: Process run
            ledger_entries: List to append fee entries to

        Returns:
            Total fees in USD
        """
        total_fees = 0.0

        # Check if fees are already in metrics
        if "fee_drag" in run.metrics:
            fee_amount = run.metrics["fee_drag"]
            if fee_amount != 0.0:
                total_fees = abs(fee_amount)
                ledger_entries.append(
                    LedgerEntry(
                        entry_id=f"{run.run_id}_fee",
                        run_id=run.run_id,
                        process_id=run.process_id,
                        entry_type=LedgerEntryType.FEE,
                        amount_usd=-total_fees,  # Fees are negative
                        timestamp=run.started_at,
                        description="Trading fees",
                        metadata={"source": "metrics.fee_drag"},
                    )
                )
            return total_fees

        # Estimate fees from capital delta and process type
        # For trading processes, estimate based on volume
        if "trading_volume_usd" in run.metrics:
            volume = run.metrics["trading_volume_usd"]
            # Assume mix of maker/taker, use weighted average
            avg_fee_bps = (self.default_maker_fee_bps + self.default_taker_fee_bps) / 2
            estimated_fees = volume * (avg_fee_bps / 10000)
            total_fees = estimated_fees
            ledger_entries.append(
                LedgerEntry(
                    entry_id=f"{run.run_id}_fee_estimated",
                    run_id=run.run_id,
                    process_id=run.process_id,
                    entry_type=LedgerEntryType.FEE,
                    amount_usd=-total_fees,
                    timestamp=run.started_at,
                    description="Estimated trading fees",
                    metadata={
                        "source": "estimated",
                        "volume_usd": volume,
                        "fee_bps": avg_fee_bps,
                    },
                )
            )
        elif run.capital_allocated_usd > 0:
            # Estimate fees as small percentage of capital for non-trading processes
            # (e.g., Earn processes may have withdrawal fees)
            estimated_fees = run.capital_allocated_usd * 0.0001  # 0.01% estimate
            total_fees = estimated_fees
            ledger_entries.append(
                LedgerEntry(
                    entry_id=f"{run.run_id}_fee_estimated",
                    run_id=run.run_id,
                    process_id=run.process_id,
                    entry_type=LedgerEntryType.FEE,
                    amount_usd=-total_fees,
                    timestamp=run.started_at,
                    description="Estimated fees (conservative)",
                    metadata={"source": "estimated", "capital_usd": run.capital_allocated_usd},
                )
            )

        return total_fees

    def _compute_funding(
        self, run: ProcessRun, ledger_entries: list[LedgerEntry]
    ) -> float:
        """Compute funding payments for a run.

        Args:
            run: Process run
            ledger_entries: List to append funding entries to

        Returns:
            Net funding in USD (positive = received, negative = paid)
        """
        total_funding = 0.0

        # Check if funding is in metrics
        if "funding_payment_usd" in run.metrics:
            funding = run.metrics["funding_payment_usd"]
            total_funding = funding
            if funding != 0.0:
                ledger_entries.append(
                    LedgerEntry(
                        entry_id=f"{run.run_id}_funding",
                        run_id=run.run_id,
                        process_id=run.process_id,
                        entry_type=LedgerEntryType.FUNDING,
                        amount_usd=funding,
                        timestamp=run.started_at,
                        description="Funding payment",
                        metadata={"source": "metrics.funding_payment_usd"},
                    )
                )

        return total_funding

    def _compute_rewards(
        self, run: ProcessRun, ledger_entries: list[LedgerEntry]
    ) -> float:
        """Compute rewards/bonuses for a run.

        Args:
            run: Process run
            ledger_entries: List to append reward entries to

        Returns:
            Total rewards in USD
        """
        total_rewards = 0.0

        # Check if rewards are in metrics
        if "reward_usd" in run.metrics:
            reward = run.metrics["reward_usd"]
            total_rewards = reward
            if reward != 0.0:
                ledger_entries.append(
                    LedgerEntry(
                        entry_id=f"{run.run_id}_reward",
                        run_id=run.run_id,
                        process_id=run.process_id,
                        entry_type=LedgerEntryType.REWARD,
                        amount_usd=reward,
                        timestamp=run.started_at,
                        description="Reward/bonus",
                        metadata={"source": "metrics.reward_usd"},
                    )
                )

        # Check outputs for rewards
        if "reward_amount" in run.outputs:
            reward = float(run.outputs.get("reward_amount", 0.0))
            total_rewards += reward
            if reward != 0.0:
                ledger_entries.append(
                    LedgerEntry(
                        entry_id=f"{run.run_id}_reward_output",
                        run_id=run.run_id,
                        process_id=run.process_id,
                        entry_type=LedgerEntryType.REWARD,
                        amount_usd=reward,
                        timestamp=run.finished_at or run.started_at,
                        description="Reward from process output",
                        metadata={"source": "outputs.reward_amount"},
                    )
                )

        return total_rewards

    def _compute_opportunity_cost(
        self,
        run: ProcessRun,
        capital_allocated: float,
        time_hours: float,
        ledger_entries: list[LedgerEntry],
    ) -> float:
        """Compute opportunity cost for locked capital.

        Args:
            run: Process run
            capital_allocated: Capital allocated in USD
            time_hours: Time capital was locked in hours
            ledger_entries: List to append opportunity cost entries to

        Returns:
            Opportunity cost in USD
        """
        if capital_allocated <= 0 or time_hours <= 0:
            return 0.0

        # Compute opportunity cost: capital * time * risk_free_rate
        # Convert APR to hourly rate
        hourly_rate = self.risk_free_rate_apr / (365 * 24)
        opportunity_cost = capital_allocated * time_hours * hourly_rate

        if opportunity_cost > 0:
            ledger_entries.append(
                LedgerEntry(
                    entry_id=f"{run.run_id}_opp_cost",
                    run_id=run.run_id,
                    process_id=run.process_id,
                    entry_type=LedgerEntryType.OPPORTUNITY_COST,
                    amount_usd=-opportunity_cost,  # Opportunity cost is negative
                    timestamp=run.started_at,
                    description=f"Opportunity cost ({time_hours:.2f}h @ {self.risk_free_rate_apr:.1%} APR)",
                    metadata={
                        "capital_usd": capital_allocated,
                        "hours": time_hours,
                        "apr": self.risk_free_rate_apr,
                    },
                )
            )

        return opportunity_cost

    def compute_portfolio_attribution(
        self, runs: list[ProcessRun]
    ) -> dict[str, AttributionResult]:
        """Compute attribution for multiple runs (grouped by process).

        Args:
            runs: List of process runs

        Returns:
            Dictionary mapping process_id to aggregated attribution result
        """
        process_attributions: dict[str, list[AttributionResult]] = {}

        # Compute attribution for each run
        for run in runs:
            if run.status not in (ProcessRunStatus.COMPLETED, ProcessRunStatus.FAILED):
                continue  # Skip incomplete runs

            attribution = self.compute_attribution(run)
            if attribution.process_id not in process_attributions:
                process_attributions[attribution.process_id] = []
            process_attributions[attribution.process_id].append(attribution)

        # Aggregate by process
        aggregated: dict[str, AttributionResult] = {}
        for process_id, attributions in process_attributions.items():
            if not attributions:
                continue

            # Sum all attributions
            total_gross = sum(a.gross_pnl_usd for a in attributions)
            total_net = sum(a.net_pnl_usd for a in attributions)
            total_fees = sum(a.total_fees_usd for a in attributions)
            total_funding = sum(a.total_funding_usd for a in attributions)
            total_rewards = sum(a.total_rewards_usd for a in attributions)
            total_opp_cost = sum(a.opportunity_cost_usd for a in attributions)
            total_hours = sum(a.capital_locked_hours for a in attributions)
            total_twc = sum(a.time_weighted_capital_usd for a in attributions)

            # Combine all ledger entries
            all_ledger_entries = []
            for a in attributions:
                all_ledger_entries.extend(a.ledger_entries)

            # Check for profit illusion
            has_illusion = any(a.profit_illusion for a in attributions) or (
                total_gross > 0 and total_net < 0
            )

            aggregated[process_id] = AttributionResult(
                run_id=f"aggregated_{process_id}",
                process_id=process_id,
                gross_pnl_usd=total_gross,
                net_pnl_usd=total_net,
                total_fees_usd=total_fees,
                total_funding_usd=total_funding,
                total_rewards_usd=total_rewards,
                opportunity_cost_usd=total_opp_cost,
                capital_locked_hours=total_hours,
                time_weighted_capital_usd=total_twc,
                ledger_entries=all_ledger_entries,
                profit_illusion=has_illusion,
            )

        return aggregated

