"""Reproducible Portfolio Evaluation.

Given a date range, replay stored snapshots & runs to compute stable metrics.
Produces a portfolio health score: stability, concentration risk, kill/scale churn.
"""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from hean.process_factory.schemas import (
    ProcessPortfolioEntry,
    ProcessPortfolioState,
    ProcessRun,
    ProcessRunStatus,
)
from hean.process_factory.storage import Storage
from hean.process_factory.truth_layer import AttributionResult, TruthLayer


class PortfolioHealthScore(BaseModel):
    """Portfolio health score metrics."""

    stability_score: float = Field(
        ..., ge=0, le=1, description="Stability score (0-1, higher is better)"
    )
    concentration_risk: float = Field(
        ..., ge=0, le=1, description="Concentration risk (0-1, lower is better)"
    )
    churn_rate: float = Field(
        ..., ge=0, description="Kill/scale churn rate (processes killed or scaled per period)"
    )
    avg_process_age_days: float = Field(
        ..., ge=0, description="Average process age in days"
    )
    net_contribution_usd: float = Field(
        ..., description="Total net contribution across all processes"
    )
    profit_illusion_count: int = Field(
        ..., ge=0, description="Number of processes with profit illusion"
    )
    core_process_count: int = Field(..., ge=0, description="Number of core processes")
    testing_process_count: int = Field(
        ..., ge=0, description="Number of testing processes"
    )
    killed_process_count: int = Field(..., ge=0, description="Number of killed processes")


class ProcessEvaluationResult(BaseModel):
    """Evaluation result for a single process."""

    process_id: str = Field(..., description="Process ID")
    recommendation: str = Field(
        ..., description="Recommendation: CORE, TESTING, KILL, or SCALE"
    )
    net_contribution_usd: float = Field(..., description="Net contribution in USD")
    runs_count: int = Field(..., ge=0, description="Number of runs")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    stability: float = Field(..., ge=0, le=1, description="Stability score (0-1)")
    reasons: list[str] = Field(
        default_factory=list, description="Reasons for recommendation"
    )


class PortfolioEvaluator:
    """Evaluates portfolio health and provides recommendations."""

    def __init__(
        self,
        storage: Storage,
        truth_layer: TruthLayer | None = None,
        min_runs_for_core: int = 20,
        min_runs_for_testing: int = 5,
        stability_window_days: int = 7,
    ) -> None:
        """Initialize portfolio evaluator.

        Args:
            storage: Storage interface
            truth_layer: Truth layer for attribution (creates default if not provided)
            min_runs_for_core: Minimum runs to consider for CORE (default 20)
            min_runs_for_testing: Minimum runs to consider for TESTING (default 5)
            stability_window_days: Days to look back for stability (default 7)
        """
        self.storage = storage
        self.truth_layer = truth_layer or TruthLayer()
        self.min_runs_for_core = min_runs_for_core
        self.min_runs_for_testing = min_runs_for_testing
        self.stability_window_days = stability_window_days

    async def evaluate_portfolio(
        self, days: int = 30
    ) -> tuple[PortfolioHealthScore, list[ProcessEvaluationResult]]:
        """Evaluate portfolio over a date range.

        Args:
            days: Number of days to look back (default 30)

        Returns:
            Tuple of (health score, process evaluation results)
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Load all runs in date range
        all_runs = await self.storage.list_runs(limit=10000)
        recent_runs = [
            r for r in all_runs if r.started_at >= cutoff_date and r.status in (
                ProcessRunStatus.COMPLETED, ProcessRunStatus.FAILED
            )
        ]

        # Load portfolio
        portfolio = await self.storage.load_portfolio()

        # Compute attribution for all runs
        attributions = self.truth_layer.compute_portfolio_attribution(recent_runs)

        # Evaluate each process
        process_results: list[ProcessEvaluationResult] = []
        for entry in portfolio:
            result = self._evaluate_process(entry, recent_runs, attributions)
            process_results.append(result)

        # Compute portfolio health score
        health_score = self._compute_health_score(portfolio, recent_runs, attributions)

        return health_score, process_results

    def _evaluate_process(
        self,
        entry: ProcessPortfolioEntry,
        runs: list[ProcessRun],
        attributions: dict[str, AttributionResult],
    ) -> ProcessEvaluationResult:
        """Evaluate a single process.

        Args:
            entry: Portfolio entry
            runs: All runs in date range
            attributions: Attribution results by process_id

        Returns:
            Evaluation result with recommendation
        """
        process_runs = [r for r in runs if r.process_id == entry.process_id]
        attribution = attributions.get(entry.process_id)

        reasons: list[str] = []
        recommendation = "KEEP"

        # Check if process should be killed
        if entry.state == ProcessPortfolioState.KILLED:
            recommendation = "KILL"
            reasons.append("Already killed")
        elif entry.fail_rate > 0.7:
            recommendation = "KILL"
            reasons.append(f"High fail rate: {entry.fail_rate:.1%}")
        elif entry.runs_count >= 10 and entry.pnl_sum < 0:
            recommendation = "KILL"
            reasons.append(f"Negative PnL after {entry.runs_count} runs")
        elif entry.max_dd > 0.25:
            recommendation = "KILL"
            reasons.append(f"High max drawdown: {entry.max_dd:.1%}")

        # Check if should be promoted to CORE
        if recommendation == "KEEP" and entry.runs_count >= self.min_runs_for_core:
            if attribution and attribution.net_pnl_usd > 0:
                stability = self._compute_stability(process_runs)
                if stability > 0.7 and entry.fail_rate < 0.3:
                    recommendation = "CORE"
                    reasons.append(
                        f"Stable performance: {stability:.1%} stability, "
                        f"{entry.fail_rate:.1%} fail rate"
                    )

        # Check if should be scaled
        if recommendation == "KEEP" and entry.runs_count >= self.min_runs_for_testing:
            if attribution and attribution.net_pnl_usd > 0:
                if entry.avg_roi > 0.1 and entry.fail_rate < 0.4:
                    recommendation = "SCALE"
                    reasons.append(
                        f"Strong performance: {entry.avg_roi:.1%} ROI, "
                        f"{entry.fail_rate:.1%} fail rate"
                    )

        # Compute stability
        stability = self._compute_stability(process_runs)

        # Compute win rate
        completed_runs = [
            r for r in process_runs
            if r.status == ProcessRunStatus.COMPLETED
        ]
        win_rate = (
            len(completed_runs) / len(process_runs)
            if process_runs else 0.0
        )

        net_contribution = (
            attribution.net_pnl_usd if attribution else entry.pnl_sum
        )

        return ProcessEvaluationResult(
            process_id=entry.process_id,
            recommendation=recommendation,
            net_contribution_usd=net_contribution,
            runs_count=entry.runs_count,
            win_rate=win_rate,
            stability=stability,
            reasons=reasons,
        )

    def _compute_stability(self, runs: list[ProcessRun]) -> float:
        """Compute stability score for a process.

        Args:
            runs: Process runs

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(runs) < 3:
            return 0.0

        # Sort by time
        sorted_runs = sorted(runs, key=lambda r: r.started_at)

        # Compute PnL variance
        pnls = [r.metrics.get("capital_delta", 0.0) for r in sorted_runs]
        if not pnls:
            return 0.0

        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std_dev = variance ** 0.5

        # Stability is inverse of coefficient of variation
        if mean_pnl == 0:
            return 0.5  # Neutral stability for zero mean
        if abs(mean_pnl) < 0.01:
            return 0.5  # Neutral for very small mean

        cv = abs(std_dev / mean_pnl) if mean_pnl != 0 else 1.0
        stability = 1.0 / (1.0 + cv)  # Normalize to 0-1

        return min(max(stability, 0.0), 1.0)

    def _compute_health_score(
        self,
        portfolio: list[ProcessPortfolioEntry],
        runs: list[ProcessRun],
        attributions: dict[str, AttributionResult],
    ) -> PortfolioHealthScore:
        """Compute portfolio health score.

        Args:
            portfolio: Portfolio entries
            runs: All runs
            attributions: Attribution results

        Returns:
            Portfolio health score
        """
        # Compute stability (average across processes)
        process_stabilities: list[float] = []
        for entry in portfolio:
            process_runs = [r for r in runs if r.process_id == entry.process_id]
            stability = self._compute_stability(process_runs)
            process_stabilities.append(stability)
        avg_stability = (
            sum(process_stabilities) / len(process_stabilities)
            if process_stabilities else 0.5
        )

        # Compute concentration risk (Herfindahl index)
        total_net = sum(
            a.net_pnl_usd for a in attributions.values() if a.net_pnl_usd > 0
        )
        if total_net > 0:
            weights = [
                max(0, a.net_pnl_usd) / total_net
                for a in attributions.values()
            ]
            herfindahl = sum(w ** 2 for w in weights)
            concentration_risk = herfindahl  # 0-1, higher = more concentrated
        else:
            concentration_risk = 0.5  # Neutral if no positive contributions

        # Compute churn rate (killed processes per period)
        killed_count = len(
            [e for e in portfolio if e.state == ProcessPortfolioState.KILLED]
        )
        # Estimate churn as killed / total
        churn_rate = (
            killed_count / len(portfolio) if portfolio else 0.0
        )

        # Compute average process age
        now = datetime.now()
        ages = []
        for entry in portfolio:
            if entry.last_run_at:
                age = (now - entry.last_run_at).total_seconds() / 86400
                ages.append(age)
        avg_age = sum(ages) / len(ages) if ages else 0.0

        # Total net contribution
        total_net = sum(a.net_pnl_usd for a in attributions.values())

        # Profit illusion count
        profit_illusion_count = sum(
            1 for a in attributions.values() if a.profit_illusion
        )

        # Count by state
        core_count = len(
            [e for e in portfolio if e.state == ProcessPortfolioState.CORE]
        )
        testing_count = len(
            [e for e in portfolio if e.state == ProcessPortfolioState.TESTING]
        )
        killed_count = len(
            [e for e in portfolio if e.state == ProcessPortfolioState.KILLED]
        )

        return PortfolioHealthScore(
            stability_score=avg_stability,
            concentration_risk=concentration_risk,
            churn_rate=churn_rate,
            avg_process_age_days=avg_age,
            net_contribution_usd=total_net,
            profit_illusion_count=profit_illusion_count,
            core_process_count=core_count,
            testing_process_count=testing_count,
            killed_process_count=killed_count,
        )

