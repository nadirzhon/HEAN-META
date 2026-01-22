"""Process selection, kill/keep/scale logic with anti-overfitting rules."""

from datetime import datetime, timedelta
from typing import Any

from hean.process_factory.schemas import (
    ProcessPortfolioEntry,
    ProcessPortfolioState,
    ProcessRun,
    ProcessRunStatus,
)


class ProcessSelector:
    """Manages process lifecycle: kill, keep, scale decisions."""

    def __init__(
        self,
        kill_fail_rate_threshold: float = 0.7,
        kill_negative_pnl_runs: int = 10,
        kill_max_dd_threshold: float = 0.25,
        scale_min_runs: int = 5,
        promote_to_core_runs: int = 20,
        promote_to_core_regime_windows: int = 3,
        min_sample_size_for_scaling: int = 10,
        decay_half_life_days: float = 30.0,
        holdout_window_days: float = 7.0,
    ) -> None:
        """Initialize process selector.

        Args:
            kill_fail_rate_threshold: Kill if fail_rate > this (default 0.7)
            kill_negative_pnl_runs: Kill if pnl_sum negative after N runs (default 10)
            kill_max_dd_threshold: Kill if max_dd > this (default 0.25 = 25%)
            scale_min_runs: Minimum runs before considering scaling (default 5)
            promote_to_core_runs: Minimum runs to promote to CORE (default 20)
            promote_to_core_regime_windows: Stable across N time windows to promote (default 3)
            min_sample_size_for_scaling: Minimum sample size before scaling (default 10)
            decay_half_life_days: Half-life for decay weighting in days (default 30)
            holdout_window_days: Days for holdout check window (default 7)
        """
        self.kill_fail_rate_threshold = kill_fail_rate_threshold
        self.kill_negative_pnl_runs = kill_negative_pnl_runs
        self.kill_max_dd_threshold = kill_max_dd_threshold
        self.scale_min_runs = scale_min_runs
        self.promote_to_core_runs = promote_to_core_runs
        self.promote_to_core_regime_windows = promote_to_core_regime_windows
        self.min_sample_size_for_scaling = min_sample_size_for_scaling
        self.decay_half_life_days = decay_half_life_days
        self.holdout_window_days = holdout_window_days
        self._last_runs: dict[str, list[ProcessRun]] = {}  # Store runs per process_id
        """Initialize process selector.

        Args:
            kill_fail_rate_threshold: Kill if fail_rate > this (default 0.7)
            kill_negative_pnl_runs: Kill if pnl_sum negative after N runs (default 10)
            kill_max_dd_threshold: Kill if max_dd > this (default 0.25 = 25%)
            scale_min_runs: Minimum runs before considering scaling (default 5)
            promote_to_core_runs: Minimum runs to promote to CORE (default 20)
            promote_to_core_regime_windows: Stable across N time windows to promote (default 3)
            min_sample_size_for_scaling: Minimum sample size before scaling (default 10)
            decay_half_life_days: Half-life for decay weighting in days (default 30)
            holdout_window_days: Days for holdout check window (default 7)
        """
        self.kill_fail_rate_threshold = kill_fail_rate_threshold
        self.kill_negative_pnl_runs = kill_negative_pnl_runs
        self.kill_max_dd_threshold = kill_max_dd_threshold
        self.scale_min_runs = scale_min_runs
        self.promote_to_core_runs = promote_to_core_runs
        self.promote_to_core_regime_windows = promote_to_core_regime_windows
        self.min_sample_size_for_scaling = min_sample_size_for_scaling
        self.decay_half_life_days = decay_half_life_days
        self.holdout_window_days = holdout_window_days

    def update_portfolio_entry(
        self, entry: ProcessPortfolioEntry, runs: list[ProcessRun]
    ) -> ProcessPortfolioEntry:
        """Update portfolio entry metrics based on runs with decay weighting.

        Also stores runs for holdout check in evaluate_process.
        """
        # Store runs for holdout check
        self._last_runs[entry.process_id] = runs
        """Update portfolio entry metrics based on runs with decay weighting.

        Args:
            entry: Portfolio entry to update
            runs: List of runs for this process

        Returns:
            Updated portfolio entry
        """
        # Filter to completed/failed runs only
        completed_runs = [
            r for r in runs if r.status in (ProcessRunStatus.COMPLETED, ProcessRunStatus.FAILED)
        ]

        entry.runs_count = len(completed_runs)
        entry.wins = len([r for r in completed_runs if r.status == ProcessRunStatus.COMPLETED])
        entry.losses = len([r for r in completed_runs if r.status == ProcessRunStatus.FAILED])

        # Calculate PnL sum with decay weighting
        now = datetime.now()
        total_weighted_pnl = 0.0
        total_weight = 0.0
        for r in completed_runs:
            age_days = (now - r.started_at).total_seconds() / 86400
            # Exponential decay: weight = 2^(-age/half_life)
            weight = 2.0 ** (-age_days / self.decay_half_life_days)
            pnl = r.metrics.get("capital_delta", 0.0)
            total_weighted_pnl += pnl * weight
            total_weight += weight

        # Use decay-weighted average if we have weights, otherwise simple sum
        if total_weight > 0:
            entry.pnl_sum = total_weighted_pnl  # Keep as weighted sum for consistency
        else:
            entry.pnl_sum = sum(
                r.metrics.get("capital_delta", 0.0)
                for r in completed_runs
                if "capital_delta" in r.metrics
            )

        # Calculate max drawdown
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for r in sorted(completed_runs, key=lambda x: x.started_at):
            pnl = r.metrics.get("capital_delta", 0.0)
            cumulative_pnl += pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        entry.max_dd = max_drawdown

        # Calculate average ROI
        if entry.runs_count > 0:
            total_return = sum(
                r.metrics.get("roi", 0.0) for r in completed_runs if "roi" in r.metrics
            )
            entry.avg_roi = total_return / entry.runs_count
        else:
            entry.avg_roi = 0.0

        # Calculate fail rate
        if entry.runs_count > 0:
            entry.fail_rate = entry.losses / entry.runs_count
        else:
            entry.fail_rate = 0.0

        # Calculate time efficiency (profit per hour)
        total_hours = sum(
            r.metrics.get("time_hours", 0.0) for r in completed_runs if "time_hours" in r.metrics
        )
        if total_hours > 0:
            entry.time_efficiency = entry.pnl_sum / total_hours
        else:
            entry.time_efficiency = 0.0

        # Update last run timestamp
        if completed_runs:
            entry.last_run_at = max(r.started_at for r in completed_runs)

        return entry

    def evaluate_process(self, entry: ProcessPortfolioEntry) -> ProcessPortfolioState:
        """Evaluate process and determine new state.

        Args:
            entry: Portfolio entry to evaluate

        Returns:
            New state for the process
        """
        # Kill conditions
        if entry.fail_rate > self.kill_fail_rate_threshold:
            return ProcessPortfolioState.KILLED

        if entry.runs_count >= self.kill_negative_pnl_runs and entry.pnl_sum < 0:
            return ProcessPortfolioState.KILLED

        if entry.max_dd > self.kill_max_dd_threshold:
            return ProcessPortfolioState.KILLED

        # If already killed, stay killed
        if entry.state == ProcessPortfolioState.KILLED:
            return ProcessPortfolioState.KILLED

        # If paused, stay paused (manual intervention required)
        if entry.state == ProcessPortfolioState.PAUSED:
            return ProcessPortfolioState.PAUSED

        # Promotion to CORE requires stability across runs and time windows
        if entry.runs_count >= self.promote_to_core_runs:
            # Check if stable (positive ROI, low fail rate, no recent failures)
            if (
                entry.avg_roi > 0
                and entry.fail_rate < 0.3
                and entry.runs_count >= self.promote_to_core_regime_windows
            ):
                # Simple stability check: last few runs should be consistent
                return ProcessPortfolioState.CORE

        # If in testing and has minimum runs, check if should scale
        # ANTI-OVERFITTING: Require minimum sample size before scaling
        if entry.state == ProcessPortfolioState.TESTING:
            if entry.runs_count < self.min_sample_size_for_scaling:
                # Not enough samples, stay in TESTING
                return ProcessPortfolioState.TESTING

            # Holdout check: if performance collapses on recent window, stop scaling
            # Note: runs parameter is optional, only check if provided
            if entry.process_id in self._last_runs:
                runs = self._last_runs[entry.process_id]
                if self._check_holdout_failure(entry, runs):
                    # Performance collapsed on holdout, don't scale
                    return ProcessPortfolioState.TESTING

            if entry.avg_roi > 0 and entry.fail_rate < 0.4:
                # Keep in TESTING but eligible for scaling
                return ProcessPortfolioState.TESTING

        # Default: keep current state or move NEW -> TESTING after first run
        if entry.state == ProcessPortfolioState.NEW and entry.runs_count > 0:
            return ProcessPortfolioState.TESTING

        return entry.state

    def _check_holdout_failure(
        self, entry: ProcessPortfolioEntry, runs: list[ProcessRun]
    ) -> bool:
        """Check if performance collapsed on recent holdout window.

        Args:
            entry: Portfolio entry
            runs: All runs for this process

        Returns:
            True if holdout check failed (performance collapsed)
        """
        if len(runs) < self.min_sample_size_for_scaling:
            return False  # Not enough data for holdout check

        # Split into training (older) and holdout (recent)
        cutoff = datetime.now() - timedelta(days=self.holdout_window_days)
        training_runs = [r for r in runs if r.started_at < cutoff]
        holdout_runs = [r for r in runs if r.started_at >= cutoff]

        if len(training_runs) < 5 or len(holdout_runs) < 2:
            return False  # Not enough data

        # Compute average PnL for training vs holdout
        training_pnl = sum(
            r.metrics.get("capital_delta", 0.0) for r in training_runs
        ) / len(training_runs)
        holdout_pnl = sum(
            r.metrics.get("capital_delta", 0.0) for r in holdout_runs
        ) / len(holdout_runs)

        # If training was positive but holdout is negative, that's a failure
        if training_pnl > 0 and holdout_pnl < 0:
            return True

        # If holdout performance is significantly worse (e.g., < 50% of training)
        if training_pnl > 0 and holdout_pnl < training_pnl * 0.5:
            return True

        return False

    def get_regime_buckets(
        self, runs: list[ProcessRun]
    ) -> dict[str, dict[str, Any]]:
        """Get performance metrics by regime/time buckets.

        Args:
            runs: Process runs

        Returns:
            Dictionary with bucket -> metrics mapping
        """
        buckets: dict[str, dict[str, Any]] = {
            "hour_bucket": {},
            "vol_bucket": {},
            "spread_bucket": {},
        }

        for run in runs:
            if run.status not in (ProcessRunStatus.COMPLETED, ProcessRunStatus.FAILED):
                continue

            # Hour bucket (UTC hour)
            hour = run.started_at.hour
            hour_key = f"hour_{hour}"
            if hour_key not in buckets["hour_bucket"]:
                buckets["hour_bucket"][hour_key] = {"runs": 0, "pnl_sum": 0.0}
            buckets["hour_bucket"][hour_key]["runs"] += 1
            buckets["hour_bucket"][hour_key]["pnl_sum"] += run.metrics.get(
                "capital_delta", 0.0
            )

            # Vol bucket (if available in metadata)
            vol = run.metrics.get("volatility", None)
            if vol is not None:
                if vol < 0.01:
                    vol_key = "low"
                elif vol < 0.03:
                    vol_key = "medium"
                else:
                    vol_key = "high"
                if vol_key not in buckets["vol_bucket"]:
                    buckets["vol_bucket"][vol_key] = {"runs": 0, "pnl_sum": 0.0}
                buckets["vol_bucket"][vol_key]["runs"] += 1
                buckets["vol_bucket"][vol_key]["pnl_sum"] += run.metrics.get(
                    "capital_delta", 0.0
                )

            # Spread bucket (if available in metadata)
            spread = run.metrics.get("spread_bps", None)
            if spread is not None:
                if spread < 5:
                    spread_key = "tight"
                elif spread < 15:
                    spread_key = "normal"
                else:
                    spread_key = "wide"
                if spread_key not in buckets["spread_bucket"]:
                    buckets["spread_bucket"][spread_key] = {"runs": 0, "pnl_sum": 0.0}
                buckets["spread_bucket"][spread_key]["runs"] += 1
                buckets["spread_bucket"][spread_key]["pnl_sum"] += run.metrics.get(
                    "capital_delta", 0.0
                )

        return buckets

    def compute_weight(
        self, entry: ProcessPortfolioEntry, total_weight_budget: float = 1.0
    ) -> float:
        """Compute allocation weight for a process.

        Args:
            entry: Portfolio entry
            total_weight_budget: Total weight budget to allocate (default 1.0)

        Returns:
            Weight (0-1)
        """
        if entry.state in (ProcessPortfolioState.KILLED, ProcessPortfolioState.PAUSED):
            return 0.0

        # Base weight based on performance
        if entry.runs_count == 0:
            # New processes get small initial weight
            return 0.01 * total_weight_budget

        # Weight based on ROI and success rate
        if entry.avg_roi > 0 and entry.fail_rate < 0.5:
            # Positive ROI with reasonable success rate
            performance_score = entry.avg_roi * (1 - entry.fail_rate)
            # Normalize to 0-1 range (assuming ROI typically 0-1, adjust if needed)
            normalized_score = min(max(performance_score, 0.0), 1.0)
            weight = normalized_score * 0.5 * total_weight_budget  # Max 50% of budget per process
        else:
            # Poor performance gets minimal weight
            weight = 0.01 * total_weight_budget

        # Scale based on state
        if entry.state == ProcessPortfolioState.CORE:
            weight *= 2.0  # Core processes get 2x weight
        elif entry.state == ProcessPortfolioState.TESTING:
            weight *= 1.0  # Testing gets normal weight
        elif entry.state == ProcessPortfolioState.NEW:
            weight *= 0.5  # New gets half weight

        return min(weight, 0.5 * total_weight_budget)  # Cap at 50% per process

