"""Capital routing and allocation."""

from hean.process_factory.schemas import (
    DailyCapitalPlan,
    ProcessPortfolioEntry,
    ProcessPortfolioState,
)


class CapitalRouter:
    """Routes capital across reserve/active/experimental buckets."""

    def __init__(
        self,
        reserve_pct: float = 0.4,
        active_pct: float = 0.5,
        experimental_pct: float = 0.1,
    ) -> None:
        """Initialize capital router.

        Args:
            reserve_pct: Percentage of capital to keep in reserve (default 40%)
            active_pct: Percentage for active/proven processes (default 50%)
            experimental_pct: Percentage for experimental processes (default 10%)

        Raises:
            ValueError: If percentages don't sum to 1.0
        """
        if abs(reserve_pct + active_pct + experimental_pct - 1.0) > 0.01:
            raise ValueError("Percentages must sum to 1.0")
        self.reserve_pct = reserve_pct
        self.active_pct = active_pct
        self.experimental_pct = experimental_pct

    def compute_daily_plan(
        self,
        total_capital_usd: float,
        portfolio: list[ProcessPortfolioEntry],
    ) -> DailyCapitalPlan:
        """Compute daily capital allocation plan.

        Args:
            total_capital_usd: Total available capital in USD
            portfolio: Current process portfolio

        Returns:
            Daily capital plan
        """
        from datetime import datetime

        reserve_usd = total_capital_usd * self.reserve_pct
        active_usd = total_capital_usd * self.active_pct
        experimental_usd = total_capital_usd * self.experimental_pct

        # Separate processes by state
        core_processes = [p for p in portfolio if p.state == ProcessPortfolioState.CORE]
        testing_processes = [
            p for p in portfolio if p.state == ProcessPortfolioState.TESTING
        ]
        new_processes = [p for p in portfolio if p.state == ProcessPortfolioState.NEW]

        allocations: dict[str, float] = {}

        # Allocate active capital to core processes (weighted by portfolio weights)
        if core_processes:
            total_weight = sum(p.weight for p in core_processes)
            if total_weight > 0:
                for process in core_processes:
                    allocation = (process.weight / total_weight) * active_usd
                    allocations[process.process_id] = allocation

        # Allocate experimental capital to testing/new processes
        # Never exceed experimental_pct threshold for unproven processes
        experimental_allocation = 0.0
        all_unproven = testing_processes + new_processes
        if all_unproven:
            # Limit total experimental allocation
            max_experimental = total_capital_usd * self.experimental_pct
            total_unproven_weight = sum(p.weight for p in all_unproven)
            if total_unproven_weight > 0:
                for process in all_unproven:
                    allocation = (process.weight / total_unproven_weight) * min(
                        experimental_usd, max_experimental - experimental_allocation
                    )
                    allocations[process.process_id] = allocation
                    experimental_allocation += allocation
                    if experimental_allocation >= max_experimental:
                        break

        # Ensure reserve is never used for experimental
        # Reserve is kept separate and not allocated to any process

        return DailyCapitalPlan(
            date=datetime.now(),
            reserve_usd=reserve_usd,
            active_usd=active_usd,
            experimental_usd=experimental_usd,
            allocations=allocations,
            total_capital_usd=total_capital_usd,
        )

    def validate_plan(self, plan: DailyCapitalPlan) -> bool:
        """Validate that a capital plan meets constraints.

        Args:
            plan: Capital plan to validate

        Returns:
            True if valid, False otherwise
        """
        # Check percentages
        reserve_pct = plan.reserve_usd / plan.total_capital_usd
        active_pct = plan.active_usd / plan.total_capital_usd
        experimental_pct = plan.experimental_usd / plan.total_capital_usd

        if abs(reserve_pct + active_pct + experimental_pct - 1.0) > 0.01:
            return False

        # Check that reserve matches expected percentage
        if abs(reserve_pct - self.reserve_pct) > 0.01:
            return False

        # Check that total allocations don't exceed active + experimental
        total_allocated = sum(plan.allocations.values())
        if total_allocated > plan.active_usd + plan.experimental_usd + 0.01:
            return False

        return True

