"""Leverage-of-Process Engine: self-amplifying loops."""


from hean.logging import get_logger
from hean.process_factory.schemas import (
    BybitEnvironmentSnapshot,
    ProcessDefinition,
    ProcessPortfolioEntry,
    ProcessRun,
)

logger = get_logger(__name__)


class LeverageEngine:
    """Leverage-of-Process Engine implementing self-amplifying loops."""

    def __init__(self) -> None:
        """Initialize leverage engine."""
        self._capability_graph: dict[str, set[str]] = {}  # capability -> set of unlocked opportunities

    def propose_capability_upgrades(
        self,
        snapshot: BybitEnvironmentSnapshot,
        portfolio: list[ProcessPortfolioEntry],
    ) -> list[ProcessDefinition]:
        """Propose process definitions that unlock new capabilities.

        Args:
            snapshot: Current environment snapshot
            portfolio: Current process portfolio

        Returns:
            List of proposed process definitions that unlock capabilities
        """
        proposals: list[ProcessDefinition] = []

        # L3 Access Leverage: Identify capability gaps
        capabilities_needed = self._identify_capability_gaps(snapshot, portfolio)
        for capability, unlocked_opportunities in capabilities_needed.items():
            # Create a process definition proposal for acquiring this capability
            # This is a simplified example - real implementation would be more sophisticated
            proposal = self._create_capability_process(capability, unlocked_opportunities)
            if proposal:
                proposals.append(proposal)

        return proposals

    def analyze_automation_leverage(
        self, runs: list[ProcessRun], process: ProcessDefinition
    ) -> ProcessDefinition | None:
        """L1 Automation Leverage: Propose automation for repeated HUMAN_TASK steps.

        Args:
            runs: Historical runs of the process
            process: Process definition

        Returns:
            Proposed automated process definition, or None if no automation candidate
        """
        # Find HUMAN_TASK steps that are repeated
        human_tasks = [step for step in process.actions if step.kind.value == "HUMAN_TASK"]

        if not human_tasks:
            return None

        # Check if human tasks have consistent outcomes
        # This is simplified - real implementation would analyze step outputs
        automation_candidates = []
        for task in human_tasks:
            # Check if this task appears in multiple runs with similar outcomes
            # For now, we'll propose automation for tasks that appear in multiple runs
            if len(runs) >= 3:
                automation_candidates.append(task)

        if not automation_candidates:
            return None

        # Create an automated version (disabled by default for safety)
        # This is a placeholder - real implementation would generate actual automation
        logger.info(
            f"Automation leverage identified for process {process.id}: "
            f"{len(automation_candidates)} steps could be automated"
        )
        # Return None for now - actual automation proposals would require more sophisticated logic
        logger.debug("[LEVERAGE_ENGINE] Automation proposal not implemented yet")
        return None

    def analyze_data_leverage(
        self, process: ProcessDefinition, portfolio: list[ProcessPortfolioEntry]
    ) -> float:
        """L2 Data Leverage: Calculate priority boost for data-producing processes.

        Args:
            process: Process definition to analyze
            portfolio: Current process portfolio

        Returns:
            Priority multiplier (1.0 = no boost, >1.0 = increased priority)
        """
        # Check if process produces predictive/diagnostic data
        # Processes that help other processes get priority boost
        data_outputs = [
            output
            for output in process.expected_outputs
            if any(
                keyword in output.lower()
                for keyword in ["metric", "data", "analysis", "suggestion", "recommendation"]
            )
        ]

        if not data_outputs:
            return 1.0

        # Count how many other processes could benefit from this data
        dependent_processes = 0
        for portfolio_entry in portfolio:
            # Check if other processes have requirements that this process's outputs satisfy
            # This is simplified - real implementation would check actual dependencies
            if portfolio_entry.process_id != process.id:
                dependent_processes += 1

        # Boost priority based on how many processes benefit
        # Max boost of 2.0x for highly useful data processes
        multiplier = 1.0 + min(dependent_processes * 0.1, 1.0)
        return multiplier

    def _identify_capability_gaps(
        self,
        snapshot: BybitEnvironmentSnapshot,
        portfolio: list[ProcessPortfolioEntry],
    ) -> dict[str, set[str]]:
        """Identify capability gaps that limit opportunities.

        Args:
            snapshot: Environment snapshot
            portfolio: Current portfolio

        Returns:
            Dict mapping capability name to set of unlocked opportunities
        """
        gaps: dict[str, set[str]] = {}

        # Example: If earn availability is UNKNOWN, that's a capability gap
        if snapshot.earn_availability.get("status") == "UNKNOWN":
            gaps["earn_api_access"] = {"earn_opportunities", "yield_farming"}

        # Example: If campaign availability is UNKNOWN, that's a capability gap
        if snapshot.campaign_availability.get("status") == "UNKNOWN":
            gaps["campaign_tracking"] = {"campaign_opportunities", "airdrop_participation"}

        # Example: If we have no API access to certain data, that's a gap
        if snapshot.source_flags.get("funding_rates") == "UNKNOWN":
            gaps["funding_rate_data"] = {"funding_arbitrage", "rate_monitoring"}

        return gaps

    def _create_capability_process(
        self, capability: str, unlocked_opportunities: set[str]
    ) -> ProcessDefinition | None:
        """Create a process definition for acquiring a capability.

        Args:
            capability: Capability name
            unlocked_opportunities: Opportunities unlocked by this capability

        Returns:
            Process definition, or None if cannot be created
        """
        # This is a placeholder - real implementation would generate actual process definitions
        # For now, return None to indicate capability processes should be manually defined
        logger.debug(
            f"Capability process proposal for {capability} "
            f"unlocking {len(unlocked_opportunities)} opportunities"
        )
        logger.debug("[LEVERAGE_ENGINE] Capability process creation not implemented yet")
        return None

    def get_meta_score_multiplier(
        self,
        process: ProcessDefinition,
        snapshot: BybitEnvironmentSnapshot,
        portfolio: list[ProcessPortfolioEntry],
    ) -> float:
        """Get meta-score multiplier for processes that unlock opportunity space.

        Args:
            process: Process definition
            snapshot: Environment snapshot
            portfolio: Current portfolio

        Returns:
            Multiplier (1.0 = no boost, >1.0 = increased score)
        """
        multiplier = 1.0

        # L2 Data Leverage
        data_multiplier = self.analyze_data_leverage(process, portfolio)
        multiplier *= data_multiplier

        # L3 Access Leverage
        capability_gaps = self._identify_capability_gaps(snapshot, portfolio)
        # Check if this process addresses capability gaps
        process_outputs = set(process.expected_outputs)
        for _capability, _opportunities in capability_gaps.items():
            # Simplified check: if process outputs could help with opportunities
            if any("opportunity" in output.lower() for output in process_outputs):
                multiplier *= 1.2  # 20% boost for access leverage

        return multiplier

