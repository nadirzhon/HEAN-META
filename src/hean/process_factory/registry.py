"""Process registry and discovery."""


from hean.logging import get_logger
from hean.process_factory.schemas import ProcessDefinition

logger = get_logger(__name__)


class ProcessRegistry:
    """Registry for process definitions (plugin system)."""

    def __init__(self) -> None:
        """Initialize process registry."""
        self._processes: dict[str, ProcessDefinition] = {}
        self._loaded = False

    def register(self, process: ProcessDefinition) -> None:
        """Register a process definition.

        Args:
            process: Process definition to register

        Raises:
            ValueError: If process ID is already registered
        """
        if process.id in self._processes:
            raise ValueError(f"Process {process.id} is already registered")
        self._validate(process)
        self._processes[process.id] = process
        logger.debug(f"Registered process: {process.id} ({process.name})")

    def get(self, process_id: str) -> ProcessDefinition | None:
        """Get a process definition by ID.

        Args:
            process_id: Process ID

        Returns:
            Process definition or None if not found
        """
        if not self._loaded:
            self._load_builtin_processes()
        return self._processes.get(process_id)

    def list_processes(self) -> list[ProcessDefinition]:
        """List all registered processes.

        Returns:
            List of process definitions
        """
        if not self._loaded:
            self._load_builtin_processes()
        return list(self._processes.values())

    def _validate(self, process: ProcessDefinition) -> None:
        """Validate a process definition.

        Args:
            process: Process definition to validate

        Raises:
            ValueError: If process definition is invalid
        """
        if not process.id:
            raise ValueError("Process ID is required")
        if not process.name:
            raise ValueError("Process name is required")
        if not process.actions:
            raise ValueError("Process must have at least one action")

        # Validate step dependencies
        step_ids = {step.step_id for step in process.actions}
        for step in process.actions:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValueError(f"Step {step.step_id} depends on unknown step {dep_id}")

        # Check for circular dependencies (simple check)
        # More sophisticated cycle detection could be added
        visited: set[str] = set()

        def check_cycle(step_id: str, path: set[str]) -> None:
            if step_id in path:
                raise ValueError(f"Circular dependency detected involving step {step_id}")
            if step_id in visited:
                return
            visited.add(step_id)
            step = next(s for s in process.actions if s.step_id == step_id)
            for dep_id in step.depends_on:
                check_cycle(dep_id, path | {step_id})

        for step in process.actions:
            if step.step_id not in visited:
                check_cycle(step.step_id, set())

    def _load_builtin_processes(self) -> None:
        """Load built-in processes from processes/ directory."""
        if self._loaded:
            return

        try:
            # Import built-in processes
            from hean.process_factory.processes import (
                p1_capital_parking,
                p2_funding_monitor,
                p3_fee_monitor,
                p4_campaign_checklist,
                p5_execution_optimizer,
                p6_opportunity_scanner,
            )

            processes = [
                p1_capital_parking.get_process_definition(),
                p2_funding_monitor.get_process_definition(),
                p3_fee_monitor.get_process_definition(),
                p4_campaign_checklist.get_process_definition(),
                p5_execution_optimizer.get_process_definition(),
                p6_opportunity_scanner.get_process_definition(),
            ]

            for process in processes:
                try:
                    self.register(process)
                except ValueError as e:
                    logger.warning(f"Failed to register built-in process {process.id}: {e}")

            self._loaded = True
            logger.info(f"Loaded {len(processes)} built-in processes")

        except ImportError as e:
            logger.warning(f"Failed to load built-in processes: {e}")

