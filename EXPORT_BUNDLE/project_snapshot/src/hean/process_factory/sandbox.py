"""Sandbox/simulated execution harness for processes."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    ProcessDefinition,
    ProcessRun,
    ProcessRunStatus,
)


class ProcessSandbox:
    """Sandbox for simulating process execution without real actions."""

    def __init__(self) -> None:
        """Initialize process sandbox."""
        pass

    async def simulate_run(
        self,
        process: ProcessDefinition,
        inputs: dict[str, Any],
        capital_allocated_usd: float = 0.0,
    ) -> ProcessRun:
        """Simulate a process run.

        Args:
            process: Process definition to simulate
            inputs: Input parameters for the process
            capital_allocated_usd: Capital allocated to this run

        Returns:
            Simulated process run
        """
        run_id = str(uuid.uuid4())
        started_at = datetime.now()

        # Create run
        run = ProcessRun(
            run_id=run_id,
            process_id=process.id,
            started_at=started_at,
            status=ProcessRunStatus.RUNNING,
            inputs=inputs,
            capital_allocated_usd=capital_allocated_usd,
        )

        try:
            # Execute steps in dependency order
            step_results: dict[str, Any] = {}
            executed_steps: set[str] = set()

            # Simple topological sort for dependencies
            remaining_steps = {step.step_id: step for step in process.actions}
            while remaining_steps:
                # Find steps with no unmet dependencies
                ready_steps = [
                    step
                    for step in remaining_steps.values()
                    if all(dep in executed_steps for dep in step.depends_on)
                ]

                if not ready_steps:
                    # Circular dependency or missing dependency
                    raise ValueError("Circular or missing dependency in process steps")

                # Execute ready steps
                for step in ready_steps:
                    result = await self._simulate_step(step, inputs, step_results)
                    step_results[step.step_id] = result
                    executed_steps.add(step.step_id)
                    del remaining_steps[step.step_id]

            # Calculate metrics based on process type and step results
            metrics = self._calculate_metrics(process, step_results, capital_allocated_usd)

            # Generate outputs
            outputs = {}
            for output_key in process.expected_outputs:
                # Try to find output in step results
                if output_key in step_results:
                    outputs[output_key] = step_results[output_key]
                else:
                    outputs[output_key] = None

            run.status = ProcessRunStatus.COMPLETED
            run.finished_at = datetime.now()
            run.metrics = metrics
            run.outputs = outputs

        except Exception as e:
            run.status = ProcessRunStatus.FAILED
            run.finished_at = datetime.now()
            run.error = str(e)
            run.metrics = {"capital_delta": -capital_allocated_usd * 0.1}  # Simulated loss

        return run

    async def _simulate_step(
        self, step: ActionStep, inputs: dict[str, Any], step_results: dict[str, Any]
    ) -> Any:
        """Simulate execution of a single step.

        Args:
            step: Action step to simulate
            inputs: Process inputs
            step_results: Results from previous steps

        Returns:
            Simulated step result
        """
        # Simulate delay
        await asyncio.sleep(0.01)  # Minimal delay for simulation

        if step.kind == ActionStepKind.HUMAN_TASK:
            # Human tasks require manual intervention - simulate as pending
            return {
                "status": "pending",
                "requires_manual_action": True,
                "description": step.description,
                "checklist": step.params.get("checklist", []),
            }

        elif step.kind == ActionStepKind.API_CALL:
            # Simulate API call
            return {
                "status": "success",
                "response": {"simulated": True, "step_id": step.step_id},
            }

        elif step.kind == ActionStepKind.COMPUTE:
            # Simulate computation
            return {"result": "computed", "step_id": step.step_id}

        elif step.kind == ActionStepKind.WAIT:
            # Simulate wait
            wait_seconds = step.params.get("seconds", 1)
            await asyncio.sleep(min(wait_seconds, 0.1))  # Cap wait time in simulation
            return {"status": "waited", "seconds": wait_seconds}

        return {"status": "unknown", "step_id": step.step_id}

    def _calculate_metrics(
        self,
        process: ProcessDefinition,
        step_results: dict[str, Any],
        capital_allocated_usd: float,
    ) -> dict[str, Any]:
        """Calculate metrics from step results.

        Args:
            process: Process definition
            step_results: Results from all steps
            capital_allocated_usd: Capital allocated

        Returns:
            Metrics dictionary
        """
        metrics: dict[str, Any] = {}

        # Simulate time based on number of steps
        time_hours = len(process.actions) * 0.1  # 0.1 hours per step
        metrics["time_hours"] = time_hours

        # Simulate capital delta based on process type
        # This is a very simple simulation - real implementation would be more sophisticated
        if capital_allocated_usd > 0:
            if process.type.value == "TRADING":
                # Simulate trading: random small profit/loss
                import random

                pnl_multiplier = random.uniform(-0.05, 0.10)  # -5% to +10%
                capital_delta = capital_allocated_usd * pnl_multiplier
                metrics["capital_delta"] = capital_delta
                metrics["roi"] = pnl_multiplier
            elif process.type.value == "EARN":
                # Simulate earn: small positive return
                earn_rate = 0.001  # 0.1% per run
                capital_delta = capital_allocated_usd * earn_rate
                metrics["capital_delta"] = capital_delta
                metrics["roi"] = earn_rate
            else:
                # Other types: minimal/no capital impact
                metrics["capital_delta"] = 0.0
                metrics["roi"] = 0.0
        else:
            metrics["capital_delta"] = 0.0
            metrics["roi"] = 0.0

        # Other metrics
        metrics["drawdown"] = abs(min(0, metrics.get("capital_delta", 0)))
        metrics["fail_rate"] = 0.0 if metrics.get("capital_delta", 0) >= 0 else 1.0
        metrics["volatility_exposure"] = 0.0
        metrics["fee_drag"] = 0.0

        return metrics

