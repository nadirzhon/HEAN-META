"""P5: Execution Quality Optimizer Suggestions."""

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    MeasurementSpec,
    ProcessDefinition,
    ProcessType,
    SafetyPolicy,
)


def get_process_definition() -> ProcessDefinition:
    """Get Execution Optimizer process definition."""
    return ProcessDefinition(
        id="p5_execution_optimizer",
        name="Execution Quality Optimizer Suggestions",
        type=ProcessType.DATA,
        description="Analyze execution quality and suggest optimizations (no orders)",
        requirements={"needs_bybit": True, "needs_ui": False},
        inputs_schema={},
        actions=[
            ActionStep(
                step_id="fetch_order_history",
                kind=ActionStepKind.API_CALL,
                description="Fetch recent order history",
                params={"limit": 100},
            ),
            ActionStep(
                step_id="analyze_execution_quality",
                kind=ActionStepKind.COMPUTE,
                description="Analyze execution quality metrics",
                depends_on=["fetch_order_history"],
            ),
            ActionStep(
                step_id="generate_optimization_suggestions",
                kind=ActionStepKind.COMPUTE,
                description="Generate optimization suggestions",
                depends_on=["analyze_execution_quality"],
            ),
        ],
        expected_outputs=["execution_metrics", "optimization_suggestions"],
        safety=SafetyPolicy(
            max_capital_usd=0.0,  # Read-only
            require_manual_approval=False,
            max_risk_factor=1.0,
        ),
        measurement=MeasurementSpec(
            metrics=["time_hours"],
            attribution_rule="direct",
        ),
        kill_conditions=[],
    )

