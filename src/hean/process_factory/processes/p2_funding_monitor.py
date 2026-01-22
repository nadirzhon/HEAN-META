"""P2: Funding Monitor & Allocation Suggestion."""

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    MeasurementSpec,
    ProcessDefinition,
    ProcessType,
    SafetyPolicy,
)


def get_process_definition() -> ProcessDefinition:
    """Get Funding Monitor process definition."""
    return ProcessDefinition(
        id="p2_funding_monitor",
        name="Funding Monitor & Allocation Suggestion",
        type=ProcessType.TRADING,
        description="Monitor funding rates and suggest capital allocation (no orders)",
        requirements={"needs_bybit": True, "needs_ui": False},
        inputs_schema={},
        actions=[
            ActionStep(
                step_id="fetch_funding_rates",
                kind=ActionStepKind.API_CALL,
                description="Fetch current funding rates for major pairs",
                params={"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
            ),
            ActionStep(
                step_id="analyze_opportunities",
                kind=ActionStepKind.COMPUTE,
                description="Analyze funding rate opportunities",
            ),
            ActionStep(
                step_id="generate_allocation_suggestion",
                kind=ActionStepKind.COMPUTE,
                description="Generate capital allocation suggestions",
                depends_on=["fetch_funding_rates", "analyze_opportunities"],
            ),
        ],
        expected_outputs=["funding_rates", "allocation_suggestions"],
        safety=SafetyPolicy(
            max_capital_usd=0.0,  # Read-only, no capital allocation
            require_manual_approval=False,
            max_risk_factor=1.0,
        ),
        measurement=MeasurementSpec(
            metrics=["time_hours"],
            attribution_rule="direct",
        ),
        kill_conditions=[],
    )

