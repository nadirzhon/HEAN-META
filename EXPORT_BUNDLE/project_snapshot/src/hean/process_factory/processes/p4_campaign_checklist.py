"""P4: Campaign/Airdrop Checklist Generator."""

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    MeasurementSpec,
    ProcessDefinition,
    ProcessType,
    SafetyPolicy,
)


def get_process_definition() -> ProcessDefinition:
    """Get Campaign Checklist process definition."""
    return ProcessDefinition(
        id="p4_campaign_checklist",
        name="Campaign/Airdrop Checklist Generator",
        type=ProcessType.CAMPAIGN,
        description="Generate checklist for participating in campaigns/airdrops",
        requirements={"needs_bybit": False, "needs_ui": True},
        inputs_schema={"campaign_name": {"type": "string"}},
        actions=[
            ActionStep(
                step_id="identify_campaign",
                kind=ActionStepKind.HUMAN_TASK,
                description="Identify campaign/airdrop opportunity",
                params={"checklist": ["Find campaign announcement", "Read requirements", "Note deadlines"]},
            ),
            ActionStep(
                step_id="generate_checklist",
                kind=ActionStepKind.COMPUTE,
                description="Generate participation checklist",
                depends_on=["identify_campaign"],
            ),
            ActionStep(
                step_id="verify_eligibility",
                kind=ActionStepKind.HUMAN_TASK,
                description="Verify eligibility requirements",
                params={"checklist": ["Check account status", "Verify requirements", "Note any restrictions"]},
                depends_on=["generate_checklist"],
            ),
        ],
        expected_outputs=["checklist", "eligibility_status", "action_items"],
        safety=SafetyPolicy(
            max_capital_usd=500.0,
            require_manual_approval=True,
            max_risk_factor=2.0,
        ),
        measurement=MeasurementSpec(
            metrics=["time_hours"],
            attribution_rule="direct",
        ),
        kill_conditions=[],
    )

