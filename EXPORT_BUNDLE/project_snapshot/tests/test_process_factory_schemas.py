"""Tests for Process Factory schemas."""

import pytest
from datetime import datetime

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    KillCondition,
    MeasurementSpec,
    ProcessDefinition,
    ProcessRun,
    ProcessRunStatus,
    ProcessType,
    SafetyPolicy,
    ScaleRule,
)


def test_action_step():
    """Test ActionStep schema."""
    step = ActionStep(
        step_id="test_step",
        kind=ActionStepKind.API_CALL,
        description="Test step",
        params={"key": "value"},
        timeout=60,
        retries=3,
        depends_on=["step1"],
    )
    assert step.step_id == "test_step"
    assert step.kind == ActionStepKind.API_CALL
    assert step.timeout == 60
    assert step.retries == 3
    assert "step1" in step.depends_on


def test_process_definition():
    """Test ProcessDefinition schema."""
    process = ProcessDefinition(
        id="test_process",
        name="Test Process",
        type=ProcessType.TRADING,
        description="Test description",
        requirements={"needs_bybit": True},
        actions=[
            ActionStep(
                step_id="step1",
                kind=ActionStepKind.COMPUTE,
                description="First step",
            )
        ],
        expected_outputs=["output1"],
        safety=SafetyPolicy(max_capital_usd=1000.0),
        measurement=MeasurementSpec(metrics=["capital_delta"]),
    )
    assert process.id == "test_process"
    assert process.type == ProcessType.TRADING
    assert len(process.actions) == 1
    assert process.actions[0].step_id == "step1"


def test_process_run():
    """Test ProcessRun schema."""
    run = ProcessRun(
        run_id="test_run",
        process_id="test_process",
        started_at=datetime.now(),
        status=ProcessRunStatus.COMPLETED,
        metrics={"capital_delta": 10.0},
        capital_allocated_usd=100.0,
    )
    assert run.run_id == "test_run"
    assert run.status == ProcessRunStatus.COMPLETED
    assert run.metrics["capital_delta"] == 10.0


def test_kill_condition():
    """Test KillCondition schema."""
    condition = KillCondition(
        metric="fail_rate",
        threshold=0.7,
        comparison=">",
        window_runs=10,
    )
    assert condition.metric == "fail_rate"
    assert condition.threshold == 0.7
    assert condition.comparison == ">"


def test_scale_rule():
    """Test ScaleRule schema."""
    rule = ScaleRule(
        metric="avg_roi",
        threshold=0.1,
        comparison=">",
        scale_multiplier=1.5,
        window_runs=5,
        max_allocation_usd=5000.0,
    )
    assert rule.metric == "avg_roi"
    assert rule.scale_multiplier == 1.5
    assert rule.max_allocation_usd == 5000.0

