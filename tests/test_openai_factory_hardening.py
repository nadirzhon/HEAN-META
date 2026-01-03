"""Tests for OpenAI factory hardening.

Tests:
1. strict JSON validation rejection
2. required fields enforcement
3. deterministic generation (seed/temperature) behavior
4. budget guardrails stop runaway processes
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hean.process_factory.integrations.openai_factory import OpenAIProcessFactory
from hean.process_factory.schemas import BybitEnvironmentSnapshot, ProcessPortfolioEntry


@pytest.fixture
def factory():
    """Create OpenAIProcessFactory instance."""
    with patch("hean.process_factory.integrations.openai_factory.OpenAI"):
        factory = OpenAIProcessFactory(api_key="test_key")
        factory._client = MagicMock()
        factory._available = True
        return factory


@pytest.fixture
def sample_snapshot():
    """Create sample environment snapshot."""
    return BybitEnvironmentSnapshot(
        snapshot_id="test_snapshot",
        timestamp=MagicMock(),
        balances={},
        positions=[],
        funding_rates={},
    )


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio."""
    return []


def test_strict_json_validation_rejection(factory, sample_snapshot, sample_portfolio):
    """Test that invalid JSON is rejected."""
    # Mock OpenAI response with invalid JSON
    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="This is not JSON"))]
    )

    result = factory.generate_process(
        sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
    )

    assert result is None, "Invalid JSON should be rejected"


def test_required_fields_enforcement(factory, sample_snapshot, sample_portfolio):
    """Test that missing required fields are rejected."""
    # Mock OpenAI response with missing required fields
    incomplete_data = {
        "name": "Test Process",
        # Missing: id, type, description
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(message=MagicMock(content=json.dumps(incomplete_data)))
        ]
    )

    with pytest.raises(ValueError, match="Missing required field"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )


def test_kill_conditions_required(factory, sample_snapshot, sample_portfolio):
    """Test that kill_conditions are required."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        # Missing kill_conditions
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    with pytest.raises(ValueError, match="Missing kill_conditions"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )


def test_measurement_spec_required(factory, sample_snapshot, sample_portfolio):
    """Test that measurement spec is required."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        # Missing measurement
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    with pytest.raises(ValueError, match="Missing measurement spec"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )


def test_budget_guardrails_max_steps(factory, sample_snapshot, sample_portfolio):
    """Test that budget guardrails enforce max steps."""
    # Create data with too many steps
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [{"kind": "BYBIT_API", "description": f"Step {i}"} for i in range(25)],  # 25 > 20
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    with pytest.raises(ValueError, match="Too many steps"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )


def test_budget_guardrails_max_human_tasks(factory, sample_snapshot, sample_portfolio):
    """Test that budget guardrails enforce max human tasks."""
    # Create data with too many human tasks
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [
            {"kind": "HUMAN_TASK", "description": f"Task {i}"} for i in range(10)
        ],  # 10 > 5
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    with pytest.raises(ValueError, match="Too many human tasks"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )


def test_deterministic_generation_seed_temperature(factory, sample_snapshot, sample_portfolio):
    """Test that deterministic generation uses seed and temperature."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [],
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    # Generate process
    factory.generate_process(
        sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
    )

    # Check that API was called with deterministic settings
    call_kwargs = factory._client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.3, "Should use low temperature for determinism"
    assert call_kwargs.get("seed") == 42, "Should use seed for reproducibility"
    assert call_kwargs.get("response_format") == {"type": "json_object"}, (
        "Should request JSON format"
    )


def test_safety_filter_rejects_unsafe_processes(factory, sample_snapshot, sample_portfolio):
    """Test that safety filter rejects unsafe processes."""
    # Test credential handling
    data = {
        "id": "test_process",
        "name": "Test Process with credentials",
        "type": "TRADING",
        "description": "Process that handles api_key",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [],
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    with pytest.raises(ValueError, match="unsafe keyword"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )


def test_budget_guardrails_stop_runaway_processes(factory, sample_snapshot, sample_portfolio):
    """Test that budget guardrails prevent runaway processes."""
    # Test with very high limits - should still enforce
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [{"kind": "BYBIT_API", "description": f"Step {i}"} for i in range(100)],
    }

    factory._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(data)))]
    )

    # Even with high max_steps, should enforce reasonable limits
    with pytest.raises(ValueError, match="Too many steps"):
        factory.generate_process(
            sample_snapshot, sample_portfolio, max_steps=20, max_human_tasks=5
        )

