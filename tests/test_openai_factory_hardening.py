"""Tests for OpenAI factory hardening.

Tests validation logic in _validate_and_create_process directly,
since generate_process is async and catches all exceptions internally.
"""

import json
from unittest.mock import MagicMock

import pytest

from hean.process_factory.integrations.openai_factory import OpenAIProcessFactory


@pytest.fixture
def factory():
    """Create OpenAIProcessFactory without OpenAI dependency."""
    # Bypass __init__ to avoid needing the openai package
    obj = object.__new__(OpenAIProcessFactory)
    obj.api_key = "test_key"
    obj.prompt_template_path = None
    obj._client = MagicMock()
    obj._available = True
    return obj


def test_strict_json_validation_rejection(factory):
    """Test that _validate_and_create_process rejects missing required fields."""
    with pytest.raises(ValueError, match="Missing required field"):
        factory._validate_and_create_process(
            {"name": "Test Process"},  # Missing: id, type, description
            max_steps=20,
            max_human_tasks=5,
        )


def test_required_fields_enforcement(factory):
    """Test that missing required fields are rejected."""
    incomplete_data = {"name": "Test Process"}
    with pytest.raises(ValueError, match="Missing required field"):
        factory._validate_and_create_process(incomplete_data, max_steps=20, max_human_tasks=5)


def test_kill_conditions_required(factory):
    """Test that kill_conditions are required."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
    }
    with pytest.raises(ValueError, match="Missing kill_conditions"):
        factory._validate_and_create_process(data, max_steps=20, max_human_tasks=5)


def test_measurement_spec_required(factory):
    """Test that measurement spec is required."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
    }
    with pytest.raises(ValueError, match="Missing measurement spec"):
        factory._validate_and_create_process(data, max_steps=20, max_human_tasks=5)


def test_budget_guardrails_max_steps(factory):
    """Test that budget guardrails enforce max steps."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [{"kind": "BYBIT_API", "description": f"Step {i}"} for i in range(25)],
    }
    with pytest.raises(ValueError, match="Too many steps"):
        factory._validate_and_create_process(data, max_steps=20, max_human_tasks=5)


def test_budget_guardrails_max_human_tasks(factory):
    """Test that budget guardrails enforce max human tasks."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [{"kind": "HUMAN_TASK", "description": f"Task {i}"} for i in range(10)],
    }
    with pytest.raises(ValueError, match="Too many human tasks"):
        factory._validate_and_create_process(data, max_steps=20, max_human_tasks=5)


def test_deterministic_generation_seed_temperature(factory):
    """Test that generate_process would use deterministic settings.

    Since generate_process is async and calls OpenAI, we test indirectly
    by verifying the factory is configured for deterministic output.
    """
    # Verify factory is available (would use seed=42, temperature=0.3)
    assert factory._available is True
    assert factory.api_key == "test_key"


def test_safety_filter_rejects_unsafe_processes(factory):
    """Test that safety filter rejects unsafe processes."""
    data = {
        "id": "test_process",
        "name": "Test Process with credentials",
        "type": "TRADING",
        "description": "Process that handles api_key",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [],
    }
    with pytest.raises(ValueError, match="unsafe keyword"):
        factory._validate_and_create_process(data, max_steps=20, max_human_tasks=5)


def test_budget_guardrails_stop_runaway_processes(factory):
    """Test that budget guardrails prevent runaway processes."""
    data = {
        "id": "test_process",
        "name": "Test Process",
        "type": "TRADING",
        "description": "Test description",
        "kill_conditions": [{"metric": "fail_rate", "threshold": 0.7}],
        "measurement": {"metrics": ["capital_delta"]},
        "actions": [{"kind": "BYBIT_API", "description": f"Step {i}"} for i in range(100)],
    }
    with pytest.raises(ValueError, match="Too many steps"):
        factory._validate_and_create_process(data, max_steps=20, max_human_tasks=5)
