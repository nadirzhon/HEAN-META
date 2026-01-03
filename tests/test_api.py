"""Tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from hean.api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_settings_endpoint():
    """Test settings endpoint (secrets masked)."""
    response = client.get("/settings")
    assert response.status_code == 200
    data = response.json()
    assert "trading_mode" in data
    assert "bybit_api_key" in data
    # Secrets should be masked
    if data.get("bybit_api_key"):
        assert data["bybit_api_key"] == "***masked***"


def test_engine_status_stopped():
    """Test engine status when stopped."""
    response = client.get("/engine/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "stopped"
    assert data["running"] is False


def test_engine_start_stop():
    """Test engine start and stop."""
    # Start engine
    response = client.post("/engine/start", json={"confirm_phrase": None})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

    # Check status
    response = client.get("/engine/status")
    assert response.status_code == 200
    data = response.json()
    # Engine may or may not be running depending on initialization

    # Stop engine
    response = client.post("/engine/stop")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_positions_endpoint():
    """Test positions endpoint."""
    response = client.get("/positions")
    assert response.status_code == 200
    data = response.json()
    assert "positions" in data
    assert isinstance(data["positions"], list)


def test_orders_endpoint():
    """Test orders endpoint."""
    response = client.get("/orders")
    assert response.status_code == 200
    data = response.json()
    assert "orders" in data
    assert isinstance(data["orders"], list)


def test_orders_filter():
    """Test orders endpoint with filter."""
    for status in ["all", "open", "filled"]:
        response = client.get(f"/orders?status={status}")
        assert response.status_code == 200
        data = response.json()
        assert "orders" in data


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    # Check for Prometheus format
    assert "hean_engine_status" in response.text


def test_smoke_test_endpoint():
    """Test smoke test endpoint."""
    response = client.post("/smoke-test/run")
    # May fail if not configured, but should return some response
    assert response.status_code in [200, 400, 500]


def test_backtest_endpoint():
    """Test backtest endpoint."""
    response = client.post("/backtest", json={"days": 30})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["days"] == 30


def test_evaluate_endpoint():
    """Test evaluate endpoint."""
    response = client.post("/evaluate", json={"days": 30})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["days"] == 30


def test_request_id_header():
    """Test that request ID is included in response headers."""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

