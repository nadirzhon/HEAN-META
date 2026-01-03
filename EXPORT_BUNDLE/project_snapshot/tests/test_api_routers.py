"""Tests for API routers."""

import pytest
from fastapi.testclient import TestClient

from hean.api.app import app

client = TestClient(app)


def test_engine_pause_resume():
    """Test engine pause and resume."""
    # Start engine first
    response = client.post("/engine/start", json={"confirm_phrase": None})
    assert response.status_code == 200
    
    # Pause
    response = client.post("/engine/pause", json={})
    assert response.status_code == 200
    
    # Resume
    response = client.post("/engine/resume")
    assert response.status_code == 200


def test_positions_endpoint():
    """Test positions endpoint."""
    response = client.get("/orders/positions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_orders_endpoint():
    """Test orders endpoint."""
    response = client.get("/orders")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    
    # Test with status filter
    response = client.get("/orders?status=open")
    assert response.status_code == 200


def test_strategies_endpoint():
    """Test strategies endpoint."""
    response = client.get("/strategies")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_risk_status_endpoint():
    """Test risk status endpoint."""
    response = client.get("/risk/status")
    assert response.status_code == 200
    data = response.json()
    assert "killswitch_triggered" in data
    assert "stop_trading" in data


def test_risk_limits_endpoint():
    """Test risk limits endpoint."""
    response = client.get("/risk/limits")
    assert response.status_code == 200
    data = response.json()
    assert "max_open_positions" in data


def test_analytics_summary_endpoint():
    """Test analytics summary endpoint."""
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_trades" in data


def test_analytics_blocks_endpoint():
    """Test blocked signals analytics endpoint."""
    response = client.get("/analytics/blocks")
    assert response.status_code == 200
    data = response.json()
    assert "total_blocks" in data


def test_reconcile_endpoint():
    """Test reconcile endpoint."""
    response = client.post("/reconcile/now")
    assert response.status_code == 200


def test_smoke_test_endpoint():
    """Test smoke test endpoint."""
    response = client.post("/smoke-test/run")
    assert response.status_code == 200


def test_jobs_endpoint():
    """Test jobs endpoint."""
    response = client.get("/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_events_stream_endpoint():
    """Test events stream endpoint (SSE)."""
    response = client.get("/events/stream", stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"


def test_logs_stream_endpoint():
    """Test logs stream endpoint (SSE)."""
    response = client.get("/logs/stream", stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Check for Prometheus format
    content = response.text
    assert "hean_" in content or "# HELP" in content


def test_live_trading_protection():
    """Test that live trading actions are protected."""
    # This test would require mocking settings to be in live mode
    # For now, just verify the endpoint exists
    response = client.post("/orders/test", json={
        "symbol": "BTCUSDT",
        "side": "buy",
        "size": 0.001
    })
    # Should either succeed (paper mode) or fail with 403 (live mode)
    assert response.status_code in (200, 403)

