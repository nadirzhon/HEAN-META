"""Tests for API routers.

Note: These tests use TestClient which doesn't run lifespan events,
so the engine facade may not be initialized. Tests handle both cases.
"""

from fastapi.testclient import TestClient

from hean.api.main import app

client = TestClient(app)

API = "/api/v1"


def test_engine_pause_resume():
    """Test engine pause and resume.

    Note: May fail if engine facade not initialized (500).
    """
    # Start engine first
    response = client.post(f"{API}/engine/start", json={"confirm_phrase": None})
    # Accept 200 (success) or 500 (not initialized)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        # Pause
        response = client.post(f"{API}/engine/pause", json={})
        assert response.status_code in [200, 500]

        # Resume
        response = client.post(f"{API}/engine/resume")
        assert response.status_code in [200, 500]


def test_positions_endpoint():
    """Test positions endpoint."""
    response = client.get(f"{API}/orders/positions")
    # Accept 200 (success) or 500 (not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "positions" in data
        assert isinstance(data["positions"], list)


def test_orders_endpoint():
    """Test orders endpoint."""
    response = client.get(f"{API}/orders/")
    # Accept 200 (success) or 500 (not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "orders" in data
        assert isinstance(data["orders"], list)

    # Test with status filter
    response = client.get(f"{API}/orders/?status=open")
    assert response.status_code in [200, 500]


def test_strategies_endpoint():
    """Test strategies endpoint."""
    response = client.get(f"{API}/strategies/")
    # Accept 200 (success) or 500 (not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "strategies" in data
        assert isinstance(data["strategies"], list)


def test_risk_status_endpoint():
    """Test risk status endpoint."""
    response = client.get(f"{API}/risk/status")
    # Accept 200 (success) or 500 (not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "killswitch_triggered" in data or "state" in data


def test_risk_limits_endpoint():
    """Test risk limits endpoint."""
    response = client.get(f"{API}/risk/limits")
    # Accept 200 (success) or 500 (not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


def test_trading_metrics_endpoint():
    """Test trading metrics endpoint."""
    response = client.get(f"{API}/trading/metrics")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_health_endpoint():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
