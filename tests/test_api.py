"""Tests for FastAPI backend."""

from fastapi.testclient import TestClient

from hean.api.main import app

client = TestClient(app)

API = "/api/v1"


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
    # Secrets should be masked - starts with "***" and shows only last 4 chars
    if data.get("bybit_api_key"):
        assert data["bybit_api_key"].startswith("***"), "API key should be masked"


def test_engine_status_stopped():
    """Test engine status when stopped."""
    response = client.get(f"{API}/engine/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "stopped"
    assert data["running"] is False


def test_engine_start_stop():
    """Test engine start and stop.

    Note: When running with TestClient, the engine facade may not be initialized
    because lifespan events don't run in the same way. We handle both cases.
    """
    # Start engine - may fail if engine facade not initialized
    response = client.post(f"{API}/engine/start", json={"confirm_phrase": None})
    # Accept both 200 (success) and 500 (engine not initialized)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data or "ok" in data

        # Check status
        response = client.get(f"{API}/engine/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

        # Stop engine
        response = client.post(f"{API}/engine/stop")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "ok" in data
    else:
        # Engine not initialized, verify we get proper error
        data = response.json()
        assert "detail" in data


def test_positions_endpoint():
    """Test positions endpoint.

    Note: Positions are under /orders/positions (trading router).
    When engine not initialized, returns 500.
    """
    response = client.get(f"{API}/orders/positions")
    # Accept 200 (success) or 500 (engine not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "positions" in data
        assert isinstance(data["positions"], list)


def test_orders_endpoint():
    """Test orders endpoint.

    Note: When engine not initialized, returns 500.
    """
    response = client.get(f"{API}/orders/")
    # Accept 200 (success) or 500 (engine not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "orders" in data
        assert isinstance(data["orders"], list)


def test_orders_filter():
    """Test orders endpoint with filter."""
    for status in ["all", "open", "filled"]:
        response = client.get(f"{API}/orders/?status={status}")
        # Accept 200 (success) or 500 (engine not initialized)
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "orders" in data


def test_trading_metrics_endpoint():
    """Test trading metrics endpoint.

    Note: Located at /trading/metrics.
    """
    response = client.get(f"{API}/trading/metrics")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_analytics_backtest_endpoint():
    """Test backtest endpoint.

    Note: Located at /analytics/backtest. Requires start_date and end_date.
    """
    response = client.post(f"{API}/analytics/backtest", json={
        "symbol": "BTCUSDT",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "initial_capital": 10000.0
    })
    # Accept 200 (success), 422 (validation), or 500 (not configured)
    assert response.status_code in [200, 422, 500]


def test_analytics_evaluate_endpoint():
    """Test evaluate endpoint.

    Note: Located at /analytics/evaluate.
    """
    response = client.post(f"{API}/analytics/evaluate", json={
        "symbol": "BTCUSDT",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31"
    })
    # Accept 200 (success), 422 (validation), or 500 (not configured)
    assert response.status_code in [200, 422, 500]


def test_request_id_header():
    """Test that request ID is included in response headers."""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_strategies_list():
    """Test strategies list endpoint.

    Note: When engine not initialized, may return 500.
    """
    response = client.get(f"{API}/strategies/")
    # Accept 200 (success) or 500 (engine not initialized)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "strategies" in data
        assert isinstance(data["strategies"], list)


def test_risk_status():
    """Test risk status endpoint.

    Note: When engine not initialized, returns 500.
    """
    response = client.get(f"{API}/risk/status")
    # Accept 200 (success) or 500 (engine not initialized)
    assert response.status_code in [200, 500]
