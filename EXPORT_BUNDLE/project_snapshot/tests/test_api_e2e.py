"""E2E smoke tests for API."""

import pytest
from fastapi.testclient import TestClient

from hean.api.app import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_e2e_smoke():
    """E2E smoke test: health → engine status → positions."""
    # 1. Health check
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"

    # 2. Engine status
    response = client.get("/engine/status")
    assert response.status_code == 200
    status_data = response.json()
    assert "status" in status_data
    assert "running" in status_data

    # 3. Positions endpoint (should work even if engine is stopped)
    response = client.get("/positions")
    assert response.status_code == 200
    positions_data = response.json()
    assert "positions" in positions_data
    assert isinstance(positions_data["positions"], list)

    # 4. Orders endpoint
    response = client.get("/orders")
    assert response.status_code == 200
    orders_data = response.json()
    assert "orders" in orders_data
    assert isinstance(orders_data["orders"], list)

    # 5. Metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "hean_engine_status" in response.text

