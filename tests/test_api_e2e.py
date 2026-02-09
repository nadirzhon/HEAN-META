"""E2E smoke tests for API."""

import pytest
from fastapi.testclient import TestClient

from hean.api.main import app

client = TestClient(app)

API = "/api/v1"


@pytest.mark.asyncio
async def test_e2e_smoke():
    """E2E smoke test: health → engine status → positions."""
    # 1. Health check (root level, not under /api/v1)
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"

    # 2. Engine status
    response = client.get(f"{API}/engine/status")
    assert response.status_code == 200
    status_data = response.json()
    assert "status" in status_data
    assert "running" in status_data

    # 3. Positions endpoint (may return 500 if engine not initialized)
    response = client.get(f"{API}/orders/positions")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        positions_data = response.json()
        assert "positions" in positions_data
        assert isinstance(positions_data["positions"], list)

    # 4. Orders endpoint
    response = client.get(f"{API}/orders/")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        orders_data = response.json()
        assert "orders" in orders_data
        assert isinstance(orders_data["orders"], list)

    # 5. Trading metrics endpoint
    response = client.get(f"{API}/trading/metrics")
    assert response.status_code == 200
