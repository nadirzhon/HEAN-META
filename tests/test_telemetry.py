"""Minimal telemetry smoke tests."""

import pytest
from fastapi.testclient import TestClient

from hean.api.app import app
from hean.api.telemetry.service import TelemetryService


@pytest.mark.asyncio
async def test_telemetry_service_emits_heartbeat():
    sent: list[tuple[str, dict]] = []

    async def fake_broadcast(topic: str, data: dict) -> None:
        sent.append((topic, data))

    service = TelemetryService(window_seconds=3)
    service.set_broadcast(fake_broadcast)

    env = await service.emit_heartbeat(
        engine_state="RUNNING",
        mode="PAPER",
        ws_clients=2,
        events_per_sec=1.0,
        source="test",
    )

    assert service.last_heartbeat() is not None
    assert sent, "Heartbeat should be broadcast to websocket topic"
    assert sent[0][0] == "system_heartbeat"
    assert env.payload["engine_state"] == "RUNNING"


def test_telemetry_ping_endpoint():
    client = TestClient(app)
    resp = client.get("/telemetry/ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
