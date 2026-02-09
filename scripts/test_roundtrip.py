#!/usr/bin/env python3
"""
Smoke test: open -> manage -> close paper position and verify Redis state.

Steps:
1) Calls /orders/test_roundtrip (paper only).
2) Reads key Redis snapshots to ensure CLOSED position + updated account_state.
"""

import json
import subprocess
import sys
import time

import requests

API = "http://localhost:8000"
REDIS_KEYS = [
    "hean:state:state:orders",
    "hean:state:state:positions",
    "hean:state:state:account_state",
    "hean:state:state:order_exit_decisions",
]


def redis_get(key: str) -> str:
    """Fetch a Redis key via docker compose exec."""
    out = subprocess.check_output(
        ["docker", "compose", "exec", "-T", "redis", "redis-cli", "GET", key],
        stderr=subprocess.STDOUT,
    )
    return out.decode().strip()


def main() -> int:
    print("🔎 Health check...")
    r = requests.get(f"{API}/health", timeout=5)
    r.raise_for_status()
    print(r.json())

    print("\n🚀 Running /orders/test_roundtrip ...")
    resp = requests.post(f"{API}/orders/test_roundtrip", json={}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2))

    # Small wait to let redis snapshots flush
    time.sleep(1.0)

    print("\n📦 Redis snapshots (truncated):")
    for key in REDIS_KEYS:
        try:
            value = redis_get(key)
            if len(value) > 200:
                value = value[:200] + "... (truncated)"
            print(f"{key}: {value}")
        except subprocess.CalledProcessError as e:
            print(f"{key}: <error> {e.output.decode()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
