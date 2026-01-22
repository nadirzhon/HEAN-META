#!/usr/bin/env python3
"""
Smoke test for Small Capital Profit Mode.

Verifies:
1. REST endpoints (/telemetry/ping, /telemetry/summary, /trading/why) include cost/edge fields
2. WS connection and subscription works
3. Small capital mode config is loaded
4. All new modules import correctly
"""

import asyncio
import json
import sys
import time
from datetime import datetime

import httpx
import websockets

# Configuration
API_BASE = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"
TIMEOUT = 10.0


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_test(name: str, status: str, message: str = ""):
    """Print test result with color coding."""
    if status == "PASS":
        symbol = f"{Colors.GREEN}✓{Colors.RESET}"
        status_str = f"{Colors.GREEN}{Colors.BOLD}PASS{Colors.RESET}"
    elif status == "FAIL":
        symbol = f"{Colors.RED}✗{Colors.RESET}"
        status_str = f"{Colors.RED}{Colors.BOLD}FAIL{Colors.RESET}"
    else:  # SKIP or INFO
        symbol = f"{Colors.YELLOW}●{Colors.RESET}"
        status_str = f"{Colors.YELLOW}{status}{Colors.RESET}"

    msg_str = f" - {message}" if message else ""
    print(f"{symbol} [{status_str}] {name}{msg_str}")


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


async def test_rest_endpoints():
    """Test REST API endpoints for small capital mode fields."""
    print_section("REST API Tests")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Test 1: /telemetry/ping
        try:
            resp = await client.get(f"{API_BASE}/telemetry/ping")
            if resp.status_code == 200:
                data = resp.json()
                if "status" in data and data["status"] == "ok":
                    print_test("/telemetry/ping", "PASS", "Basic connectivity OK")
                else:
                    print_test("/telemetry/ping", "FAIL", f"Unexpected response: {data}")
            else:
                print_test("/telemetry/ping", "FAIL", f"Status code: {resp.status_code}")
        except Exception as e:
            print_test("/telemetry/ping", "FAIL", str(e))

        # Test 2: /telemetry/summary
        try:
            resp = await client.get(f"{API_BASE}/telemetry/summary")
            if resp.status_code == 200:
                data = resp.json()
                if "engine_state" in data:
                    print_test("/telemetry/summary", "PASS", f"Engine: {data.get('engine_state')}")
                else:
                    print_test("/telemetry/summary", "FAIL", "Missing engine_state field")
            else:
                print_test("/telemetry/summary", "FAIL", f"Status code: {resp.status_code}")
        except Exception as e:
            print_test("/telemetry/summary", "FAIL", str(e))

        # Test 3: /trading/why (critical for small capital mode)
        try:
            resp = await client.get(f"{API_BASE}/trading/why")
            if resp.status_code == 200:
                data = resp.json()

                # Check for small_capital_mode field
                if "small_capital_mode" not in data:
                    print_test("/trading/why", "FAIL", "Missing small_capital_mode field")
                    return

                scm = data["small_capital_mode"]

                # Verify structure
                required_fields = [
                    "enabled",
                    "avg_cost_bps",
                    "avg_edge_bps",
                    "edge_cost_ratio",
                    "top_block_reasons",
                    "decision_counts",
                ]

                missing = [f for f in required_fields if f not in scm]
                if missing:
                    print_test(
                        "/trading/why",
                        "FAIL",
                        f"Missing fields: {', '.join(missing)}",
                    )
                else:
                    enabled = scm.get("enabled", False)
                    avg_cost = scm.get("avg_cost_bps")
                    avg_edge = scm.get("avg_edge_bps")
                    ratio = scm.get("edge_cost_ratio")

                    msg = f"Enabled={enabled}"
                    if avg_cost is not None and avg_edge is not None:
                        msg += f", Edge={avg_edge:.1f} bps, Cost={avg_cost:.1f} bps"
                    if ratio is not None:
                        msg += f", Ratio={ratio:.2f}x"

                    print_test("/trading/why", "PASS", msg)

                    # Show decision counts
                    dc = scm.get("decision_counts", {})
                    if dc:
                        print_test(
                            "  Decision Counts",
                            "INFO",
                            f"CREATE={dc.get('create', 0)}, SKIP={dc.get('skip', 0)}, BLOCK={dc.get('block', 0)}",
                        )

                    # Show top block reasons
                    reasons = scm.get("top_block_reasons", [])
                    if reasons:
                        top_3 = reasons[:3]
                        reason_str = ", ".join(
                            [f"{r['reason']}({r['count']})" for r in top_3]
                        )
                        print_test("  Top Block Reasons", "INFO", reason_str)

            else:
                print_test("/trading/why", "FAIL", f"Status code: {resp.status_code}")
        except Exception as e:
            print_test("/trading/why", "FAIL", str(e))


async def test_websocket():
    """Test WebSocket connection and subscription."""
    print_section("WebSocket Tests")

    try:
        # Connect to WebSocket
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            print_test("WS Connection", "PASS", "Connected successfully")

            # Subscribe to system_heartbeat
            subscribe_msg = {
                "type": "subscribe",
                "topic": "system_heartbeat",
            }
            await ws.send(json.dumps(subscribe_msg))

            # Wait for subscription confirmation or heartbeat
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get("type") in ["subscribed", "system_heartbeat"]:
                    print_test(
                        "WS Subscribe",
                        "PASS",
                        f"Topic: system_heartbeat, Type: {data.get('type')}",
                    )
                else:
                    print_test("WS Subscribe", "SKIP", f"Unexpected message: {data.get('type')}")
            except asyncio.TimeoutError:
                print_test("WS Subscribe", "SKIP", "No heartbeat received (engine may be stopped)")

    except Exception as e:
        print_test("WS Connection", "FAIL", str(e))


async def test_module_imports():
    """Test that all new modules can be imported."""
    print_section("Module Import Tests")

    modules_to_test = [
        "hean.execution.cost_engine",
        "hean.execution.market_filters",
        "hean.execution.trade_gating",
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print_test(f"Import {module_name}", "PASS")
        except ImportError as e:
            print_test(f"Import {module_name}", "FAIL", str(e))
        except Exception as e:
            print_test(f"Import {module_name}", "FAIL", f"Unexpected error: {e}")


async def test_config_loaded():
    """Test that small capital mode config is loaded."""
    print_section("Configuration Tests")

    try:
        from hean.config import settings

        # Check small capital mode settings
        checks = [
            ("small_capital_mode", settings.small_capital_mode, True),
            ("min_notional_usd", settings.min_notional_usd, 10.0),
            ("cost_edge_multiplier", settings.cost_edge_multiplier, 4.0),
            ("max_spread_bps", settings.max_spread_bps, 8.0),
        ]

        for name, actual, expected in checks:
            if actual == expected:
                print_test(f"Config: {name}", "PASS", f"= {actual}")
            else:
                print_test(
                    f"Config: {name}",
                    "FAIL",
                    f"Expected {expected}, got {actual}",
                )

    except Exception as e:
        print_test("Config Load", "FAIL", str(e))


async def main():
    """Run all smoke tests."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"Small Capital Profit Mode - Smoke Test")
    print(f"{'=' * 60}{Colors.RESET}\n")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"API Base: {API_BASE}")
    print(f"WS URL: {WS_URL}\n")

    # Run tests
    await test_module_imports()
    await test_config_loaded()
    await test_rest_endpoints()
    await test_websocket()

    # Summary
    print_section("Summary")
    print(f"{Colors.GREEN}{Colors.BOLD}Smoke tests completed!{Colors.RESET}")
    print(f"\nIf any tests failed, check:")
    print(f"  1. Docker containers are running: docker ps")
    print(f"  2. API logs: docker logs hean-api")
    print(f"  3. Environment variables in backend.env")
    print(f"  4. Config loaded correctly in /telemetry/summary\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.RESET}")
        sys.exit(1)
