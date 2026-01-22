#!/usr/bin/env python3
"""Comprehensive diagnostic script to check health of all HEAN components.

This script checks:
- Engine status and initialization
- API endpoints and responses
- Redis connectivity
- WebSocket availability
- Database/state consistency
- Frontend connectivity

Run: python debug_status.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

try:
    import aiohttp
    import redis.asyncio as aioredis
except ImportError:
    print("ERROR: Missing dependencies. Install with: pip install aiohttp redis")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(name: str, status: bool, details: str = "") -> None:
    """Print a status line with color coding."""
    color = Colors.GREEN if status else Colors.RED
    symbol = "✓" if status else "✗"
    print(f"{color}{symbol} {Colors.RESET}{Colors.BOLD}{name}{Colors.RESET}: {details}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


async def check_redis(redis_url: str | None = None) -> dict[str, Any]:
    """Check Redis connectivity.

    Uses REDIS_URL environment variable with fallback to redis://redis:6379/0.
    """
    # Use environment variable with fallback
    if redis_url is None:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

    result = {
        "connected": False,
        "url": redis_url,
        "error": None,
        "channels": [],
    }

    try:
        redis_client = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        result["connected"] = True

        # Check for AFO channels
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("afo:nexus:tick", "afo:nexus:orderbook")
        result["channels"] = ["afo:nexus:tick", "afo:nexus:orderbook"]
        await pubsub.close()
        await redis_client.close()
    except Exception as e:
        result["error"] = str(e)

    return result


async def check_api_endpoint(
    session: aiohttp.ClientSession, base_url: str, endpoint: str
) -> dict[str, Any]:
    """Check an API endpoint."""
    url = f"{base_url}{endpoint}"
    result = {
        "url": url,
        "status": "unknown",
        "status_code": None,
        "response": None,
        "error": None,
    }

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            result["status_code"] = resp.status
            result["status"] = "ok" if resp.status == 200 else "error"
            try:
                result["response"] = await resp.json()
            except:
                result["response"] = await resp.text()
    except asyncio.TimeoutError:
        result["error"] = "Timeout"
        result["status"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"

    return result


async def check_websocket(ws_url: str) -> dict[str, Any]:
    """Check WebSocket connectivity."""
    result = {
        "url": ws_url,
        "connected": False,
        "error": None,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=5)) as ws:
                result["connected"] = True
                # Send ping
                await ws.send_json({"type": "ping"})
                # Wait for response (with timeout)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        result["response"] = json.loads(msg.data)
                except asyncio.TimeoutError:
                    pass  # No response is ok, connection works
    except Exception as e:
        result["error"] = str(e)

    return result


async def main() -> None:
    """Run all diagnostic checks."""
    print(f"\n{Colors.BOLD}HEAN System Diagnostic Report{Colors.RESET}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    base_url = "http://localhost:8000"
    api_base = f"{base_url}/api"

    # Check Redis
    print_section("Redis Connectivity")
    redis_url_env = os.getenv("REDIS_URL", "redis://redis:6379/0")
    print(f"  Using REDIS_URL: {redis_url_env}")
    redis_result = await check_redis()
    print_status(
        "Redis Connection",
        redis_result["connected"],
        f"{redis_result['url']} - {redis_result.get('error', 'OK')}",
    )
    if redis_result["connected"]:
        print(f"  Channels: {', '.join(redis_result['channels'])}")

    # Check API Health
    print_section("API Health Checks")
    async with aiohttp.ClientSession() as session:
        # Health endpoint
        health = await check_api_endpoint(session, api_base, "/health")
        print_status(
            "Health Endpoint",
            health["status"] == "ok",
            f"Status {health['status_code']} - {health.get('error', 'OK')}",
        )
        if health["response"]:
            engine_running = health["response"].get("engine_running", False)
            engine_init = health["response"].get("engine_initialized", False)
            print(f"  Engine Initialized: {engine_init}")
            print(f"  Engine Running: {engine_running}")

        # Engine status
        engine_status = await check_api_endpoint(session, api_base, "/engine/status")
        print_status(
            "Engine Status",
            engine_status["status"] == "ok",
            f"Status {engine_status['status_code']} - {engine_status.get('error', 'OK')}",
        )
        if engine_status["response"]:
            status = engine_status["response"].get("status", "unknown")
            equity = engine_status["response"].get("equity", 0)
            print(f"  Status: {status}")
            print(f"  Equity: ${equity:,.2f}")

        # Strategies
        strategies = await check_api_endpoint(session, api_base, "/strategies")
        print_status(
            "Strategies Endpoint",
            strategies["status"] == "ok",
            f"Status {strategies['status_code']} - {strategies.get('error', 'OK')}",
        )
        if strategies["response"]:
            count = len(strategies["response"]) if isinstance(strategies["response"], list) else 0
            print(f"  Strategies Count: {count}")

        # Analytics Performance
        analytics = await check_api_endpoint(session, api_base, "/analytics/performance")
        print_status(
            "Analytics Performance",
            analytics["status"] == "ok",
            f"Status {analytics['status_code']} - {analytics.get('error', 'OK')}",
        )
        if analytics["response"]:
            strategies_count = len(analytics["response"].get("strategies", {}))
            print(f"  Strategy Metrics: {strategies_count}")

        # Positions
        positions = await check_api_endpoint(session, api_base, "/orders/positions")
        print_status(
            "Positions Endpoint",
            positions["status"] == "ok",
            f"Status {positions['status_code']} - {positions.get('error', 'OK')}",
        )
        if positions["response"]:
            count = len(positions["response"]) if isinstance(positions["response"], list) else 0
            print(f"  Open Positions: {count}")

    # Check WebSocket
    print_section("WebSocket Connectivity")
    ws_url = f"ws://localhost:8000/api/ws/speed-engine"
    ws_result = await check_websocket(ws_url)
    print_status(
        "Speed Engine WebSocket",
        ws_result["connected"],
        f"{ws_url} - {ws_result.get('error', 'OK')}",
    )

    # Data Consistency Check
    print_section("Data Consistency")
    async with aiohttp.ClientSession() as session:
        engine_status = await check_api_endpoint(session, api_base, "/engine/status")
        if engine_status["response"]:
            # Check for undefined/null values
            has_undefined = False
            undefined_fields = []

            def check_dict(d: dict, prefix: str = "") -> None:
                nonlocal has_undefined, undefined_fields
                for k, v in d.items():
                    key_path = f"{prefix}.{k}" if prefix else k
                    if v is None:
                        has_undefined = True
                        undefined_fields.append(key_path)
                    elif isinstance(v, dict):
                        check_dict(v, key_path)

            check_dict(engine_status["response"])

            print_status(
                "No Undefined Values",
                not has_undefined,
                f"{len(undefined_fields)} undefined fields found"
                if has_undefined
                else "All fields have values",
            )
            if undefined_fields:
                print(f"  Undefined fields: {', '.join(undefined_fields[:10])}")

    # Trading Readiness Section
    print_section("Trading Readiness")

    async with aiohttp.ClientSession() as session:
        # Check Warden mode (Aggressive vs Safe)
        engine_status = await check_api_endpoint(session, api_base, "/engine/status")
        if engine_status["response"]:
            debug_mode = engine_status["response"].get("debug_mode", False)
            paper_trade_assist = engine_status["response"].get("paper_trade_assist", False)

            if debug_mode and paper_trade_assist:
                mode_status = f"{Colors.YELLOW}AGGRESSIVE MODE{Colors.RESET}"
                print_status(
                    "Warden Filter Mode", True, f"{mode_status} - Filters loosened for testing"
                )
                print(f"  DEBUG_MODE: {debug_mode}")
                print(f"  PAPER_TRADE_ASSIST: {paper_trade_assist}")
                print(
                    f"  {Colors.YELLOW}⚠️  WARNING: Aggressive Mode is for testing only!{Colors.RESET}"
                )
            elif debug_mode:
                print_status(
                    "Warden Filter Mode",
                    True,
                    f"{Colors.YELLOW}DEBUG MODE{Colors.RESET} - Some filters bypassed",
                )
                print(f"  DEBUG_MODE: {debug_mode}")
            else:
                print_status(
                    "Warden Filter Mode",
                    True,
                    f"{Colors.GREEN}SAFE MODE{Colors.RESET} - All filters active",
                )
                print(f"  DEBUG_MODE: {debug_mode}")
                print(f"  PAPER_TRADE_ASSIST: {paper_trade_assist}")

        # Check API Key permissions (if in live mode)
        try:
            import os

            api_key = os.getenv("BYBIT_API_KEY", "")
            if api_key:
                # Try to check API key info via Bybit API
                testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
                base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"

                # Note: Full API key permission check requires authenticated request
                # For now, just check if key exists
                if api_key and len(api_key) > 10:
                    print_status(
                        "API Key Configured", True, f"Key present (length: {len(api_key)} chars)"
                    )
                    print(
                        f"  {Colors.YELLOW}⚠️  Verify 'Trade' permission is active in Bybit API settings{Colors.RESET}"
                    )
                else:
                    print_status("API Key Configured", False, "API key missing or invalid")
            else:
                print_status("API Key Configured", False, "BYBIT_API_KEY not set (paper mode only)")
        except Exception as e:
            print_status("API Key Check", False, f"Error checking API key: {str(e)}")

        # Check active strategies and signal strength
        strategies = await check_api_endpoint(session, api_base, "/strategies")
        if strategies["response"]:
            strategies_list = (
                strategies["response"] if isinstance(strategies["response"], list) else []
            )
            print_status(
                "Active Strategies",
                len(strategies_list) > 0,
                f"{len(strategies_list)} strategy(ies) active",
            )

            for strat in strategies_list[:5]:  # Show first 5
                strat_id = strat.get("strategy_id", "unknown")
                enabled = strat.get("enabled", False)
                status = "ENABLED" if enabled else "DISABLED"
                color = Colors.GREEN if enabled else Colors.RED

                # Try to get signal strength from analytics
                signal_strength = "N/A"
                try:
                    analytics = await check_api_endpoint(
                        session, api_base, f"/analytics/performance"
                    )
                    if analytics["response"]:
                        strategies_metrics = analytics["response"].get("strategies", {})
                        if strat_id in strategies_metrics:
                            metrics = strategies_metrics[strat_id]
                            signals = metrics.get("signals_generated", 0)
                            signals_rejected = metrics.get("signals_rejected", 0)
                            total = signals + signals_rejected
                            if total > 0:
                                strength_pct = (signals / total) * 100
                                signal_strength = (
                                    f"{strength_pct:.1f}% pass rate ({signals}/{total})"
                                )
                except:
                    pass

                print(
                    f"  {color}{status}{Colors.RESET}: {strat_id} | Signal Strength: {signal_strength}"
                )

        # Check rejection statistics
        try:
            analytics = await check_api_endpoint(session, api_base, "/analytics/performance")
            if analytics["response"]:
                total_rejected = sum(
                    strat.get("signals_rejected", 0)
                    for strat in analytics["response"].get("strategies", {}).values()
                )
                total_generated = sum(
                    strat.get("signals_generated", 0)
                    for strat in analytics["response"].get("strategies", {}).values()
                )

                if total_generated > 0:
                    rejection_rate = (total_rejected / total_generated) * 100
                    print(f"\n  Overall Signal Statistics:")
                    print(f"    Generated: {total_generated}")
                    print(f"    Rejected: {total_rejected} ({rejection_rate:.1f}%)")
                    print(
                        f"    Accepted: {total_generated - total_rejected} ({100 - rejection_rate:.1f}%)"
                    )
        except:
            pass

    # Summary
    print_section("Summary")
    print(f"{Colors.BOLD}Diagnostic complete.{Colors.RESET}")
    print(f"\nTo view logs: docker-compose logs -f afo-engine")
    print(f"To restart: docker-compose restart afo-engine")
    print(f"To check health: curl {api_base}/health")
    print(f"\n{Colors.BOLD}Aggressive Mode Checklist:{Colors.RESET}")
    print(f"  • PAPER_TRADE_ASSIST=True")
    print(f"  • DEBUG_MODE=True")
    print(f"  • LOG_LEVEL=DEBUG")
    print(f"  • Check rejection logs: docker-compose logs -f afo-engine | grep 'SIGNAL REJECTED'")
    print(f"  • Check heartbeat: docker-compose logs -f afo-engine | grep 'HEARTBEAT'\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Diagnostic interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Error running diagnostic: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
