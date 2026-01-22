#!/usr/bin/env python3
"""Comprehensive smoke test for all HEAN system components.

Tests all endpoints, strategies, agent generation, and self-improvement.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Any

import httpx


API_BASE = "http://localhost:8000"
TIMEOUT = 30.0


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(message: str, status: str = "info") -> None:
    """Print colored status message."""
    color = {
        "success": Colors.GREEN,
        "error": Colors.RED,
        "warning": Colors.YELLOW,
        "info": Colors.BLUE,
    }.get(status, Colors.RESET)

    symbol = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ",
    }.get(status, "•")

    print(f"{color}{symbol} {message}{Colors.RESET}")


async def test_health(client: httpx.AsyncClient) -> bool:
    """Test health endpoint."""
    print_status("Testing /health endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_status(f"Health check passed: {data.get('status', 'unknown')}", "success")
            return True
        else:
            print_status(f"Health check failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Health check error: {e}", "error")
        return False


async def test_engine_status(client: httpx.AsyncClient) -> bool:
    """Test engine status endpoint."""
    print_status("Testing /engine/status endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/engine/status", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_status(f"Engine status: {data.get('status', 'unknown')}", "success")
            return True
        else:
            print_status(f"Engine status failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Engine status error: {e}", "error")
        return False


async def test_start_engine(client: httpx.AsyncClient) -> bool:
    """Test engine start."""
    print_status("Testing /engine/start endpoint...", "info")
    try:
        response = await client.post(
            f"{API_BASE}/engine/start",
            json={"confirm_phrase": None},
            timeout=TIMEOUT,
        )
        if response.status_code in (200, 201):
            data = response.json()
            print_status(f"Engine started: {data.get('status', 'unknown')}", "success")
            # Wait a bit for engine to initialize
            await asyncio.sleep(3)
            return True
        else:
            print_status(f"Engine start failed: {response.status_code} - {response.text}", "error")
            return False
    except Exception as e:
        print_status(f"Engine start error: {e}", "error")
        return False


async def test_positions(client: httpx.AsyncClient) -> bool:
    """Test positions endpoint."""
    print_status("Testing /orders/positions endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/orders/positions", timeout=TIMEOUT)
        if response.status_code == 200:
            positions = response.json()
            print_status(f"Positions retrieved: {len(positions)} positions", "success")
            return True
        else:
            print_status(f"Positions failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Positions error: {e}", "error")
        return False


async def test_orders(client: httpx.AsyncClient) -> bool:
    """Test orders endpoint."""
    print_status("Testing /orders endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/orders", timeout=TIMEOUT)
        if response.status_code == 200:
            orders = response.json()
            print_status(f"Orders retrieved: {len(orders)} orders", "success")
            return True
        else:
            print_status(f"Orders failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Orders error: {e}", "error")
        return False


async def test_test_order_buy(client: httpx.AsyncClient) -> bool:
    """Test placing a test BUY order."""
    print_status("Testing /orders/test (BUY) endpoint...", "info")
    try:
        response = await client.post(
            f"{API_BASE}/orders/test",
            json={
                "symbol": "BTCUSDT",
                "side": "buy",
                "size": 0.001,
                "price": None,  # Will be resolved from cache
            },
            timeout=TIMEOUT,
        )
        if response.status_code in (200, 201):
            data = response.json()
            print_status(f"Test BUY order placed: {data.get('message', 'unknown')}", "success")
            await asyncio.sleep(2)  # Wait for order processing
            return True
        else:
            print_status(
                f"Test BUY order failed: {response.status_code} - {response.text}", "error"
            )
            return False
    except Exception as e:
        print_status(f"Test BUY order error: {e}", "error")
        return False


async def test_test_order_sell(client: httpx.AsyncClient) -> bool:
    """Test placing a test SELL order."""
    print_status("Testing /orders/test (SELL) endpoint...", "info")
    try:
        # Wait a bit to ensure price cache is populated after BUY test
        await asyncio.sleep(1)

        # Use BTCUSDT with explicit price to ensure it works
        # First get current price from engine status
        status_resp = await client.get(f"{API_BASE}/engine/status", timeout=TIMEOUT)
        price = 50000.0  # Default fallback
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            # Try to extract price from status if available
            if "current_prices" in status_data:
                price = status_data["current_prices"].get("BTCUSDT", price)

        response = await client.post(
            f"{API_BASE}/orders/test",
            json={
                "symbol": "BTCUSDT",
                "side": "sell",
                "size": 0.001,
                "price": price,  # Use explicit price
            },
            timeout=TIMEOUT,
        )
        if response.status_code in (200, 201):
            data = response.json()
            print_status(f"Test SELL order placed: {data.get('message', 'unknown')}", "success")
            await asyncio.sleep(2)
            return True
        else:
            print_status(
                f"Test SELL order failed: {response.status_code} - {response.text}", "error"
            )
            return False
    except Exception as e:
        print_status(f"Test SELL order error: {e}", "error")
        return False


async def test_test_roundtrip(client: httpx.AsyncClient) -> bool:
    """Test roundtrip order."""
    print_status("Testing /orders/test_roundtrip endpoint...", "info")
    try:
        response = await client.post(
            f"{API_BASE}/orders/test_roundtrip",
            json={
                "symbol": "BTCUSDT",
                "side": "buy",
                "size": 0.001,
                "take_profit_pct": 0.3,
                "stop_loss_pct": 0.3,
                "hold_timeout_sec": 10,
            },
            timeout=15.0,
        )
        if response.status_code in (200, 201):
            data = response.json()
            print_status(f"Roundtrip test completed: {data.get('status', 'unknown')}", "success")
            return True
        else:
            print_status(
                f"Roundtrip test failed: {response.status_code} - {response.text}", "error"
            )
            return False
    except Exception as e:
        print_status(f"Roundtrip test error: {e}", "error")
        return False


async def test_strategies(client: httpx.AsyncClient) -> bool:
    """Test strategies endpoint."""
    print_status("Testing /strategies endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/strategies", timeout=TIMEOUT)
        if response.status_code == 200:
            strategies = response.json()
            print_status(f"Strategies retrieved: {len(strategies)} strategies", "success")
            for strategy in strategies:
                print_status(
                    f"  - {strategy.get('id', 'unknown')}: {strategy.get('enabled', False)}", "info"
                )
            return True
        else:
            print_status(f"Strategies failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Strategies error: {e}", "error")
        return False


async def test_risk_status(client: httpx.AsyncClient) -> bool:
    """Test risk status endpoint."""
    print_status("Testing /risk/status endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/risk/status", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_status(
                f"Risk status retrieved: killswitch={data.get('killswitch_triggered', False)}",
                "success",
            )
            return True
        else:
            print_status(f"Risk status failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Risk status error: {e}", "error")
        return False


async def test_trading_metrics(client: httpx.AsyncClient) -> bool:
    """Test trading metrics endpoint."""
    print_status("Testing /trading/metrics endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/trading/metrics", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_status(f"Trading metrics retrieved: {data.get('status', 'unknown')}", "success")
            return True
        else:
            print_status(f"Trading metrics failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Trading metrics error: {e}", "error")
        return False


async def test_why_not_trading(client: httpx.AsyncClient) -> bool:
    """Test why not trading endpoint."""
    print_status("Testing /trading/why endpoint...", "info")
    try:
        response = await client.get(f"{API_BASE}/trading/why", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_status(
                f"Why not trading: engine_running={data.get('engine_running', False)}", "success"
            )
            return True
        else:
            print_status(f"Why not trading failed: {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Why not trading error: {e}", "error")
        return False


async def test_agent_generation() -> bool:
    """Test agent generation with Gemini."""
    print_status("Testing agent generation with Gemini API...", "info")
    try:
        import os
        from hean.agent_generation.generator import AgentGenerator

        # Ensure API key is loaded from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Try to load from backend.env
            try:
                with open("backend.env") as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            os.environ["GEMINI_API_KEY"] = api_key
                            break
            except Exception:
                pass

        generator = AgentGenerator()
        if generator.llm_client is None:
            print_status("No LLM client configured (Gemini API key may be missing)", "warning")
            return False

        # Check if it's Gemini
        client_type = type(generator.llm_client).__name__
        client_module = type(generator.llm_client).__module__
        is_gemini = "generativeai" in client_module or "genai" in str(client_module)

        if is_gemini:
            print_status(f"Gemini client initialized successfully ({client_type})", "success")
        else:
            print_status(f"LLM client initialized: {client_type} (not Gemini)", "info")

        return True
    except Exception as e:
        print_status(f"Agent generation error: {e}", "error")
        return False


async def test_catalyst_system() -> bool:
    """Test catalyst system initialization."""
    print_status("Testing catalyst system...", "info")
    try:
        from hean.agent_generation.catalyst import ImprovementCatalyst
        from hean.portfolio.accounting import PortfolioAccounting

        accounting = PortfolioAccounting(300.0)
        catalyst = ImprovementCatalyst(
            accounting=accounting,
            strategies={},
            check_interval_minutes=30,
        )
        print_status("Catalyst system initialized successfully", "success")
        return True
    except Exception as e:
        print_status(f"Catalyst system error: {e}", "error")
        return False


async def main() -> None:
    """Run comprehensive smoke tests."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}")
    print("HEAN Comprehensive Smoke Test")
    print(f"{'=' * 60}{Colors.RESET}\n")

    results: dict[str, bool] = {}

    async with httpx.AsyncClient() as client:
        # Basic health checks
        results["health"] = await test_health(client)
        results["engine_status"] = await test_engine_status(client)

        # Engine control
        results["start_engine"] = await test_start_engine(client)

        # Trading endpoints
        results["positions"] = await test_positions(client)
        results["orders"] = await test_orders(client)

        # Test orders (different methods)
        results["test_order_buy"] = await test_test_order_buy(client)
        results["test_order_sell"] = await test_test_order_sell(client)
        results["test_roundtrip"] = await test_test_roundtrip(client)

        # Strategies and risk
        results["strategies"] = await test_strategies(client)
        results["risk_status"] = await test_risk_status(client)
        results["trading_metrics"] = await test_trading_metrics(client)
        results["why_not_trading"] = await test_why_not_trading(client)

        # Agent generation and catalyst
        results["agent_generation"] = await test_agent_generation()
        results["catalyst_system"] = await test_catalyst_system()

    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}{Colors.RESET}\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.RESET} - {test_name}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}\n")

    if passed == total:
        print_status("All tests passed! ✓", "success")
        sys.exit(0)
    else:
        print_status(f"{total - passed} test(s) failed", "error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
