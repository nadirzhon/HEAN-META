"""System endpoints including AI Catalyst changelog."""

import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

import hean.api.state as state
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/changelog/today")
async def get_changelog_today(request: Request) -> dict:
    """Get today's changelog from git log or changelog_today.json.

    Returns:
        Dictionary with available flag, entries, and reason if unavailable.
    """
    try:
        # Try git log first
        try:
            result = subprocess.run(
                ["git", "log", "--since=today", "--pretty=format:%h|%s|%an|%ad", "--date=iso"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path(__file__).parent.parent.parent.parent,
            )

            if result.returncode == 0 and result.stdout.strip():
                entries = []
                for line in result.stdout.strip().split("\n"):
                    if "|" in line:
                        parts = line.split("|", 3)
                        if len(parts) >= 4:
                            entries.append({
                                "hash": parts[0],
                                "message": parts[1],
                                "author": parts[2],
                                "date": parts[3],
                            })

                return {
                    "available": True,
                    "source": "git",
                    "entries": entries,
                    "count": len(entries),
                }
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug(f"Git log not available: {e}")

        # Fallback to changelog_today.json
        changelog_path = Path(__file__).parent.parent.parent.parent / "changelog_today.json"
        if changelog_path.exists():
            import json
            with open(changelog_path) as f:
                data = json.load(f)
                return {
                    "available": True,
                    "source": "changelog_today.json",
                    "entries": data.get("entries", []),
                    "count": len(data.get("entries", [])),
                }

        # Not available
        return {
            "available": False,
            "reason": "git not available / changelog missing",
            "entries": [],
            "count": 0,
        }
    except Exception as e:
        logger.error(f"Failed to get changelog: {e}", exc_info=True)
        return {
            "available": False,
            "reason": str(e),
            "entries": [],
            "count": 0,
        }


@router.get("/v1/dashboard")
async def get_dashboard(request: Request) -> dict[str, Any]:
    """Return a lightweight dashboard snapshot for legacy UI polling."""
    engine_facade = state.get_engine_facade(request)
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade is not ready")

    status = await engine_facade.get_status()
    snapshot = await engine_facade.get_trading_state()
    account_state = snapshot.get("account_state")
    positions = snapshot.get("positions") or []
    orders = snapshot.get("orders") or []

    equity = (
        status.get("equity")
        or (account_state or {}).get("equity")
        or settings.initial_capital
    )
    initial_capital = status.get("initial_capital") or settings.initial_capital or 1
    daily_pnl = status.get("daily_pnl") or 0.0
    return_pct = ((equity - initial_capital) / initial_capital * 100) if initial_capital else 0.0

    return {
        "account_state": account_state,
        "metrics": {
            "equity": equity,
            "daily_pnl": daily_pnl,
            "return_pct": return_pct,
            "open_positions": len(positions),
        },
        "positions": positions,
        "orders": orders,
        "status": {
            "engine_running": bool(status.get("running")),
            "trading_mode": status.get("trading_mode") or settings.trading_mode,
        },
    }


@router.get("/cpp/status")
async def get_cpp_status() -> dict[str, Any]:
    """Get C++ modules status for diagnostics.

    Returns:
        dict with module availability, performance hints, and build instructions
    """
    try:
        from hean.cpp_modules import get_cpp_status as _get_cpp_status

        return _get_cpp_status()
    except Exception as e:
        logger.warning(f"Failed to get C++ status: {e}")
        return {
            "indicators_cpp_available": False,
            "order_router_cpp_available": False,
            "performance_hint": "C++ modules not available - using Python fallback",
            "build_instructions": "Run: ./scripts/build_cpp_modules.sh",
            "error": str(e),
        }
