"""Changelog router for AI Catalyst."""

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/changelog/today")
async def get_today_changelog() -> dict[str, Any]:
    """Get today's changelog (improvements, changes, agent activity).

    Returns changelog from:
    1. Git log --since=today (if available)
    2. changelog_today.json file (if exists)
    3. Fallback: empty changelog
    """
    changelog_items: list[dict[str, Any]] = []

    # Try git log first
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "git",
                "log",
                "--since=today",
                "--pretty=format:%H|%an|%ae|%at|%s",
                "--no-merges",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commit_hash, author_name, author_email, timestamp, message = parts
                    changelog_items.append({
                        "type": "git_commit",
                        "commit_hash": commit_hash[:8],
                        "author": author_name,
                        "timestamp": datetime.fromtimestamp(int(timestamp)).isoformat(),
                        "message": message,
                        "category": "code_change",
                    })
    except Exception as e:
        logger.debug(f"Git log not available: {e}")

    # Try changelog_today.json file
    changelog_file = Path(__file__).parent.parent.parent.parent / "changelog_today.json"
    if changelog_file.exists():
        try:
            with open(changelog_file, "r") as f:
                file_changelog = json.load(f)
                if isinstance(file_changelog, list):
                    changelog_items.extend(file_changelog)
        except Exception as e:
            logger.debug(f"Failed to read changelog_today.json: {e}")

    # If no changelog items, return a placeholder
    if not changelog_items:
        changelog_items.append({
            "type": "system_info",
            "message": "No changes recorded today",
            "timestamp": datetime.utcnow().isoformat(),
            "category": "info",
        })

    return {
        "status": "ok",
        "date": datetime.utcnow().date().isoformat(),
        "items_count": len(changelog_items),
        "items": changelog_items,
    }


@router.get("/agents")
async def get_agents() -> dict[str, Any]:
    """Get list of active agents (from agent registry).

    Returns list of agents and their current status.
    """
    # Try to get agent registry from app state
    try:
        # Import here to avoid circular dependency
        from hean.api.state import app_state

        if hasattr(app_state, "agent_registry") and app_state.agent_registry:
            agents = app_state.agent_registry.get_agents()
            return {
                "status": "ok",
                "agents_count": len(agents),
                "agents": agents,
            }
    except Exception as e:
        logger.debug(f"Agent registry not available: {e}")

    # Fallback: return empty list
    return {
        "status": "ok",
        "agents_count": 0,
        "agents": [],
        "note": "Agent registry not initialized",
    }
