"""System endpoints including AI Catalyst changelog."""

import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

import hean.api.state as state
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
