#!/bin/sh
# ============================================================================
# HEAN Symbiont-X Healthcheck Script
# ============================================================================
# Purpose: Verify trading process is running and responsive
# Usage: Called by Docker healthcheck (Dockerfile.testnet)
# Exit codes: 0 = healthy, 1 = unhealthy
# ============================================================================

# Check if trading process is running
MAIN_PID=$(pgrep -f "python.*live_testnet" | head -1)
if [ -z "$MAIN_PID" ]; then
    echo "UNHEALTHY: Trading process not running"
    exit 1
fi

# Check heartbeat file age (POSIX-compatible - works on Alpine/BusyBox)
if [ -f /app/logs/heartbeat.txt ]; then
    # Use POSIX-compatible date command (works on Alpine)
    LAST_HEARTBEAT=$(date -r /app/logs/heartbeat.txt +%s 2>/dev/null || echo 0)
    NOW=$(date +%s)
    AGE=$((NOW - LAST_HEARTBEAT))

    # Heartbeat must be updated within last 5 minutes (300 seconds)
    if [ $AGE -gt 300 ]; then
        echo "UNHEALTHY: Heartbeat stale ($AGE seconds old, max 300)"
        exit 1
    fi

    echo "HEALTHY: Trading process running (PID: $MAIN_PID, heartbeat: ${AGE}s ago)"
    exit 0
fi

# If no heartbeat file exists yet (startup phase), check if process exists
echo "HEALTHY: Trading process running (PID: $MAIN_PID, no heartbeat file yet)"
exit 0
