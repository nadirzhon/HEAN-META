#!/bin/bash
# =============================================================================
# HEAN Bonjour Beacon
# Advertises the HEAN Trading API on the local network via Bonjour/mDNS
# so that the iOS app can auto-discover the backend without manual IP config.
#
# Usage:
#   ./scripts/hean-beacon.sh          # Start beacon (foreground)
#   ./scripts/hean-beacon.sh start    # Start beacon (background)
#   ./scripts/hean-beacon.sh stop     # Stop background beacon
#   ./scripts/hean-beacon.sh status   # Check if beacon is running
# =============================================================================

PORT="${HEAN_API_PORT:-8000}"
PID_FILE="/tmp/hean-beacon.pid"
SERVICE_NAME="HEAN Trading API"
SERVICE_TYPE="_hean-api._tcp"

case "${1:-}" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Beacon already running (PID $(cat "$PID_FILE"))"
            exit 0
        fi
        echo "Starting HEAN Bonjour beacon on port $PORT..."
        nohup dns-sd -R "$SERVICE_NAME" "$SERVICE_TYPE" local "$PORT" > /tmp/hean-beacon.log 2>&1 &
        echo $! > "$PID_FILE"
        echo "Beacon started (PID $!) — iOS app will auto-discover at $PORT"
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                echo "Beacon stopped (PID $PID)"
            else
                echo "Beacon not running (stale PID file)"
            fi
            rm -f "$PID_FILE"
        else
            echo "No beacon PID file found"
        fi
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Beacon running (PID $(cat "$PID_FILE")) on port $PORT"
        else
            echo "Beacon not running"
        fi
        ;;
    *)
        # Foreground mode (Ctrl+C to stop)
        echo "Broadcasting HEAN API via Bonjour: $SERVICE_TYPE on port $PORT"
        echo "iOS app will auto-discover this server. Press Ctrl+C to stop."
        exec dns-sd -R "$SERVICE_NAME" "$SERVICE_TYPE" local "$PORT"
        ;;
esac
