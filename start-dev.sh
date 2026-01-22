#!/bin/bash
# Start development environment with hot-reload for both API and UI

echo "🚀 Starting HEAN in development mode with live reload..."
echo ""
echo "Features enabled:"
echo "  ✅ API hot-reload (uvicorn --reload)"
echo "  ✅ UI hot-reload (Vite HMR)"
echo "  ✅ Volume mounting for instant code changes"
echo ""
echo "URLs:"
echo "  🔹 API:  http://localhost:8000"
echo "  🔹 UI:   http://localhost:5173"
echo "  🔹 Docs: http://localhost:8000/docs"
echo ""

# Stop any running containers
docker compose down

# Start dev environment
docker compose --profile dev up --build

# Cleanup on exit
trap "docker compose --profile dev down" EXIT
