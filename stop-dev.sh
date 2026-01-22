#!/bin/bash
# Stop development environment

echo "🛑 Stopping HEAN development environment..."

docker compose --profile dev down

echo "✅ Development environment stopped"
