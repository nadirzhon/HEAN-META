.PHONY: install test lint run dev dev-up dev-down dev-logs docker-build docker-up docker-down docker-logs docker-restart

# Install dependencies
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest

# Lint and type check
lint:
	ruff check src/
	mypy src/

# Run the system (CLI)
run:
	python -m hean.main run

# Development: start with hot-reload (API + UI)
dev:
	./start-dev.sh

# Alternative: start dev environment with docker compose directly
dev-up:
	docker compose --profile dev up --build
	@echo "🚀 Development environment started with live reload:"
	@echo "  🔹 API:  http://localhost:8000 (auto-reload on .py changes)"
	@echo "  🔹 UI:   http://localhost:5173 (HMR on React changes)"
	@echo "  🔹 Docs: http://localhost:8000/docs"
	@echo ""
	@echo "📝 Edit files in ./src or ./apps/ui/src to see instant changes!"

# Stop dev environment
dev-down:
	docker compose --profile dev down
	@echo "✅ Development environment stopped"

# Show dev logs
dev-logs:
	docker compose --profile dev logs -f

# Show only API dev logs
dev-logs-api:
	docker compose --profile dev logs -f api-dev

# Show only UI dev logs
dev-logs-ui:
	docker compose --profile dev logs -f ui-dev

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

docker-clean:
	docker-compose down -v
	docker system prune -f

# Combined: build and run
docker-run: docker-build docker-up
	@echo "System is running. View logs with: make docker-logs"

# Monitoring stack
monitoring-up:
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "Monitoring stack started:"
	@echo "  - Prometheus: http://localhost:9091"
	@echo "  - Grafana: http://localhost:3001 (admin/admin)"

monitoring-down:
	docker-compose -f docker-compose.monitoring.yml down

monitoring-logs:
	docker-compose -f docker-compose.monitoring.yml logs -f

# Test commands
test-truth-layer:
	pytest tests/test_truth_layer_invariants.py -v

test-selector:
	pytest tests/test_selector_anti_overfitting.py -v

test-openai:
	pytest tests/test_openai_factory_hardening.py -v

test-idempotency:
	pytest tests/test_idempotency_resilience.py -v

# Report/evaluate commands
report:
	python -m hean.main process report

evaluate:
	python -m hean.main evaluate --days 30
