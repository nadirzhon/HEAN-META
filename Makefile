.PHONY: install test lint run dev docker-build docker-up docker-down docker-logs docker-restart

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

# Development: start API + frontend + monitoring
dev:
	docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d --build
	@echo "Development environment started:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Command Center: http://localhost:3000"
	@echo "  - Prometheus: http://localhost:9091"
	@echo "  - Grafana: http://localhost:3001 (admin/admin)"

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
