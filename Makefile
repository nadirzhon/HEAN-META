# ================================================================================
# HEAN - AI Trading System Makefile (Monorepo Orchestrator)
# ================================================================================

.PHONY: help install test lint run dev docker-build docker-up docker-down docker-logs docker-restart monitoring-up monitoring-down

.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker-compose
DOCKER_COMPOSE_PROD := docker-compose -f docker-compose.production.yml
DOCKER_COMPOSE_DEV := docker-compose --profile dev
DOCKER_COMPOSE_MONITORING := docker-compose -f docker-compose.monitoring.yml

# PYTHONPATH for workspace packages (used by test/run/lint targets)
BACKEND_PYTHONPATH := $(shell echo backend/packages/hean-{core,exchange,portfolio,risk,execution,strategies,physics,intelligence,observability,symbiont,api,app}/src | tr ' ' ':')

help: ## Show this help message
	@echo "HEAN - AI Trading System (Monorepo)"
	@echo "===================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ================================================================================
# Submodule Management
# ================================================================================

submodules-init: ## Initialize and clone all submodules
	git submodule update --init --recursive

submodules-update: ## Update all submodules to latest remote
	git submodule update --remote --merge

# ================================================================================
# Development Setup (Backend)
# ================================================================================

install: ## Install Python dependencies via uv
	cd backend && uv sync --dev

test: ## Run tests
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) pytest

lint: ## Run linter and type checker
	cd backend && ruff check packages/
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) mypy packages/

run: ## Run the system (CLI)
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) python -m hean.main run

# ================================================================================
# Docker Development
# ================================================================================

dev: ## Start development environment with hot reload
	$(DOCKER_COMPOSE_DEV) up

dev-build: ## Build development images
	$(DOCKER_COMPOSE) build

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start containers in background
	docker-compose up -d

docker-down: ## Stop containers
	docker-compose down

docker-logs: ## View container logs
	docker-compose logs -f

docker-restart: ## Restart containers
	docker-compose restart

docker-clean: ## Remove containers and volumes
	docker-compose down -v
	docker system prune -f

docker-run: docker-build docker-up ## Build and run containers
	@echo "System is running. View logs with: make docker-logs"

# ================================================================================
# Production Deployment
# ================================================================================

prod-build: ## Build production images
	$(DOCKER_COMPOSE_PROD) build --pull

prod-up: ## Start production environment
	$(DOCKER_COMPOSE_PROD) up -d

prod-down: ## Stop production environment
	$(DOCKER_COMPOSE_PROD) down

prod-logs: ## Show production logs
	$(DOCKER_COMPOSE_PROD) logs -f

prod-with-monitoring: ## Start production with monitoring
	$(DOCKER_COMPOSE_PROD) --profile monitoring up -d

# ================================================================================
# Monitoring Stack
# ================================================================================

monitoring-up: ## Start monitoring stack
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "Monitoring stack started:"
	@echo "  - Prometheus: http://localhost:9091"
	@echo "  - Grafana: http://localhost:3001 (admin/admin)"

monitoring-down: ## Stop monitoring stack
	docker-compose -f docker-compose.monitoring.yml down

monitoring-logs: ## View monitoring logs
	docker-compose -f docker-compose.monitoring.yml logs -f

# ================================================================================
# Database Operations
# ================================================================================

redis-cli: ## Connect to Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

redis-backup: ## Backup Redis data
	$(DOCKER_COMPOSE) exec redis redis-cli BGSAVE
	docker cp hean-redis:/data/dump.rdb ./backups/redis-$$(date +%Y%m%d-%H%M%S).rdb
	@echo "Backup created in ./backups/"

# ================================================================================
# Kubernetes
# ================================================================================

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f infra/k8s/

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f infra/k8s/

k8s-status: ## Check Kubernetes status
	kubectl get pods,svc,ingress -n hean-production

# ================================================================================
# Frontend Development
# ================================================================================

dashboard-dev: ## Start dashboard dev server
	cd dashboard && npm install && npm run dev

dashboard-build: ## Build dashboard for production
	cd dashboard && npm run build

ios-build: ## Build iOS app for simulator
	xcodebuild -project ios/HEAN.xcodeproj -scheme HEAN \
		-destination 'platform=iOS Simulator,name=iPhone 17 Pro' -quiet build

# ================================================================================
# Security & Audit
# ================================================================================

scan: ## Scan images for vulnerabilities
	trivy image hean-api:latest

# ================================================================================
# Utilities
# ================================================================================

shell-api: ## Open shell in API container
	$(DOCKER_COMPOSE) exec api bash

shell-redis: ## Open shell in Redis container
	$(DOCKER_COMPOSE) exec redis sh

stats: ## Show container resource usage
	docker stats

# Quick test (skip tests needing Bybit API keys)
test-quick: ## Run tests excluding Bybit connection tests
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) pytest tests/ --ignore=tests/test_bybit_http.py --ignore=tests/test_bybit_websocket.py -q

smoke: ## Run smoke test against running API
	bash backend/scripts/smoke_test.sh

# Test commands
test-truth-layer:
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) pytest tests/test_truth_layer_invariants.py -v

test-selector:
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) pytest tests/test_selector_anti_overfitting.py -v

test-openai:
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) pytest tests/test_openai_factory_hardening.py -v

test-idempotency:
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) pytest tests/test_idempotency_resilience.py -v

# Report/evaluate commands
report:
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) python -m hean.main process report

evaluate:
	cd backend && PYTHONPATH=$(BACKEND_PYTHONPATH) python -m hean.main evaluate --days 30
