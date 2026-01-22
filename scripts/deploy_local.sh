#!/usr/bin/env bash
#
# Local deployment script for HEAN project
# Mimics the GitHub Actions workflow for local testing
#
# Usage:
#   ./scripts/deploy_local.sh          # Normal deploy with cache
#   ./scripts/deploy_local.sh --rebuild # Force rebuild without cache
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Parse arguments
REBUILD=false
if [ "$1" = "--rebuild" ] || [ "$1" = "--no-cache" ]; then
    REBUILD=true
    print_warning "Force rebuild enabled (--no-cache)"
fi

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

print_header "HEAN Local Deployment Script"
echo "Project Root: $PROJECT_ROOT"
echo "Rebuild: $REBUILD"
echo "Timestamp: $(date)"
echo ""

# Step 1: System diagnostics
print_header "Step 1: System Diagnostics"

echo "System: $(uname -a)"
echo ""

if ! command -v docker &> /dev/null; then
    print_error "Docker not found! Please install Docker Desktop."
    exit 1
else
    print_success "Docker found"
    docker version | head -5
fi

echo ""

if ! docker compose version &> /dev/null; then
    print_error "Docker Compose not found! Please install Docker Compose."
    exit 1
else
    print_success "Docker Compose found"
    docker compose version
fi

echo ""

# Step 2: Check docker-compose.yml
print_header "Step 2: Checking Docker Compose File"

if [ -f "docker-compose.yml" ]; then
    print_success "Found docker-compose.yml"
elif [ -f "compose.yaml" ]; then
    print_success "Found compose.yaml"
else
    print_error "No docker-compose.yml or compose.yaml found!"
    exit 1
fi

echo ""

# Step 3: Stop existing containers
print_header "Step 3: Stopping Existing Containers"

if docker compose ps | grep -q "Up"; then
    print_info "Stopping running containers..."
    docker compose down || true
    print_success "Containers stopped"
else
    print_info "No running containers to stop"
fi

echo ""

# Step 4: Deploy
print_header "Step 4: Deploying with Docker Compose"

if [ "$REBUILD" = true ]; then
    print_info "Building images without cache..."
    docker compose build --no-cache
    print_info "Starting containers..."
    docker compose up -d --remove-orphans
else
    print_info "Building and starting containers (with cache)..."
    docker compose up -d --build --remove-orphans
fi

print_success "Deployment command completed"
echo ""

# Step 5: Wait and verify
print_header "Step 5: Verifying Deployment"

print_info "Waiting 5 seconds for containers to start..."
sleep 5

echo ""
echo "Container Status:"
docker compose ps

echo ""
echo "Running Containers:"
docker ps --filter "name=hean" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""

# Step 6: Health checks
print_header "Step 6: Health Checks"

print_info "Waiting 10 seconds for services to initialize..."
sleep 10

# Check API
if curl -f -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
    print_success "API health check PASSED (http://localhost:8000/health)"
else
    print_warning "API health check FAILED (http://localhost:8000/health)"
    print_info "API may still be starting up. Check logs below."
fi

# Check UI
if curl -f -s --max-time 5 http://localhost:3000 > /dev/null 2>&1; then
    print_success "UI health check PASSED (http://localhost:3000)"
else
    print_warning "UI health check FAILED (http://localhost:3000)"
    print_info "UI may still be starting up. Check logs below."
fi

echo ""

# Step 7: Show logs
print_header "Step 7: Recent Container Logs"

docker compose logs --tail=50

echo ""

# Step 8: Cleanup
print_header "Step 8: Cleanup"

print_info "Removing dangling images..."
docker image prune -f || true

echo ""
echo "Docker Disk Usage:"
docker system df

echo ""

# Step 9: Summary
print_header "Deployment Summary"

echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
echo "Timestamp: $(date)"
echo ""

echo "Final Container Status:"
docker compose ps

echo ""
print_success "Deployment completed successfully!"
echo ""

print_info "Access the application:"
echo "  API:  http://localhost:8000"
echo "  UI:   http://localhost:3000"
echo "  Docs: http://localhost:8000/docs"
echo ""

print_info "Useful commands:"
echo "  View logs:    docker compose logs -f"
echo "  Stop:         docker compose down"
echo "  Restart:      docker compose restart"
echo "  Shell (API):  docker compose exec api bash"
echo "  Shell (UI):   docker compose exec hean-ui sh"
echo ""

print_success "Done! 🚀"
