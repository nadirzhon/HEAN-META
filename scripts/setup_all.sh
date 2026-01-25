#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         HEAN-META Full Stack Setup & Deployment            ║"
echo "║         Технологическая Модернизация 2.0                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

command -v docker >/dev/null 2>&1 || {
    echo -e "${RED}❌ Docker not installed${NC}"
    exit 1
}

command -v docker-compose >/dev/null 2>&1 || {
    echo -e "${RED}❌ Docker Compose not installed${NC}"
    exit 1
}

command -v python3 >/dev/null 2>&1 || {
    echo -e "${RED}❌ Python 3 not installed${NC}"
    exit 1
}

command -v cargo >/dev/null 2>&1 || {
    echo -e "${RED}❌ Rust/Cargo not installed${NC}"
    exit 1
}

echo -e "${GREEN}✅ All prerequisites met${NC}\n"

# Create .env if not exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}📝 Creating .env file...${NC}"
    cp .env.example .env 2>/dev/null || {
        echo -e "${YELLOW}⚠️  .env.example not found, creating minimal .env${NC}"
        cat > .env <<EOF
# HEAN-META Configuration
POSTGRES_PASSWORD=changeme
SECRET_KEY=$(openssl rand -hex 32)
GRAFANA_PASSWORD=admin
EOF
    }
    echo -e "${GREEN}✅ .env created${NC}\n"
fi

# Build C++ Order Engine
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🔨 Building C++ Order Engine...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

cd hean_meta_cpp
./build.sh
cd ..

echo -e "${GREEN}✅ C++ Order Engine built successfully${NC}\n"

# Build Rust Market Data Service
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🦀 Building Rust Market Data Service...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

cd market-data-service
cargo build --release
cd ..

echo -e "${GREEN}✅ Rust service built successfully${NC}\n"

# Build Docker images
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🐳 Building Docker images...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

docker-compose -f docker-compose.full-stack.yml build

echo -e "${GREEN}✅ Docker images built${NC}\n"

# Start services
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🚀 Starting all services...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

docker-compose -f docker-compose.full-stack.yml up -d

echo -e "${GREEN}✅ Services started${NC}\n"

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 15

# Check health
echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🏥 Checking service health...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

services=("api" "market-data" "ui" "keydb" "prometheus" "grafana")
healthy_count=0

for service in "${services[@]}"; do
    container_name="hean-${service}"
    if [ "$service" = "market-data" ]; then
        container_name="hean-meta-market-data"
    elif [ "$service" = "prometheus" ]; then
        container_name="hean-meta-prometheus"
    elif [ "$service" = "grafana" ]; then
        container_name="hean-meta-grafana"
    fi

    if docker ps | grep -q "$container_name"; then
        echo -e "${GREEN}✅ $service is running${NC}"
        ((healthy_count++))
    else
        echo -e "${RED}❌ $service failed to start${NC}"
    fi
done

echo ""

# Display access information
if [ $healthy_count -eq ${#services[@]} ]; then
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              🎉 Setup Complete! 🎉                          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${BLUE}📊 Access Points:${NC}"
    echo ""
    echo -e "  ${GREEN}Frontend Dashboard:${NC}      http://localhost:3000"
    echo -e "  ${GREEN}Backend API:${NC}             http://localhost:8000"
    echo -e "  ${GREEN}API Docs (Swagger):${NC}     http://localhost:8000/docs"
    echo -e "  ${GREEN}Rust Market Data:${NC}       http://localhost:8080"
    echo -e "  ${GREEN}Grafana:${NC}                http://localhost:3001"
    echo -e "     └─ Username: ${YELLOW}admin${NC}"
    echo -e "     └─ Password: ${YELLOW}admin${NC}"
    echo -e "  ${GREEN}Prometheus:${NC}             http://localhost:9090"
    echo -e "  ${GREEN}cAdvisor:${NC}               http://localhost:8081"
    echo ""

    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}View logs:${NC}              docker-compose -f docker-compose.full-stack.yml logs -f [service]"
    echo -e "  ${YELLOW}Stop services:${NC}          docker-compose -f docker-compose.full-stack.yml down"
    echo -e "  ${YELLOW}Restart service:${NC}        docker-compose -f docker-compose.full-stack.yml restart [service]"
    echo -e "  ${YELLOW}Check status:${NC}           docker-compose -f docker-compose.full-stack.yml ps"
    echo ""

    echo -e "${BLUE}📈 Performance Monitoring:${NC}"
    echo ""
    echo -e "  ${YELLOW}KeyDB benchmark:${NC}        python scripts/benchmark_keydb.py"
    echo -e "  ${YELLOW}C++ engine test:${NC}        cd hean_meta_cpp && ./build/test_order_engine"
    echo -e "  ${YELLOW}Python example:${NC}         python hean_meta_cpp/python/example.py"
    echo ""

    echo -e "${BLUE}🔐 Security:${NC}"
    echo ""
    echo -e "  ${YELLOW}⚠️  Change Grafana password:${NC} Set GRAFANA_PASSWORD in .env"
    echo -e "  ${YELLOW}⚠️  Set KeyDB password:${NC}      Edit keydb.conf (requirepass)"
    echo ""

    echo -e "${GREEN}✨ Stack Summary:${NC}"
    echo -e "  ✅ KeyDB (Multi-threaded Redis)"
    echo -e "  ✅ C++ Order Engine (<100μs latency)"
    echo -e "  ✅ Rust Market Data Service"
    echo -e "  ✅ FastAPI Backend"
    echo -e "  ✅ React Frontend"
    echo -e "  ✅ Prometheus + Grafana Monitoring"
    echo ""

    echo -e "${BLUE}📚 Documentation:${NC}"
    echo -e "  📄 KeyDB:           ${YELLOW}KEYDB_MIGRATION_GUIDE.md${NC}"
    echo -e "  📄 C++ Engine:      ${YELLOW}hean_meta_cpp/README.md${NC}"
    echo -e "  📄 Rust Service:    ${YELLOW}market-data-service/README.md${NC}"
    echo ""

else
    echo -e "${RED}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              ⚠️  Setup Incomplete ⚠️                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${YELLOW}Some services failed to start. Check logs:${NC}"
    echo -e "  docker-compose -f docker-compose.full-stack.yml logs"
    echo ""
fi
