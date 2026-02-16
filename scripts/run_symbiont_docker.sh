#!/bin/bash
# HEAN SYMBIONT X - Docker Launch Script
# Автоматический запуск SYMBIONT X через Docker

set -e

echo "🧬 =========================================="
echo "   HEAN SYMBIONT X - Docker Launcher"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Menu
echo "Select mode:"
echo "  1) Full System Demo (offline simulation)"
echo "  2) Bybit Testnet Live Trading"
echo "  3) Build Docker images"
echo "  4) View logs"
echo "  5) Stop all containers"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}🚀 Starting Full System Demo...${NC}"
        echo ""
        docker compose up symbiont-demo
        ;;
    2)
        echo ""
        echo -e "${BLUE}📡 Starting Bybit Testnet Live Trading...${NC}"
        echo -e "${YELLOW}⚠️  Make sure .env.symbiont contains valid API keys${NC}"
        echo ""
        docker compose up -d symbiont-testnet
        echo ""
        echo -e "${GREEN}✅ Container started in background${NC}"
        echo "View logs with: docker compose logs -f symbiont-testnet"
        echo "Stop with: docker compose stop symbiont-testnet"
        ;;
    3)
        echo ""
        echo -e "${BLUE}🔨 Building Docker images...${NC}"
        echo ""
        docker compose build symbiont-demo symbiont-testnet
        echo ""
        echo -e "${GREEN}✅ Build complete${NC}"
        ;;
    4)
        echo ""
        echo "Select logs to view:"
        echo "  1) Full System Demo"
        echo "  2) Bybit Testnet"
        echo ""
        read -p "Enter choice [1-2]: " log_choice

        case $log_choice in
            1)
                docker compose logs -f symbiont-demo
                ;;
            2)
                docker compose logs -f symbiont-testnet
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
        ;;
    5)
        echo ""
        echo -e "${YELLOW}🛑 Stopping all SYMBIONT containers...${NC}"
        docker compose stop symbiont-demo symbiont-testnet
        echo ""
        echo -e "${GREEN}✅ Containers stopped${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Done!${NC}"
