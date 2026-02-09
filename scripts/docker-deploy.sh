#!/bin/bash
# ============================================
# HEAN Docker Deployment Script
# ============================================
set -e

echo "🐳 HEAN Docker Deployment"
echo "=========================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose is not installed.${NC}"
    exit 1
fi

# Parse arguments
ENV=${1:-production}
ACTION=${2:-deploy}

echo -e "${GREEN}Environment: ${ENV}${NC}"
echo -e "${GREEN}Action: ${ACTION}${NC}"

# Environment file check
if [ "$ENV" = "production" ]; then
    COMPOSE_FILE="docker-compose.production.yml"
    if [ ! -f "backend.env" ]; then
        echo -e "${RED}❌ backend.env not found!${NC}"
        echo "Please create backend.env with required configuration."
        exit 1
    fi
else
    COMPOSE_FILE="docker-compose.yml"
fi

case $ACTION in
    deploy)
        echo -e "${YELLOW}🏗️  Building images...${NC}"
        docker-compose -f $COMPOSE_FILE build --pull
        
        echo -e "${YELLOW}🚀 Starting services...${NC}"
        docker-compose -f $COMPOSE_FILE up -d
        
        echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
        sleep 10
        
        # Health check
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is healthy${NC}"
        else
            echo -e "${RED}❌ API health check failed${NC}"
            docker-compose -f $COMPOSE_FILE logs api
            exit 1
        fi
        
        if curl -f http://localhost:3000 > /dev/null 2>&1 || curl -f http://localhost:80 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ UI is healthy${NC}"
        else
            echo -e "${RED}❌ UI health check failed${NC}"
            docker-compose -f $COMPOSE_FILE logs ui
            exit 1
        fi
        
        echo -e "${GREEN}✅ Deployment successful!${NC}"
        echo ""
        echo "Access your application at:"
        if [ "$ENV" = "production" ]; then
            echo "  🌐 UI: http://localhost"
            echo "  🔌 API: http://localhost/api"
        else
            echo "  🌐 UI: http://localhost:3000"
            echo "  🔌 API: http://localhost:8000"
        fi
        ;;
    
    update)
        echo -e "${YELLOW}🔄 Updating services...${NC}"
        docker-compose -f $COMPOSE_FILE pull
        docker-compose -f $COMPOSE_FILE up -d
        echo -e "${GREEN}✅ Update complete${NC}"
        ;;
    
    stop)
        echo -e "${YELLOW}⏹️  Stopping services...${NC}"
        docker-compose -f $COMPOSE_FILE down
        echo -e "${GREEN}✅ Services stopped${NC}"
        ;;
    
    restart)
        echo -e "${YELLOW}🔄 Restarting services...${NC}"
        docker-compose -f $COMPOSE_FILE restart
        echo -e "${GREEN}✅ Services restarted${NC}"
        ;;
    
    logs)
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    
    status)
        docker-compose -f $COMPOSE_FILE ps
        ;;
    
    backup)
        echo -e "${YELLOW}💾 Creating backup...${NC}"
        mkdir -p backups
        docker-compose -f $COMPOSE_FILE exec -T redis redis-cli BGSAVE
        sleep 2
        docker cp hean-redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d-%H%M%S).rdb
        echo -e "${GREEN}✅ Backup created in ./backups/${NC}"
        ;;
    
    clean)
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        docker-compose -f $COMPOSE_FILE down -v
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
    
    *)
        echo "Usage: $0 [production|development] [deploy|update|stop|restart|logs|status|backup|clean]"
        exit 1
        ;;
esac
