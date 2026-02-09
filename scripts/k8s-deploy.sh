#!/bin/bash
# ============================================
# HEAN Kubernetes Deployment Script
# ============================================
set -e

echo "☸️  HEAN Kubernetes Deployment"
echo "=============================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables
NAMESPACE="hean-production"
ACTION=${1:-deploy}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed.${NC}"
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster.${NC}"
    exit 1
fi

case $ACTION in
    deploy)
        echo -e "${YELLOW}📦 Creating namespace...${NC}"
        kubectl apply -f k8s/namespace.yaml
        
        echo -e "${YELLOW}⚙️  Creating ConfigMap...${NC}"
        kubectl apply -f k8s/configmap.yaml
        
        echo -e "${BLUE}🔐 Checking secrets...${NC}"
        if ! kubectl get secret hean-secrets -n $NAMESPACE &> /dev/null; then
            echo -e "${RED}❌ Secret 'hean-secrets' not found!${NC}"
            echo "Please create secrets first:"
            echo "  kubectl create secret generic hean-secrets \\"
            echo "    --from-literal=BYBIT_API_KEY='your-key' \\"
            echo "    --from-literal=OPENAI_API_KEY='sk-...' \\"
            echo "    -n $NAMESPACE"
            exit 1
        fi
        echo -e "${GREEN}✅ Secrets exist${NC}"
        
        echo -e "${YELLOW}💾 Deploying Redis...${NC}"
        kubectl apply -f k8s/redis-deployment.yaml
        
        echo -e "${YELLOW}⏳ Waiting for Redis to be ready...${NC}"
        kubectl wait --for=condition=ready pod -l component=redis -n $NAMESPACE --timeout=120s
        
        echo -e "${YELLOW}🔌 Deploying API...${NC}"
        kubectl apply -f k8s/api-deployment.yaml
        
        echo -e "${YELLOW}🌐 Deploying UI...${NC}"
        kubectl apply -f k8s/ui-deployment.yaml
        
        echo -e "${YELLOW}⏳ Waiting for deployments...${NC}"
        kubectl rollout status deployment/hean-api -n $NAMESPACE --timeout=300s
        kubectl rollout status deployment/hean-ui -n $NAMESPACE --timeout=300s
        
        echo -e "${GREEN}✅ Deployment successful!${NC}"
        echo ""
        echo "To access your application:"
        echo "  kubectl get ingress -n $NAMESPACE"
        ;;
    
    update)
        echo -e "${YELLOW}🔄 Updating deployments...${NC}"
        kubectl apply -f k8s/api-deployment.yaml
        kubectl apply -f k8s/ui-deployment.yaml
        kubectl rollout restart deployment/hean-api -n $NAMESPACE
        kubectl rollout restart deployment/hean-ui -n $NAMESPACE
        echo -e "${GREEN}✅ Update complete${NC}"
        ;;
    
    rollback)
        echo -e "${YELLOW}⏪ Rolling back...${NC}"
        kubectl rollout undo deployment/hean-api -n $NAMESPACE
        kubectl rollout undo deployment/hean-ui -n $NAMESPACE
        echo -e "${GREEN}✅ Rollback complete${NC}"
        ;;
    
    status)
        echo -e "${BLUE}📊 Cluster Status:${NC}"
        kubectl get all -n $NAMESPACE
        echo ""
        echo -e "${BLUE}🔍 Pod Status:${NC}"
        kubectl get pods -n $NAMESPACE -o wide
        echo ""
        echo -e "${BLUE}🌐 Services:${NC}"
        kubectl get svc -n $NAMESPACE
        echo ""
        echo -e "${BLUE}🚪 Ingress:${NC}"
        kubectl get ingress -n $NAMESPACE
        ;;
    
    logs)
        COMPONENT=${2:-api}
        echo -e "${YELLOW}📜 Showing logs for: ${COMPONENT}${NC}"
        kubectl logs -f -l component=$COMPONENT -n $NAMESPACE --tail=100
        ;;
    
    scale)
        COMPONENT=${2:-api}
        REPLICAS=${3:-3}
        echo -e "${YELLOW}📈 Scaling ${COMPONENT} to ${REPLICAS} replicas...${NC}"
        kubectl scale deployment/hean-$COMPONENT -n $NAMESPACE --replicas=$REPLICAS
        echo -e "${GREEN}✅ Scaled successfully${NC}"
        ;;
    
    shell)
        COMPONENT=${2:-api}
        echo -e "${YELLOW}🐚 Opening shell in ${COMPONENT}...${NC}"
        POD=$(kubectl get pod -n $NAMESPACE -l component=$COMPONENT -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -it $POD -n $NAMESPACE -- /bin/bash
        ;;
    
    delete)
        echo -e "${RED}⚠️  WARNING: This will delete all HEAN resources!${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo -e "${YELLOW}🗑️  Deleting resources...${NC}"
            kubectl delete -f k8s/
            echo -e "${GREEN}✅ Resources deleted${NC}"
        else
            echo "Aborted."
        fi
        ;;
    
    backup)
        echo -e "${YELLOW}💾 Creating backup...${NC}"
        mkdir -p backups
        kubectl exec -n $NAMESPACE deployment/redis -- redis-cli BGSAVE
        sleep 2
        POD=$(kubectl get pod -n $NAMESPACE -l component=redis -o jsonpath='{.items[0].metadata.name}')
        kubectl cp $NAMESPACE/$POD:/data/dump.rdb ./backups/redis-k8s-$(date +%Y%m%d-%H%M%S).rdb
        echo -e "${GREEN}✅ Backup created in ./backups/${NC}"
        ;;
    
    health)
        echo -e "${BLUE}🏥 Health Check:${NC}"
        
        # Check API
        API_READY=$(kubectl get deployment hean-api -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
        API_DESIRED=$(kubectl get deployment hean-api -n $NAMESPACE -o jsonpath='{.spec.replicas}')
        if [ "$API_READY" = "$API_DESIRED" ]; then
            echo -e "${GREEN}✅ API: ${API_READY}/${API_DESIRED} ready${NC}"
        else
            echo -e "${RED}❌ API: ${API_READY}/${API_DESIRED} ready${NC}"
        fi
        
        # Check UI
        UI_READY=$(kubectl get deployment hean-ui -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
        UI_DESIRED=$(kubectl get deployment hean-ui -n $NAMESPACE -o jsonpath='{.spec.replicas}')
        if [ "$UI_READY" = "$UI_DESIRED" ]; then
            echo -e "${GREEN}✅ UI: ${UI_READY}/${UI_DESIRED} ready${NC}"
        else
            echo -e "${RED}❌ UI: ${UI_READY}/${UI_DESIRED} ready${NC}"
        fi
        
        # Check Redis
        REDIS_READY=$(kubectl get deployment redis -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
        if [ "$REDIS_READY" = "1" ]; then
            echo -e "${GREEN}✅ Redis: ready${NC}"
        else
            echo -e "${RED}❌ Redis: not ready${NC}"
        fi
        ;;
    
    *)
        echo "Usage: $0 [action]"
        echo ""
        echo "Actions:"
        echo "  deploy    - Deploy all resources"
        echo "  update    - Update deployments"
        echo "  rollback  - Rollback to previous version"
        echo "  status    - Show cluster status"
        echo "  logs      - Show logs (usage: logs [api|ui|redis])"
        echo "  scale     - Scale deployment (usage: scale [api|ui] [replicas])"
        echo "  shell     - Open shell in pod (usage: shell [api|ui|redis])"
        echo "  delete    - Delete all resources"
        echo "  backup    - Backup Redis data"
        echo "  health    - Check health status"
        exit 1
        ;;
esac
