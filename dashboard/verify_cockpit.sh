#!/bin/bash

# Cockpit Tab Verification Script
# Checks that all files exist and dependencies are met

set -e

echo "🎛️  HEAN Cockpit Tab Verification"
echo "================================="
echo ""

DASHBOARD_DIR="/Users/macbookpro/Desktop/HEAN/dashboard"
ERRORS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "$DASHBOARD_DIR" ]; then
  echo -e "${RED}✗ Dashboard directory not found at $DASHBOARD_DIR${NC}"
  exit 1
fi

cd "$DASHBOARD_DIR"

echo "📁 Checking file structure..."
echo ""

# List of required files
FILES=(
  "src/types/api.ts"
  "src/components/ui/AnimatedNumber.tsx"
  "src/components/cockpit/EquityChart.tsx"
  "src/components/cockpit/AssetDonutChart.tsx"
  "src/components/cockpit/MetricRow.tsx"
  "src/components/cockpit/LiveFeed.tsx"
  "src/components/cockpit/PositionsSummary.tsx"
  "src/components/cockpit/index.ts"
  "src/components/tabs/CockpitTab.tsx"
  "src/components/tabs/index.ts"
  "src/utils/testData.ts"
)

for file in "${FILES[@]}"; do
  if [ -f "$file" ]; then
    echo -e "${GREEN}✓${NC} $file"
  else
    echo -e "${RED}✗${NC} $file (MISSING)"
    ERRORS=$((ERRORS + 1))
  fi
done

echo ""
echo "📦 Checking dependencies..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
  echo -e "${YELLOW}⚠${NC}  node_modules not found - run 'npm install'"
  ERRORS=$((ERRORS + 1))
else
  # Check for required packages
  DEPS=("react" "react-dom" "framer-motion" "recharts" "clsx")

  for dep in "${DEPS[@]}"; do
    if [ -d "node_modules/$dep" ]; then
      echo -e "${GREEN}✓${NC} $dep"
    else
      echo -e "${RED}✗${NC} $dep (MISSING - run 'npm install $dep')"
      ERRORS=$((ERRORS + 1))
    fi
  done
fi

echo ""
echo "🔌 Checking backend API..."
echo ""

# Check if backend is running
if curl -s -f http://localhost:8000/api/v1/engine/status > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} Backend API is accessible at http://localhost:8000"

  # Test each endpoint
  ENDPOINTS=(
    "engine/status"
    "orders/positions"
    "risk/governor/status"
    "trading/metrics"
  )

  for endpoint in "${ENDPOINTS[@]}"; do
    if curl -s -f "http://localhost:8000/api/v1/$endpoint" > /dev/null 2>&1; then
      echo -e "${GREEN}✓${NC} GET /api/v1/$endpoint"
    else
      echo -e "${YELLOW}⚠${NC}  GET /api/v1/$endpoint (not responding)"
    fi
  done
else
  echo -e "${YELLOW}⚠${NC}  Backend API not running at http://localhost:8000"
  echo "   Start it with: cd /Users/macbookpro/Desktop/HEAN && python -m hean.main run"
fi

echo ""
echo "📝 Documentation files..."
echo ""

DOCS=(
  "COCKPIT_TAB_README.md"
  "COCKPIT_IMPLEMENTATION_SUMMARY.md"
  "COCKPIT_QUICK_START.md"
  "INTEGRATION_EXAMPLE.tsx"
)

for doc in "${DOCS[@]}"; do
  if [ -f "$doc" ]; then
    echo -e "${GREEN}✓${NC} $doc"
  else
    echo -e "${YELLOW}⚠${NC}  $doc (missing)"
  fi
done

echo ""
echo "================================="

if [ $ERRORS -eq 0 ]; then
  echo -e "${GREEN}✓ All checks passed!${NC}"
  echo ""
  echo "Next steps:"
  echo "1. Import CockpitTab into your main app"
  echo "2. Run 'npm run dev' to start development server"
  echo "3. Open browser and verify all components render"
  echo ""
  echo "See COCKPIT_QUICK_START.md for integration guide."
  exit 0
else
  echo -e "${RED}✗ $ERRORS error(s) found${NC}"
  echo ""
  echo "Fix the errors above and run this script again."
  exit 1
fi
