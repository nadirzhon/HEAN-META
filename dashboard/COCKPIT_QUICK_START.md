# Cockpit Tab - Quick Start Guide

## TL;DR - Copy/Paste Integration

### 1. Install Dependencies (if not already installed)

```bash
cd /Users/macbookpro/Desktop/HEAN/dashboard
npm install recharts framer-motion clsx
```

### 2. Import and Use

```tsx
import { CockpitTab } from "./src/components/tabs/CockpitTab";

function App() {
  return (
    <div style={{ background: "#0D0D1A", minHeight: "100vh" }}>
      <CockpitTab />
    </div>
  );
}
```

### 3. Verify Backend is Running

```bash
# Terminal 1: Start HEAN backend
cd /Users/macbookpro/Desktop/HEAN
python -m hean.main run

# Terminal 2: Test API
curl http://localhost:8000/api/v1/engine/status

# Should return JSON with status, equity, etc.
```

### 4. Start Dashboard

```bash
npm run dev
```

That's it! CockpitTab will:
- Fetch data from API on mount
- Refresh every 5 seconds
- Connect to WebSocket for live updates
- Show loading/error states automatically
- Handle all animations

## What You Get

### 6 Metric Cards (Top Row)
- Total Equity: $XX,XXX.XX
- Daily PnL: +$XX.XX (green) or -$XX.XX (red)
- Win Rate: XX.X%
- Risk State: NORMAL (green badge)
- Active Positions: X
- Signals/min: X.X

### 2 Charts (Middle Row)
- **Left**: Equity curve over time (purple gradient area chart)
- **Right**: Portfolio allocation donut chart by asset

### 2 Tables (Bottom Row)
- **Left**: Open positions (symbol, side, size, entry, PnL)
- **Right**: Live event feed (scrolling, color-coded)

## Customization

### Change API URL

Edit `/src/components/tabs/CockpitTab.tsx`:

```typescript
// Line 42-43
const API_BASE = "http://localhost:8000/api/v1";  // Change this
const WS_URL = "ws://localhost:8000/ws";          // And this
```

### Change Refresh Rate

Edit `/src/components/tabs/CockpitTab.tsx`:

```typescript
// Line 188
const interval = setInterval(fetchData, 5000); // 5000ms = 5 seconds
```

### Change Color Palette

Edit colors directly in components or create a theme context. All colors are hardcoded as hex values (not CSS variables) for visibility.

## Troubleshooting

### Problem: Blank screen
**Solution:** Check browser console for errors. Verify backend is running.

### Problem: "Failed to fetch data"
**Solution:** Backend not running or wrong URL. Check `http://localhost:8000/api/v1/engine/status` in browser.

### Problem: WebSocket not connecting
**Solution:** Check Network tab > WS. Verify backend WebSocket endpoint is accessible.

### Problem: TypeScript errors
**Solution:** Run `npm install` to ensure all dependencies are installed.

### Problem: Numbers not animating
**Solution:** Verify framer-motion is installed: `npm list framer-motion`

### Problem: Charts not showing
**Solution:** Verify recharts is installed: `npm list recharts`

## File Structure

```
dashboard/src/
├── types/
│   └── api.ts                    # Backend API types
├── components/
│   ├── ui/
│   │   └── AnimatedNumber.tsx    # Animated number component
│   ├── cockpit/                  # All cockpit components
│   │   ├── EquityChart.tsx
│   │   ├── AssetDonutChart.tsx
│   │   ├── MetricRow.tsx
│   │   ├── LiveFeed.tsx
│   │   ├── PositionsSummary.tsx
│   │   └── index.ts
│   └── tabs/
│       └── CockpitTab.tsx        # Main component (start here)
└── utils/
    └── testData.ts               # Mock data (dev only)
```

## API Endpoints Expected

Your backend must provide these endpoints:

```
GET  /api/v1/engine/status
→ { status, running, equity, daily_pnl, initial_capital }

GET  /api/v1/orders/positions
→ [{ position_id, symbol, side, size, entry_price, current_price, unrealized_pnl, ... }]

GET  /api/v1/risk/governor/status
→ { risk_state, level, reason_codes, quarantined_symbols, can_clear }

GET  /api/v1/trading/metrics
→ { counters: { last_1m: {...}, last_5m: {...}, session: {...} } }

WS   ws://localhost:8000/ws
→ Stream of events: { type, timestamp, data, ... }
```

## Testing Without Backend

Use mock data for isolated testing:

```tsx
import { CockpitTab } from "./src/components/tabs/CockpitTab";
import { generateMockCockpitData } from "./src/utils/testData";

function TestApp() {
  // WARNING: For testing only!
  const mockData = generateMockCockpitData();

  return (
    <div style={{ background: "#0D0D1A", minHeight: "100vh" }}>
      {/* TODO: Create a MockCockpitTab that uses mockData */}
      <CockpitTab />
    </div>
  );
}
```

**Note:** testData.ts provides mock generators, but CockpitTab itself always uses real API. You'd need to create a separate MockCockpitTab component for isolated testing.

## Performance Tips

1. **Keep backend healthy**: Slow API responses = slow dashboard
2. **Monitor WebSocket**: Check for dropped connections in console
3. **Limit history**: Default 100 points for equity - adjust if needed
4. **Limit events**: Default 100 events in feed - adjust if needed

## Ready for Production?

- [x] Backend running and accessible
- [x] API endpoints returning correct data shape
- [x] WebSocket endpoint working
- [x] Dependencies installed (recharts, framer-motion, clsx)
- [x] TypeScript compiles without errors
- [x] No console errors in browser DevTools
- [x] Error boundary wrapping CockpitTab (recommended)

## Full Documentation

For complete details, see:
- `COCKPIT_TAB_README.md` - Component documentation
- `COCKPIT_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `INTEGRATION_EXAMPLE.tsx` - 5 integration patterns

## Support

If issues persist:
1. Check all files exist at expected paths
2. Verify TypeScript types match backend response
3. Test API endpoints manually with curl
4. Check WebSocket in browser Network tab
5. Look for CORS issues if backend on different port

---

**Quick Start complete.** Import CockpitTab, run backend, start dev server. Everything else is automatic.
