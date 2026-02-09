# Phase 4 iOS - Quick Reference

## What Changed

### 1. APIEndpoints.swift - 15 New Endpoints
```swift
// Engine Control
.engineStart, .engineStop, .enginePause, .engineResume

// Bulk Operations
.cancelAllOrders, .closeAllPositions

// Diagnostics
.tradingWhy, .tradingMetrics, .telemetryEvents

// Physics
.physicsState(symbol), .physicsParticipants(symbol), .physicsAnomalies(limit)

// Brain
.brainThoughts, .brainAnalysis

// Temporal
.temporalStack
```

### 2. SettingsView - Engine Controls
- ✅ Start Engine (with confirmation)
- ✅ Pause Engine (with confirmation)
- ✅ Resume Engine (with confirmation)
- ✅ Stop Engine (with confirmation)
- ✅ Emergency Kill Switch (existing, maintained)

### 3. Navigation Links Added
- **LiveView** → MarketsView (top-leading toolbar)
- **ActionView** → StrategiesView (top-trailing toolbar)
- **XRayView** → SignalFeedView (top-trailing toolbar)

### 4. WebSocket Topics (4 new)
```swift
websocket.subscribe(topic: "system_heartbeat")
websocket.subscribe(topic: "account_state")
websocket.subscribe(topic: "physics_events")
websocket.subscribe(topic: "brain_events")
```

## Files Modified (6)
1. `ios/HEAN/Core/Networking/APIEndpoints.swift`
2. `ios/HEAN/Features/Settings/SettingsView.swift`
3. `ios/HEAN/Features/Live/LiveView.swift`
4. `ios/HEAN/Features/Action/ActionView.swift`
5. `ios/HEAN/Features/XRay/XRayView.swift`
6. `ios/HEAN/Services/Services.swift`

## How to Test

### Engine Control
1. Open Settings tab
2. Tap Start Engine → Confirm
3. Tap Pause Engine → Confirm
4. Tap Resume Engine → Confirm
5. Tap Stop Engine → Confirm
6. Verify loading states and messages

### Navigation
1. **Live tab** → Tap Markets icon (chart icon, top-left)
2. **Action tab** → Tap Strategies icon (brain icon, top-right)
3. **X-Ray tab** → Tap Signals icon (waveform icon, top-right)
4. Verify all views load correctly

### WebSocket
1. Check console logs for topic subscriptions
2. Monitor connection status
3. Verify existing orders/positions updates still work
4. New topics are subscribed but not yet handled (Phase 5)

## API Endpoint Examples

### Start Engine
```swift
POST /api/v1/engine/start
Body: { "confirm_phrase": "I_UNDERSTAND_LIVE_TRADING" }
```

### Trading Diagnostics
```swift
GET /api/v1/trading/why
GET /api/v1/trading/metrics
```

### Physics Data
```swift
GET /api/v1/physics/state?symbol=BTCUSDT
GET /api/v1/physics/participants?symbol=BTCUSDT
GET /api/v1/physics/anomalies?limit=10
```

## Risk Level: LOW
- UI changes only
- No backend modifications
- No data model changes
- Proper error handling
- User confirmations in place

## Build: Not Required
All changes use existing compiled Swift files. No Xcode project modifications needed.

---
**Status:** ✅ COMPLETE
**Next:** Phase 5 - Real-time event handlers for new WebSocket topics
