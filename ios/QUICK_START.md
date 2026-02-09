# HEAN iOS - Quick Start Guide

Get up and running with HEAN iOS components in 5 minutes.

---

## Files Created

```
ios/HEAN/DesignSystem/
├── Colors/AppColors.swift              ✓ Design tokens
├── Typography/AppTypography.swift      ✓ Typography system
└── Components/
    ├── GlassCard.swift                 ✓ 1. Glassmorphism card
    ├── PriceTicker.swift               ✓ 2. Real-time price
    ├── Sparkline.swift                 ✓ 3. Mini chart
    ├── PnLBadge.swift                  ✓ 4. P&L indicator
    ├── RiskBadge.swift                 ✓ 5. Risk state
    ├── StatusIndicator.swift           ✓ 6. Connection status
    ├── SkeletonView.swift              ✓ 7. Loading placeholder
    ├── CandlestickChart.swift          ✓ 8. OHLCV chart
    └── ComponentShowcase.swift         ✓ Demo view

ios/
├── README.md                           ✓ Main docs
├── COMPONENTS_VISUAL_GUIDE.md          ✓ Visual reference
├── INTEGRATION_GUIDE.md                ✓ Backend integration
└── DEPLOYMENT_COMPLETE.md              ✓ Full summary
```

**Total: 8 components + design system + 4 docs**

---

## Component Overview

### 1. GlassCard
Premium card with glassmorphism effect.

**Usage:**
```swift
GlassCard {
    Text("Content")
}
```

**Features:** Blur, gradient border, shadow

---

### 2. PriceTicker
Real-time price with animations.

**Usage:**
```swift
PriceTicker(
    symbol: "BTCUSDT",
    price: 42_350.75,
    changePercent: 3.45,
    size: .large
)
```

**Animations:** Pulse on change, flash background

---

### 3. Sparkline
Mini trend chart.

**Usage:**
```swift
Sparkline(
    dataPoints: [100, 105, 110, 115],
    showGradient: true
)
```

**Auto-colors:** Green (up), Red (down)

---

### 4. PnLBadge
Profit/loss indicator.

**Usage:**
```swift
PnLBadge(value: 1234.56, format: .dollar, size: .compact)
```

**Formats:** dollar, percent, combined

---

### 5. RiskBadge
Risk state indicator.

**Usage:**
```swift
RiskBadge(state: .normal, variant: .compact)
```

**States:** NORMAL, SOFT_BRAKE, QUARANTINE, HARD_STOP

---

### 6. StatusIndicator
Connection status.

**Usage:**
```swift
StatusIndicator(status: .connected, latency: 45)
```

**States:** connected, disconnected, reconnecting

---

### 7. SkeletonView
Loading placeholder.

**Usage:**
```swift
MyView()
    .skeleton(isLoading: true)
```

**Animation:** Shimmer (1.5s loop)

---

### 8. CandlestickChart
Full OHLCV chart.

**Usage:**
```swift
CandlestickChart(
    candles: candleData,
    currentPrice: 42_500
)
```

**Gestures:** Pinch to zoom, drag to scroll

---

## 5-Minute Setup

### Step 1: Create Xcode Project
```bash
cd /Users/macbookpro/Desktop/HEAN/ios
# Create new iOS App in Xcode
# Name: HEAN
# Interface: SwiftUI
# Language: Swift
```

### Step 2: Add Files
Drag into Xcode:
- `HEAN/DesignSystem/` folder

### Step 3: Update App Entry Point
Replace `HEANApp.swift` content with ComponentShowcase launcher.

### Step 4: Run
```
Cmd+R to build and run
```

**Result:** ComponentShowcase displays all 8 components working.

---

## Color Reference

```swift
// Dark theme backgrounds
AppColors.backgroundPrimary    // #0A0A0F (darkest)
AppColors.backgroundSecondary  // #12121A (cards)
AppColors.backgroundTertiary   // #1A1A24 (elevated)

// Semantic colors
AppColors.accentPrimary        // #00D4FF (cyan)
AppColors.success              // #22C55E (green)
AppColors.error                // #EF4444 (red)
AppColors.warning              // #F59E0B (orange)

// Text
AppColors.textPrimary          // #FFFFFF (white)
AppColors.textSecondary        // #A1A1AA (gray)
```

---

## Common Patterns

### Dashboard Header
```swift
GlassCard {
    HStack {
        Text("HEAN").font(.title)
        Spacer()
        StatusIndicator(status: .connected, latency: 32)
    }
    .padding()
}
```

### Price Card
```swift
GlassCard {
    VStack {
        PriceTicker(symbol: "BTC", price: 42350.75, changePercent: 3.45, size: .large)
        Sparkline(dataPoints: last24h)
    }
    .padding()
}
```

### Portfolio Summary
```swift
GlassCard {
    VStack {
        HStack {
            Text("Total P&L")
            Spacer()
            PnLBadge(value: 2456.78, format: .dollar, size: .expanded)
        }
        HStack {
            Text("Risk State")
            Spacer()
            RiskBadge(state: .normal, variant: .compact)
        }
    }
    .padding()
}
```

---

## Design Tokens

### Spacing
```swift
AppTypography.xs   // 8pt
AppTypography.sm   // 12pt
AppTypography.md   // 16pt
AppTypography.lg   // 20pt
AppTypography.xl   // 24pt
```

### Corner Radius
```swift
AppTypography.radiusSm  // 8pt
AppTypography.radiusMd  // 12pt
AppTypography.radiusLg  // 16pt
```

### Animations
```swift
AppAnimation.fast    // 0.15s
AppAnimation.normal  // 0.25s
AppAnimation.slow    // 0.4s
AppAnimation.spring  // Spring with bounce
```

---

## Next Steps

1. **Explore Components**
   - Run ComponentShowcase
   - Open component files
   - View Xcode previews (Cmd+Option+Enter)

2. **Read Documentation**
   - `README.md` - Full API reference
   - `COMPONENTS_VISUAL_GUIDE.md` - Visual examples
   - `INTEGRATION_GUIDE.md` - Backend connection

3. **Start Building**
   - Create DashboardView
   - Connect to backend WebSocket
   - Build trading interface

---

## Support

**Documentation:**
- Main README: `/Users/macbookpro/Desktop/HEAN/ios/README.md`
- Visual Guide: `/Users/macbookpro/Desktop/HEAN/ios/COMPONENTS_VISUAL_GUIDE.md`
- Integration: `/Users/macbookpro/Desktop/HEAN/ios/INTEGRATION_GUIDE.md`

**Component Files:**
All in: `/Users/macbookpro/Desktop/HEAN/ios/HEAN/DesignSystem/Components/`

---

**Ready to build. Start with ComponentShowcase.**
