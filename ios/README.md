# HEAN iOS Trading App

Premium SwiftUI components for the HEAN crypto trading dashboard.

## Overview

Production-quality iOS app featuring:
- Real-time price tickers with animations
- Interactive candlestick charts
- Glassmorphism design system
- Risk management indicators
- WebSocket status monitoring
- Profit/Loss tracking

## Project Structure

```
ios/HEAN/
├── App/
│   ├── HEANApp.swift              # Main app entry point
│   └── ContentView.swift          # Root view (production)
│
├── DesignSystem/
│   ├── Colors/
│   │   └── AppColors.swift        # Color design tokens
│   ├── Typography/
│   │   └── AppTypography.swift    # Typography & spacing tokens
│   └── Components/
│       ├── GlassCard.swift        # Glassmorphism card
│       ├── PriceTicker.swift      # Real-time price display
│       ├── Sparkline.swift        # Mini trend chart
│       ├── PnLBadge.swift         # Profit/Loss indicator
│       ├── RiskBadge.swift        # Risk state indicator
│       ├── StatusIndicator.swift  # Connection status
│       ├── SkeletonView.swift     # Loading placeholder
│       ├── CandlestickChart.swift # Full OHLCV chart
│       └── ComponentShowcase.swift # Demo view
│
├── Core/
│   ├── Networking/                # API client
│   ├── WebSocket/                 # Real-time data
│   ├── DI/                        # Dependency injection
│   ├── Utils/                     # Utilities
│   └── Logger/                    # Logging
│
└── Features/
    ├── Dashboard/                 # Main dashboard
    ├── Trading/                   # Trading interface
    ├── Portfolio/                 # Portfolio view
    └── Settings/                  # App settings
```

## Design System

### Color Tokens

```swift
// Backgrounds
AppColors.backgroundPrimary    // #0A0A0F - Main background
AppColors.backgroundSecondary  // #12121A - Cards
AppColors.backgroundTertiary   // #1A1A24 - Elevated surfaces

// Accents
AppColors.accentPrimary        // #00D4FF - Primary accent
AppColors.success              // #22C55E - Bullish/positive
AppColors.error                // #EF4444 - Bearish/negative
AppColors.warning              // #F59E0B - Warnings

// Text
AppColors.textPrimary          // #FFFFFF - Primary text
AppColors.textSecondary        // #A1A1AA - Secondary text
AppColors.textTertiary         // #71717A - Tertiary text
```

### Spacing (8pt grid)

```swift
AppTypography.xs  // 8pt
AppTypography.sm  // 12pt
AppTypography.md  // 16pt
AppTypography.lg  // 20pt
AppTypography.xl  // 24pt
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
AppAnimation.spring  // Spring animation with bounce
```

## Components

### 1. GlassCard

Premium glassmorphism card with blur and gradient border.

```swift
GlassCard {
    VStack {
        Text("Title")
        Text("Content")
    }
    .padding()
}
```

**Features:**
- Ultra-thin material blur
- Gradient border
- Configurable corner radius
- Drop shadow

### 2. PriceTicker

Real-time price display with animations.

```swift
PriceTicker(
    symbol: "BTCUSDT",
    price: 42_350.75,
    changePercent: 3.45,
    size: .large
)
```

**Sizes:** `.small`, `.medium`, `.large`

**Features:**
- Monospaced price font
- Auto-colored change indicator
- Pulse animation on price change
- Flash background (green/red)
- VoiceOver accessible

### 3. Sparkline

Mini trend chart with auto-coloring.

```swift
Sparkline(
    dataPoints: [100, 105, 103, 110, 115],
    showGradient: true,
    smoothCurves: true
)
```

**Features:**
- Auto-color based on trend
- Optional gradient fill
- Smooth bezier curves
- Responsive sizing

### 4. PnLBadge

Profit/Loss indicator chip.

```swift
PnLBadge(value: 1234.56, format: .dollar, size: .compact)
PnLBadge(value: 12.34, format: .percent, size: .expanded)
PnLBadge(value: 2500.00, format: .combined, size: .expanded)
```

**Formats:**
- `.dollar` - $1,234.56
- `.percent` - +12.34%
- `.combined` - $1,234.56 (+12.34%)

**Sizes:** `.compact`, `.expanded`

### 5. RiskBadge

Risk state indicator with animated pulse.

```swift
RiskBadge(state: .normal, variant: .compact)
RiskBadge(state: .softBrake, variant: .expanded)
```

**States:**
- `.normal` - All systems operational (green)
- `.softBrake` - Reduced position sizing (yellow)
- `.quarantine` - Trading paused (orange)
- `.hardStop` - Trading halted (red)

**Variants:**
- `.compact` - Icon only
- `.expanded` - Icon + label + description

### 6. StatusIndicator

Connection status with pulsing animation.

```swift
StatusIndicator(
    status: .connected,
    latency: 45,
    showLabel: true
)
```

**States:** `.connected`, `.disconnected`, `.reconnecting`

**Features:**
- Colored status dot
- Optional latency display
- Pulsing animation for reconnecting

### 7. SkeletonView

Loading placeholder with shimmer.

```swift
// Any view can show skeleton state
Text("Loading...")
    .skeleton(isLoading: true)

// Or wrap in SkeletonView
SkeletonView(isLoading: isLoading) {
    MyComplexView()
}
```

**Features:**
- Automatic shimmer animation
- Works with any view shape
- 1.5s loop duration

### 8. CandlestickChart

Full OHLCV candlestick chart.

```swift
CandlestickChart(
    candles: candleData,
    currentPrice: 42_500,
    showGrid: true,
    showVolume: true
)
```

**Features:**
- Green/red candles
- High/low wicks
- Grid lines
- Price scale on right
- Pinch to zoom gesture
- Drag to scroll gesture
- Current price dashed line

**Data Model:**
```swift
struct Candle {
    let timestamp: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double
}
```

## Quick Start

### 1. View Component Showcase

The `ComponentShowcase` view demonstrates all components in action:

```swift
@main
struct HEANApp: App {
    var body: some Scene {
        WindowGroup {
            ComponentShowcase()
        }
    }
}
```

### 2. Build and Run

1. Open Xcode
2. Select HEAN target
3. Choose simulator or device
4. Press Cmd+R to build and run

### 3. Preview Individual Components

All components include SwiftUI previews:

```swift
#Preview {
    PriceTicker(
        symbol: "BTCUSDT",
        price: 42_350.75,
        changePercent: 3.45,
        size: .large
    )
}
```

## Usage Examples

### Dashboard Card with Price

```swift
GlassCard {
    VStack(alignment: .leading, spacing: AppTypography.md) {
        HStack {
            Text("BTC/USDT")
                .font(AppTypography.headline())
            Spacer()
            StatusIndicator(status: .connected, latency: 32)
        }

        PriceTicker(
            symbol: "BTCUSDT",
            price: btcPrice,
            changePercent: 3.45,
            size: .large
        )

        Sparkline(
            dataPoints: last24hPrices,
            smoothCurves: true
        )
    }
    .padding(AppTypography.md)
}
```

### Portfolio Summary

```swift
GlassCard {
    VStack(spacing: AppTypography.md) {
        HStack {
            Text("Total P&L")
            Spacer()
            PnLBadge(value: 2_456.78, format: .dollar, size: .expanded)
        }

        HStack {
            Text("Today")
            Spacer()
            PnLBadge(value: 15.67, format: .percent, size: .compact)
        }

        HStack {
            Text("Risk State")
            Spacer()
            RiskBadge(state: .normal, variant: .compact)
        }
    }
    .padding(AppTypography.md)
}
```

### Loading State

```swift
struct MyView: View {
    @State private var isLoading = true
    @State private var data: [Item] = []

    var body: some View {
        VStack {
            if isLoading {
                // Skeleton placeholders
                ForEach(0..<5) { _ in
                    RoundedRectangle(cornerRadius: 8)
                        .fill(AppColors.backgroundSecondary)
                        .frame(height: 60)
                }
                .skeleton(isLoading: true)
            } else {
                // Real data
                ForEach(data) { item in
                    ItemRow(item: item)
                }
            }
        }
        .onAppear {
            loadData()
        }
    }
}
```

## Accessibility

All components are VoiceOver accessible:

```swift
// Automatic accessibility labels
PriceTicker(symbol: "BTC", price: 42350.75, changePercent: 3.45, size: .medium)
// VoiceOver: "BTCUSDT price $42,350.75, change +3.45%"

RiskBadge(state: .softBrake, variant: .expanded)
// VoiceOver: "Risk state: SOFT BRAKE. Reduced position sizing"

StatusIndicator(status: .connected, latency: 45, showLabel: true)
// VoiceOver: "Status: Connected, latency 45 milliseconds"
```

## Animation Best Practices

### Price Updates

```swift
@State private var price: Double = 42_350.75

// Price ticker animates automatically on change
PriceTicker(symbol: "BTC", price: price, changePercent: 2.5, size: .large)

// Update price to trigger animation
price = 42_500.00  // Triggers pulse + flash
```

### State Transitions

```swift
@State private var riskState: RiskBadge.RiskState = .normal

// Risk badge animates when state changes
RiskBadge(state: riskState, variant: .expanded)

// Change state
withAnimation(AppAnimation.spring) {
    riskState = .softBrake  // Triggers pulse animation
}
```

## Performance

### List Optimization

Use `LazyVStack`/`LazyHStack` for large lists:

```swift
ScrollView {
    LazyVStack(spacing: AppTypography.md) {
        ForEach(prices) { price in
            PriceTicker(
                symbol: price.symbol,
                price: price.value,
                changePercent: price.change,
                size: .small
            )
        }
    }
}
```

### Chart Optimization

Candlestick chart uses `Canvas` for efficient rendering of large datasets (50+ candles).

## Troubleshooting

### Issue: Components not visible

**Solution:** Ensure dark mode is enabled:
```swift
ContentView()
    .preferredColorScheme(.dark)
```

### Issue: Animations not working

**Solution:** Check that state changes trigger view updates:
```swift
@State private var value: Double = 100  // Use @State
```

### Issue: Shimmer not animating

**Solution:** Ensure `isLoading` is true:
```swift
.skeleton(isLoading: true)  // Not false
```

## Next Steps

1. **Integrate WebSocket:** Connect `PriceTicker` to real-time data
2. **Add Navigation:** Implement tab bar and navigation
3. **Build Features:** Create Dashboard, Trading, Portfolio views
4. **Connect API:** Wire up backend endpoints
5. **Add Persistence:** Store user preferences

## Resources

- [SwiftUI Documentation](https://developer.apple.com/documentation/swiftui)
- [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [Accessibility Best Practices](https://developer.apple.com/accessibility/)

## License

Proprietary - HEAN Trading System
