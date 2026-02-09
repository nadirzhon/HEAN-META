# HEAN iOS Components - Visual Guide

Complete visual reference for all SwiftUI components with code examples.

---

## 1. GlassCard

**Premium glassmorphism card with ultra-thin blur and gradient border**

```swift
GlassCard {
    VStack(alignment: .leading, spacing: 12) {
        Text("Trading Stats")
            .font(.system(size: 18, weight: .semibold))
        Text("Real-time data")
            .font(.system(size: 14))
            .foregroundColor(Color(hex: "A1A1AA"))
    }
    .padding(16)
}
```

**Visual Representation:**
```
┌─────────────────────────────────┐
│  ✨ Glassmorphism Effect       │
│                                 │
│  Trading Stats                  │
│  Real-time data                 │
│                                 │
└─────────────────────────────────┘
```

**Features:**
- Ultra-thin material blur background
- Gradient border (white 30% → 10% → 5%)
- Configurable corner radius (8pt, 12pt, 16pt)
- Drop shadow (10pt radius)

**Variants:**
```swift
// Basic (default)
GlassCard { content }

// Large radius
GlassCard(cornerRadius: 16) { content }

// Compact with small shadow
GlassCard(cornerRadius: 8, shadowRadius: 5) { content }
```

---

## 2. PriceTicker

**Real-time price display with pulse animation and flash effects**

```swift
PriceTicker(
    symbol: "BTCUSDT",
    price: 42_350.75,
    changePercent: 3.45,
    size: .large
)
```

**Visual Representation (Bullish):**
```
┌──────────────────┐
│ BTCUSDT          │  ← Gray label
│ $42,350.75       │  ← White monospaced (pulses on change)
│ ↗ +3.45%         │  ← Green with up arrow
└──────────────────┘
   (Flash green background on price increase)
```

**Visual Representation (Bearish):**
```
┌──────────────────┐
│ ETHUSDT          │  ← Gray label
│ $2,245.30        │  ← White monospaced
│ ↘ -1.25%         │  ← Red with down arrow
└──────────────────┘
   (Flash red background on price decrease)
```

**Sizes:**

`.small` (compact grid layout):
```
┌──────┬──────┬──────┐
│ BTC  │ ETH  │ SOL  │
│$42.3k│$2.2k │$95.40│
│↗+3.4%│↘-1.2%│↗+5.2%│
└──────┴──────┴──────┘
```

`.medium` (standard):
```
┌─────────────┐
│ BTCUSDT     │
│ $42,350.75  │
│ ↗ +3.45%    │
└─────────────┘
```

`.large` (featured):
```
┌───────────────────┐
│ BTCUSDT           │
│ $42,350.75        │
│ ↗ +3.45%          │
└───────────────────┘
```

**Animations:**
- **Pulse:** Scale 1.0 → 1.05 → 1.0 (0.15s) on price change
- **Flash:** Background green/red fade (0.3s) on price change

---

## 3. Sparkline

**Mini trend chart with auto-coloring**

```swift
Sparkline(
    dataPoints: [100, 105, 103, 110, 115, 112, 120, 125, 130],
    showGradient: true,
    smoothCurves: true
)
```

**Visual Representation (Upward Trend - Green):**
```
          ╱‾‾╲
        ╱′    ‾╲    ╱‾‾
      ╱′        ╲  ╱
    ╱′           ╲╱
  ╱′
═════════════════════════
```
(Green line with gradient fill below)

**Visual Representation (Downward Trend - Red):**
```
╲
 ╲      ╱‾╲
  ╲    ╱   ╲  ╱‾╲
   ╲  ╱     ╲╱   ╲
    ╲╱            ╲__
═════════════════════════
```
(Red line with gradient fill below)

**Variants:**

**Smooth Curves (Bezier):**
```swift
Sparkline(dataPoints: prices, smoothCurves: true)
```
```
     ╱‾‾‾╲
   ╱′     ‾╲    ╱‾
  ╱         ╲  ╱
 ′           ╲╱
```

**Angular (Sharp):**
```swift
Sparkline(dataPoints: prices, smoothCurves: false)
```
```
      /‾‾\
    /     \   /‾
  /        \_/
/
```

**No Gradient:**
```swift
Sparkline(dataPoints: prices, showGradient: false)
```
(Line only, no fill)

---

## 4. PnLBadge

**Profit/Loss indicator chip with auto-coloring**

```swift
PnLBadge(value: 1234.56, format: .dollar, size: .compact)
```

**Visual Representation:**

**Positive (Green):**
```
┌──────────────┐
│ ↗ +$1,234.56 │  ← Green text on green bg (15% opacity)
└──────────────┘
```

**Negative (Red):**
```
┌─────────────┐
│ ↘ -$567.89  │  ← Red text on red bg (15% opacity)
└─────────────┘
```

**Zero (Gray):**
```
┌─────────┐
│ — $0.00 │  ← Gray text on gray bg (10% opacity)
└─────────┘
```

**Formats:**

**Dollar:**
```swift
PnLBadge(value: 1234.56, format: .dollar)
// Output: ↗ +$1,234.56
```

**Percent:**
```swift
PnLBadge(value: 12.34, format: .percent)
// Output: ↗ +12.34%
```

**Combined:**
```swift
PnLBadge(value: 2500.00, format: .combined)
// Output: ↗ +$2,500.00 (+25.00%)
```

**Sizes:**

**Compact:**
```
┌────────┐
│↗+$1.2k │  ← Small, condensed
└────────┘
```

**Expanded:**
```
┌──────────────┐
│ ↗ +$1,234.56 │  ← Larger padding
└──────────────┘
```

---

## 5. RiskBadge

**Risk state indicator with animated pulse for alerts**

```swift
RiskBadge(state: .normal, variant: .compact)
```

**Visual Representation:**

**NORMAL (Green) - Compact:**
```
 ╭───╮
 │ ✓ │  ← Shield icon in green circle
 ╰───╯
```

**SOFT_BRAKE (Yellow) - Compact:**
```
 ╭───╮
 │ ⚠ │  ← Warning icon in yellow circle (pulsing)
 ╰───╯
```

**QUARANTINE (Orange) - Compact:**
```
 ╭───╮
 │ ⏸ │  ← Pause icon in orange circle (pulsing)
 ╰───╯
```

**HARD_STOP (Red) - Compact:**
```
 ╭───╮
 │ ⊗ │  ← Stop icon in red circle (pulsing)
 ╰───╯
```

**Expanded Variant:**

**NORMAL:**
```
┌─────────────────────────────────────┐
│ ✓ NORMAL                            │
│                                     │
│ All systems operational             │
└─────────────────────────────────────┘
```
(Green text on green background 15% opacity)

**SOFT_BRAKE:**
```
┌─────────────────────────────────────┐
│ ⚠ SOFT BRAKE                        │
│                                     │
│ Reduced position sizing             │
└─────────────────────────────────────┘
```
(Yellow text on yellow background 15% opacity, pulsing border)

**QUARANTINE:**
```
┌─────────────────────────────────────┐
│ ⏸ QUARANTINE                        │
│                                     │
│ Trading paused for review           │
└─────────────────────────────────────┘
```
(Orange text on orange background 15% opacity, pulsing border)

**HARD_STOP:**
```
┌─────────────────────────────────────┐
│ ⊗ HARD STOP                         │
│                                     │
│ Trading halted                      │
└─────────────────────────────────────┘
```
(Red text on red background 15% opacity, pulsing border)

**Animation:**
- Non-normal states pulse border: opacity 30% → 60% → 30% (1s loop)
- Compact variant scales background: 1.0 → 1.2 fade out (1s loop)

---

## 6. StatusIndicator

**Connection status with optional latency**

```swift
StatusIndicator(status: .connected, latency: 45, showLabel: true)
```

**Visual Representation:**

**Connected (Green):**
```
● Connected (45ms)
```
(Solid green dot)

**Reconnecting (Yellow):**
```
◉ Reconnecting
```
(Yellow dot with pulsing ring)

**Disconnected (Red):**
```
● Disconnected
```
(Solid red dot)

**Variants:**

**With Label + Latency:**
```swift
StatusIndicator(status: .connected, latency: 32, showLabel: true)
// Output: ● Connected (32ms)
```

**With Label Only:**
```swift
StatusIndicator(status: .reconnecting, showLabel: true)
// Output: ◉ Reconnecting
```

**Compact (No Label):**
```swift
StatusIndicator(status: .connected, showLabel: false)
// Output: ●
```

**In Context (Header Bar):**
```
┌─────────────────────────────────────┐
│ HEAN              ● Connected (32ms) │
└─────────────────────────────────────┘
```

**Animation:**
- Reconnecting state: Ring scales 1.0 → 1.8 and fades (1.5s loop)

---

## 7. SkeletonView

**Loading placeholder with shimmer animation**

```swift
VStack {
    RoundedRectangle(cornerRadius: 8)
        .fill(Color.gray)
        .frame(width: 200, height: 40)
}
.skeleton(isLoading: true)
```

**Visual Representation:**

**Static (isLoading: false):**
```
┌──────────────┐
│   Content    │
└──────────────┘
```

**Loading with Shimmer (isLoading: true):**
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ ░░░░░░░░░░░░ │  →   │ ▒▒░░░░░░░░░░ │  →   │ ░░░░░░▒▒░░░░ │
└──────────────┘      └──────────────┘      └──────────────┘
   (Frame 1)             (Frame 2)             (Frame 3)
```
(Shimmer sweeps left to right, 1.5s loop)

**Card Skeleton:**
```
┌───────────────────────────┐
│ ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░│  ← Title (shimmer)
│                           │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░│  ← Large text (shimmer)
│                           │
│ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░│  ← Small text (shimmer)
└───────────────────────────┘
```

**Usage:**
```swift
// Wrap any view
MyView()
    .skeleton(isLoading: isDataLoading)

// Or use directly
SkeletonView(isLoading: true) {
    ComplexView()
}
```

---

## 8. CandlestickChart

**Full OHLCV chart with gestures**

```swift
CandlestickChart(
    candles: candleData,
    currentPrice: 42_500,
    showGrid: true,
    showVolume: true
)
```

**Visual Representation:**

```
$43,500 ──────────────────────────────────── $43,500
        │     │ ▓
        │     ┃ ▓
$43,000 ┼─────┃─▓─────────────────────────── $43,000
        │     ┃ ▓         ┃
        │   ╭─╯ ░       ╭─╯ ░
$42,500 ┼───┃───░───────┃───░─ ─ ─ ─ ─ ─ ─ ─ $42,500  ← Current price (dashed)
        │   │   ░     ╭─╯   ░   ╭─╮
        │   │   ░     │     ░   │ ▓
$42,000 ┼───┼───░─────┼─────░───┼─▓───────── $42,000
        │   │   ░     │     ░   │ ▓
        │   │   ░     ╰─╮   ░   ╰─╯
$41,500 ──────────────────────────────────── $41,500
        09:00 10:00 11:00 12:00 13:00
```

**Legend:**
- `▓` = Green candle (close > open) - Bullish
- `░` = Red candle (close < open) - Bearish
- `│` or `┃` = Wick (high-low range)
- `─ ─` = Current price line (dashed, cyan)
- Grid lines (horizontal) at price levels

**Features:**

1. **Candle Components:**
   - **Body:** Rectangle from open to close (green/red)
   - **Wick:** Line from low to high
   - **Minimum body height:** 2pt (for doji candles)

2. **Price Scale (Right):**
   ```
   $43,500
   $43,000
   $42,500  ← 5 evenly spaced levels
   $42,000
   $41,500
   ```

3. **Grid Lines:**
   - Horizontal lines at price scale levels
   - Gray (opacity 20%)

4. **Current Price Line:**
   - Dashed cyan line
   - Updates in real-time

**Gestures:**

**Pinch to Zoom:**
```
Scale range: 0.5x → 3.0x
Default: 1.0x

0.5x = More candles visible (zoomed out)
3.0x = Fewer candles visible (zoomed in)
```

**Drag to Scroll:**
```
← Drag left:  Show older candles
→ Drag right: Show newer candles

Auto-clamps at data boundaries
```

**Example Interaction:**
```
[User pinches out]
Scale: 1.0x → 2.0x
Candles visible: 30 → 15
Candle width: 8pt → 16pt

[User drags left]
Offset: -120pt
Shows candles from 10 positions earlier
```

---

## Component Combinations

### Dashboard Header
```swift
GlassCard {
    HStack {
        VStack(alignment: .leading) {
            Text("HEAN")
                .font(.system(size: 28, weight: .bold))
            Text("Premium Trading")
                .font(.system(size: 14))
                .foregroundColor(.gray)
        }
        Spacer()
        StatusIndicator(status: .connected, latency: 32)
    }
    .padding()
}
```
**Visual:**
```
┌─────────────────────────────────────┐
│  HEAN              ● Connected (32ms) │
│  Premium Trading                    │
└─────────────────────────────────────┘
```

### Price Card with Trend
```swift
GlassCard {
    VStack(alignment: .leading) {
        HStack {
            Text("BTC/USDT")
            Spacer()
            PnLBadge(value: 3.45, format: .percent)
        }

        PriceTicker(
            symbol: "BTCUSDT",
            price: 42_350.75,
            changePercent: 3.45,
            size: .large
        )

        Sparkline(dataPoints: last24h)
    }
    .padding()
}
```
**Visual:**
```
┌───────────────────────────────────┐
│ BTC/USDT           ↗ +3.45%      │
│                                   │
│ BTCUSDT                           │
│ $42,350.75                        │
│ ↗ +3.45%                          │
│                                   │
│      ╱‾‾╲                         │
│    ╱′    ‾╲    ╱‾‾                │
│  ╱′        ╲  ╱                   │
│╱′           ╲╱                    │
└───────────────────────────────────┘
```

### Portfolio Summary
```swift
GlassCard {
    VStack(spacing: 16) {
        HStack {
            Text("Total P&L")
            Spacer()
            PnLBadge(value: 2_456.78, format: .dollar, size: .expanded)
        }

        Divider()

        HStack {
            Text("Risk State")
            Spacer()
            RiskBadge(state: .normal, variant: .compact)
        }
    }
    .padding()
}
```
**Visual:**
```
┌─────────────────────────────────────┐
│ Total P&L        ↗ +$2,456.78      │
│ ─────────────────────────────────── │
│ Risk State                   ✓     │
└─────────────────────────────────────┘
```

### Loading Card
```swift
GlassCard {
    VStack(alignment: .leading, spacing: 12) {
        RoundedRectangle(cornerRadius: 8)
            .fill(Color.gray.opacity(0.3))
            .frame(width: 120, height: 20)

        RoundedRectangle(cornerRadius: 8)
            .fill(Color.gray.opacity(0.3))
            .frame(width: 200, height: 40)
    }
    .padding()
}
.skeleton(isLoading: true)
```
**Visual:**
```
┌───────────────────────────┐
│ ▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░│  ← Shimmer animation
│                           │
│ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░│  ← Shimmer animation
└───────────────────────────┘
```

---

## Color Reference

### Background Colors
```
#0A0A0F  ██████  backgroundPrimary    (Darkest - main bg)
#12121A  ██████  backgroundSecondary  (Cards)
#1A1A24  ██████  backgroundTertiary   (Elevated surfaces)
```

### Accent Colors
```
#00D4FF  ██████  accentPrimary  (Cyan - primary actions)
#22C55E  ██████  success        (Green - bullish/positive)
#EF4444  ██████  error          (Red - bearish/negative)
#F59E0B  ██████  warning        (Orange - warnings)
```

### Text Colors
```
#FFFFFF  ██████  textPrimary    (White - main text)
#A1A1AA  ██████  textSecondary  (Light gray - labels)
#71717A  ██████  textTertiary   (Dark gray - hints)
```

---

## Animation Timeline

### PriceTicker Price Change
```
0ms     ──────────●──────────  Normal state

150ms   ──────────●──────────  Scale: 1.0 → 1.05 (pulse)

300ms   ──────────●──────────  Scale: 1.05 → 1.0 (return)
        ┌────────────────────┐
        │ Flash background   │  Green/red flash fades
        └────────────────────┘
```

### RiskBadge Pulse (Non-Normal)
```
0ms     ┌──────┐  Opacity: 30%
        │ SOFT │
        └──────┘

500ms   ┌──────┐  Opacity: 30% → 60%
        │ BRAKE│
        └──────┘

1000ms  ┌──────┐  Opacity: 60% → 30%
        │      │
        └──────┘

(Repeats indefinitely)
```

### StatusIndicator Reconnecting
```
0ms     ●  Ring scale: 1.0, opacity: 1.0

750ms   ◉  Ring scale: 1.0 → 1.8, opacity: 1.0 → 0.0

1500ms  ●  Ring scale: 1.0, opacity: 1.0 (reset)

(Repeats indefinitely)
```

### SkeletonView Shimmer
```
0ms     ░░░░░░░░░░░░  Shimmer at position 0%

750ms   ▒▒░░░░░░░░░░  Shimmer at position 50%

1500ms  ░░░░░░░░▒▒░░  Shimmer at position 100%

(Repeats indefinitely)
```

---

## File Reference

All components located in `/Users/macbookpro/Desktop/HEAN/ios/HEAN/DesignSystem/`

```
Components/
├── GlassCard.swift           ✓ Glassmorphism card
├── PriceTicker.swift         ✓ Real-time price display
├── Sparkline.swift           ✓ Mini trend chart
├── PnLBadge.swift            ✓ Profit/loss indicator
├── RiskBadge.swift           ✓ Risk state indicator
├── StatusIndicator.swift     ✓ Connection status
├── SkeletonView.swift        ✓ Loading placeholder
├── CandlestickChart.swift    ✓ OHLCV chart
└── ComponentShowcase.swift   ✓ Demo view

Colors/
└── AppColors.swift           ✓ Design token colors

Typography/
└── AppTypography.swift       ✓ Typography & spacing
```

---

## Quick Copy-Paste Examples

### Bullish Price Card
```swift
GlassCard {
    PriceTicker(
        symbol: "BTCUSDT",
        price: 42_350.75,
        changePercent: 3.45,
        size: .large
    )
}
```

### Bearish Price Card
```swift
GlassCard {
    PriceTicker(
        symbol: "ETHUSDT",
        price: 2_245.30,
        changePercent: -1.25,
        size: .large
    )
}
```

### Portfolio Summary
```swift
GlassCard {
    VStack(spacing: 16) {
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
    }
    .padding()
}
```

### Risk Dashboard
```swift
VStack(spacing: 16) {
    RiskBadge(state: .normal, variant: .expanded)

    HStack(spacing: 12) {
        StatusIndicator(status: .connected, latency: 45)
        Spacer()
        Text("All Systems Operational")
    }
}
```

### Chart View
```swift
GlassCard {
    VStack(alignment: .leading) {
        Text("BTC/USDT - 1H")
            .font(.headline)

        CandlestickChart(
            candles: candleData,
            currentPrice: 42_500,
            showGrid: true,
            showVolume: true
        )

        Text("Pinch to zoom • Drag to scroll")
            .font(.caption)
            .foregroundColor(.gray)
    }
    .padding()
}
```

---

**End of Visual Guide**
