# HEAN iOS Design DNA

**Version:** 1.0
**Last Updated:** 2026-01-31
**Target Platform:** iOS 17+, SwiftUI

---

## Design Principles

### Operator-Grade Interface Philosophy

1. **Clarity beats decoration** - Every pixel must serve decision-making
2. **Progressive disclosure** - Summary → drill-down on demand
3. **The Three Questions Test** - Every screen answers:
   - What is happening? (Current state)
   - Why? (Context and causation)
   - What do I do next? (Actionable next steps)

### Mobile-Specific Constraints

- **One-handed operation** - Critical actions within thumb reach
- **Glanceable information** - Key metrics readable in <1 second
- **Fault-tolerant** - Accidental taps must not execute destructive actions
- **Offline-aware** - Graceful degradation when connectivity drops

---

## 1. Color System

### Dark-First Palette

**Primary Background Hierarchy**
```swift
// Surface hierarchy (from deepest to highest)
Color.background.primary   = #0A0E14  // Base canvas
Color.background.secondary = #131820  // Elevated cards
Color.background.tertiary  = #1C2128  // Highest elevation (modals)
Color.background.overlay   = #000000.opacity(0.75)  // Sheet backgrounds
```

**Accent Colors** (Maximum 2 for focused attention)
```swift
Color.accent.primary   = #00D4FF  // Cyan - primary actions, links
Color.accent.secondary = #7C3AED  // Purple - secondary emphasis, badges
```

**Semantic Colors**
```swift
// Trading-specific semantics
Color.trading.long     = #10B981  // Green - buy/long positions
Color.trading.short    = #EF4444  // Red - sell/short positions
Color.trading.neutral  = #6B7280  // Gray - neutral/closed

// System states
Color.status.success   = #10B981  // Trade executed, system healthy
Color.status.warning   = #F59E0B  // Soft brake, caution states
Color.status.error     = #EF4444  // Hard stop, critical errors
Color.status.info      = #3B82F6  // Informational messages

// Risk Governor States
Color.risk.normal      = #10B981  // NORMAL - full trading
Color.risk.softBrake   = #F59E0B  // SOFT_BRAKE - reduced sizing
Color.risk.quarantine  = #F97316  // QUARANTINE - observe only
Color.risk.hardStop    = #EF4444  // HARD_STOP - all trading halted
```

**Text & Icon Hierarchy**
```swift
Color.text.primary     = #F9FAFB  // WCAG AAA on primary bg (18.2:1)
Color.text.secondary   = #9CA3AF  // Subheadings, labels (7.1:1)
Color.text.tertiary    = #6B7280  // Metadata, timestamps (4.8:1)
Color.text.disabled    = #4B5563  // Disabled state (3.2:1)

Color.icon.primary     = #F9FAFB
Color.icon.secondary   = #9CA3AF
Color.icon.accent      = #00D4FF
```

**Borders & Dividers**
```swift
Color.border.subtle    = #1F2937  // Card borders, dividers
Color.border.default   = #374151  // Interactive element borders
Color.border.emphasis  = #4B5563  // Focused state borders
```

**Gradients** (Use sparingly)
```swift
// Price increase glow (subtle, 10% opacity)
Gradient.priceUp = LinearGradient(
    colors: [Color.trading.long.opacity(0.1), Color.clear],
    startPoint: .leading, endPoint: .trailing
)

// Price decrease glow
Gradient.priceDown = LinearGradient(
    colors: [Color.trading.short.opacity(0.1), Color.clear],
    startPoint: .leading, endPoint: .trailing
)

// Glassmorphism backgrounds (use only for overlays)
Gradient.glass = LinearGradient(
    colors: [Color.white.opacity(0.05), Color.white.opacity(0.02)],
    startPoint: .topLeading, endPoint: .bottomTrailing
)
```

---

## 2. Typography

### Font Stack

**System Fonts (Recommended)**
```swift
// Default: SF Pro (Apple's system font)
// - SF Pro Display: headings, large text (20pt+)
// - SF Pro Text: body text (19pt and below)
// - SF Mono: prices, numbers, code

Font.system(.body, design: .default)      // General text
Font.system(.body, design: .monospaced)   // Prices, numbers
Font.system(.body, design: .rounded)      // Optional: friendly UI elements
```

**Type Scale** (8pt baseline grid)
```swift
// Headings
Font.heading.h1 = .system(size: 32, weight: .bold, design: .default)
Font.heading.h2 = .system(size: 24, weight: .semibold, design: .default)
Font.heading.h3 = .system(size: 20, weight: .semibold, design: .default)
Font.heading.h4 = .system(size: 16, weight: .semibold, design: .default)

// Body
Font.body.large  = .system(size: 17, weight: .regular)  // iOS default body
Font.body.medium = .system(size: 15, weight: .regular)
Font.body.small  = .system(size: 13, weight: .regular)

// Specialized
Font.caption.default = .system(size: 12, weight: .regular)
Font.caption.micro   = .system(size: 10, weight: .medium)   // Timestamps, badges

// Price displays (monospaced for stable alignment)
Font.price.hero  = .system(size: 40, weight: .bold, design: .monospaced)
Font.price.large = .system(size: 28, weight: .semibold, design: .monospaced)
Font.price.medium = .system(size: 20, weight: .medium, design: .monospaced)
Font.price.small = .system(size: 15, weight: .regular, design: .monospaced)
```

**Line Heights & Letter Spacing**
```swift
// Tight: headings, prices (1.1x)
// Normal: body text (1.4x - system default)
// Relaxed: long-form content (1.6x)

// Example
Text("BTC/USDT")
    .font(Font.heading.h2)
    .lineSpacing(1.1 * 24)  // 1.1x multiplier
    .tracking(0.02 * 24)    // 2% letter spacing
```

**Accessibility**
- Support Dynamic Type (sizeCategory environment value)
- Minimum touch target: 44x44pt (Apple HIG)
- Prices and critical numbers should support up to 3 size categories larger

---

## 3. Spacing & Layout

### Grid System (8pt baseline)

**Spacing Tokens**
```swift
enum Spacing {
    static let xxxs: CGFloat = 2   // Tight spacing in dense areas
    static let xxs: CGFloat = 4    // Icon-to-text gaps
    static let xs: CGFloat = 8     // Minimum breathing room
    static let sm: CGFloat = 12    // Related element spacing
    static let md: CGFloat = 16    // Default card padding
    static let lg: CGFloat = 24    // Section spacing
    static let xl: CGFloat = 32    // Major section breaks
    static let xxl: CGFloat = 48   // Screen-level spacing
}
```

**Margins & Padding**
```swift
// Screen-level margins
let screenMargin: CGFloat = 16  // Horizontal edge insets
let sectionSpacing: CGFloat = 24  // Between major sections

// Card padding
let cardPadding = EdgeInsets(
    top: Spacing.md,      // 16
    leading: Spacing.md,  // 16
    bottom: Spacing.md,   // 16
    trailing: Spacing.md  // 16
)

// List row padding
let rowPadding = EdgeInsets(
    top: Spacing.sm,      // 12
    leading: Spacing.md,  // 16
    bottom: Spacing.sm,   // 12
    trailing: Spacing.md  // 16
)
```

### Standard Layouts

**Card Pattern**
```swift
VStack(alignment: .leading, spacing: Spacing.md) {
    // Card content
}
.padding(cardPadding)
.background(Color.background.secondary)
.cornerRadius(12)
.shadow(color: .black.opacity(0.1), radius: 8, y: 2)
```

**Tab Bar Specs**
```swift
// Height: 49pt (Apple standard) + safeAreaInsets.bottom
// Icon size: 28x28pt (active), 24x24pt (inactive)
// Label: Font.caption.default
// Spacing icon-to-label: Spacing.xxs (4pt)
```

**Header Specs**
```swift
// Navigation bar height: 44pt + safeAreaInsets.top
// Large title: Font.heading.h1 (34pt, iOS 11+)
// Inline title: Font.heading.h4 (17pt)
```

**Safe Area Handling**
```swift
// Always respect safe area for primary content
// Use .ignoresSafeArea() only for full-bleed backgrounds

ScrollView {
    VStack {
        // Content here respects safe area by default
    }
}
.background(Color.background.primary.ignoresSafeArea())
```

**One-Handed Reach Zones** (iPhone 15 Pro reference)
```swift
// Green zone (easy reach): Bottom 0-300pt
// Yellow zone (stretch): 300-600pt
// Red zone (two-handed): 600pt+

// Critical actions (Buy/Sell) → Green zone
// Navigation, filters → Yellow zone
// Informational headers → Red zone acceptable
```

---

## 4. Component Specifications

### 4.1 PriceTicker

**Purpose:** Real-time price display with directional pulse effect.

**States:**
- `idle` - No recent change
- `increasing` - Price went up (pulse green)
- `decreasing` - Price went down (pulse red)

**Anatomy:**
```swift
HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
    VStack(alignment: .leading, spacing: 2) {
        Text("BTC/USDT")
            .font(Font.caption.default)
            .foregroundColor(Color.text.secondary)

        Text("$64,235.80")
            .font(Font.price.large)
            .foregroundColor(Color.text.primary)
            .monospacedDigit()  // iOS 15+ for tabular numbers
    }

    // Delta indicator
    HStack(spacing: 2) {
        Image(systemName: "arrow.up")
            .font(.system(size: 10, weight: .bold))
        Text("+2.34%")
            .font(Font.caption.default)
    }
    .foregroundColor(Color.trading.long)
    .padding(.horizontal, 6)
    .padding(.vertical, 4)
    .background(Color.trading.long.opacity(0.15))
    .cornerRadius(4)
}
.padding(Spacing.md)
.background(Color.background.secondary)
.cornerRadius(8)
.overlay(
    // Pulse ring on price change
    RoundedRectangle(cornerRadius: 8)
        .stroke(Color.trading.long.opacity(pulseOpacity), lineWidth: 2)
)
```

**Animation:**
- Pulse duration: 300ms (spring animation)
- Pulse opacity: 1.0 → 0.0
- Number transition: Odometer effect (0.2s ease-out)

**Haptic:** Light impact feedback on price change (if user preference enabled)

---

### 4.2 Sparkline

**Purpose:** Micro trend visualization (24h price history).

**Specs:**
- Height: 40pt (compact), 60pt (regular)
- Width: Flexible (min 80pt)
- Line weight: 1.5pt
- Gradient fill: Optional, 20% opacity below line

**Anatomy:**
```swift
// Using Swift Charts (iOS 16+)
Chart(priceHistory) { point in
    LineMark(
        x: .value("Time", point.timestamp),
        y: .value("Price", point.price)
    )
    .foregroundStyle(lineColor)
    .lineStyle(StrokeStyle(lineWidth: 1.5, lineCap: .round))
}
.chartXAxis(.hidden)
.chartYAxis(.hidden)
.frame(height: 40)
```

**Colors:**
- Positive trend: `Color.trading.long`
- Negative trend: `Color.trading.short`
- Neutral: `Color.text.tertiary`

---

### 4.3 CandlestickChart

**Purpose:** Main chart view for price action analysis.

**Specs:**
- Use TradingView's Lightweight Charts (WebView wrapper)
- Alternative: Build native with Swift Charts (iOS 16+)
- Pinch-to-zoom, two-finger pan
- Candlestick width: 8pt (1h), 4pt (15m), 12pt (1d)

**Touch Interactions:**
- Single tap: Show crosshair + price tooltip
- Long press: Scrub through time
- Pinch: Zoom timeframe
- Double tap: Reset zoom

**Overlay Elements:**
- Current price line (dashed, accent color)
- Volume bars (bottom 25% of chart area)
- Moving averages (optional, toggled)

---

### 4.4 OrderBook Visualization

**Purpose:** Live bid/ask depth display.

**Layout:**
```swift
VStack(spacing: 0) {
    // Asks (red, top half)
    ForEach(asks.prefix(10).reversed()) { level in
        OrderBookRow(
            price: level.price,
            size: level.size,
            side: .ask,
            depth: level.depth / maxDepth
        )
    }

    // Spread indicator
    HStack {
        Text("Spread")
            .font(Font.caption.default)
            .foregroundColor(Color.text.tertiary)
        Spacer()
        Text("$0.50 (0.01%)")
            .font(Font.caption.default.monospacedDigit())
            .foregroundColor(Color.text.secondary)
    }
    .padding(.horizontal, Spacing.sm)
    .padding(.vertical, Spacing.xs)
    .background(Color.background.tertiary)

    // Bids (green, bottom half)
    ForEach(bids.prefix(10)) { level in
        OrderBookRow(
            price: level.price,
            size: level.size,
            side: .bid,
            depth: level.depth / maxDepth
        )
    }
}
```

**OrderBookRow Anatomy:**
```swift
ZStack(alignment: .leading) {
    // Depth bar (background)
    GeometryReader { geo in
        Rectangle()
            .fill(barColor.opacity(0.2))
            .frame(width: geo.size.width * depth)
    }

    // Price + Size (foreground)
    HStack {
        Text(price.formatted(.number.precision(.fractionLength(2))))
            .font(Font.body.small.monospacedDigit())
            .foregroundColor(priceColor)

        Spacer()

        Text(size.formatted(.number.precision(.fractionLength(4))))
            .font(Font.body.small.monospacedDigit())
            .foregroundColor(Color.text.secondary)
    }
    .padding(.horizontal, Spacing.sm)
    .padding(.vertical, 4)
}
.frame(height: 24)
```

**Colors:**
- Ask bar: `Color.trading.short.opacity(0.2)`
- Bid bar: `Color.trading.long.opacity(0.2)`
- Price text: Primary for asks, primary for bids
- Size text: Secondary for both

---

### 4.5 PositionRow

**Purpose:** Display open position in list.

**Anatomy:**
```swift
HStack(spacing: Spacing.md) {
    // Left: Symbol + Side
    VStack(alignment: .leading, spacing: 2) {
        Text("BTC/USDT")
            .font(Font.body.medium.weight(.semibold))
            .foregroundColor(Color.text.primary)

        HStack(spacing: 4) {
            Text(position.side.rawValue.uppercased())  // "LONG" or "SHORT"
                .font(Font.caption.default.weight(.medium))
                .foregroundColor(sideColor)

            Text("\(position.leverage)x")
                .font(Font.caption.default)
                .foregroundColor(Color.text.tertiary)
        }
    }

    Spacer()

    // Right: PnL + Entry Price
    VStack(alignment: .trailing, spacing: 2) {
        Text(position.unrealizedPnl.formatted(.currency(code: "USD")))
            .font(Font.body.medium.weight(.semibold).monospacedDigit())
            .foregroundColor(pnlColor)

        Text("Entry: \(position.entryPrice.formatted())")
            .font(Font.caption.default.monospacedDigit())
            .foregroundColor(Color.text.tertiary)
    }
}
.padding(rowPadding)
.background(Color.background.secondary)
.cornerRadius(8)
.overlay(
    RoundedRectangle(cornerRadius: 8)
        .strokeBorder(sideColor.opacity(0.3), lineWidth: 1)
)
```

**Swipe Actions:**
- Leading: Close position (destructive, requires confirmation)
- Trailing: Edit stop loss / take profit

---

### 4.6 PnLChip

**Purpose:** Compact profit/loss indicator.

**Variants:**
- `inline` - Horizontal layout (icon + text)
- `stacked` - Vertical layout (text over icon)

**Anatomy:**
```swift
HStack(spacing: 4) {
    Image(systemName: pnl >= 0 ? "arrow.up.right" : "arrow.down.right")
        .font(.system(size: 10, weight: .bold))

    Text(pnl.formatted(.currency(code: "USD")))
        .font(Font.caption.default.monospacedDigit())
}
.foregroundColor(pnlColor)
.padding(.horizontal, 8)
.padding(.vertical, 4)
.background(pnlColor.opacity(0.15))
.cornerRadius(6)
```

**Colors:**
- Positive: `Color.trading.long`
- Negative: `Color.trading.short`
- Zero: `Color.text.tertiary`

---

### 4.7 RiskBadge

**Purpose:** Display Risk Governor state.

**States:**
- `NORMAL` - Green, no icon
- `SOFT_BRAKE` - Orange, warning icon
- `QUARANTINE` - Deep orange, alert icon
- `HARD_STOP` - Red, stop icon

**Anatomy:**
```swift
HStack(spacing: 6) {
    if let icon = stateIcon {
        Image(systemName: icon)
            .font(.system(size: 12, weight: .bold))
    }

    Text(state.rawValue)
        .font(Font.caption.default.weight(.semibold))
        .textCase(.uppercase)
}
.foregroundColor(.white)
.padding(.horizontal, 10)
.padding(.vertical, 6)
.background(stateColor)
.cornerRadius(8)
```

**Icon Mapping:**
```swift
switch state {
case .normal:      icon = nil
case .softBrake:   icon = "exclamationmark.triangle.fill"
case .quarantine:  icon = "eye.fill"
case .hardStop:    icon = "hand.raised.fill"
}
```

**Haptic:** Notification feedback on state change (warning/error)

---

### 4.8 StatusIndicator

**Purpose:** WebSocket connection status.

**States:**
- `connected` - Green dot, no text (default)
- `reconnecting` - Orange dot, animated pulse
- `disconnected` - Red dot, "Offline" label

**Anatomy:**
```swift
HStack(spacing: 6) {
    Circle()
        .fill(statusColor)
        .frame(width: 8, height: 8)
        .overlay(
            // Pulse ring when reconnecting
            Circle()
                .stroke(statusColor.opacity(pulseOpacity), lineWidth: 2)
                .scaleEffect(pulseScale)
        )

    if showLabel {
        Text(statusText)
            .font(Font.caption.default)
            .foregroundColor(Color.text.secondary)
    }
}
```

**Animation (reconnecting state):**
- Pulse duration: 1.5s (infinite repeat)
- Scale: 1.0 → 1.6
- Opacity: 0.8 → 0.0

---

## 5. Motion Design Guide

### Duration Tokens

```swift
enum Duration {
    static let instant: TimeInterval = 0.1   // 100ms - checkbox, switch toggle
    static let fast: TimeInterval = 0.15     // 150ms - button press feedback
    static let normal: TimeInterval = 0.25   // 250ms - default transitions
    static let slow: TimeInterval = 0.4      // 400ms - sheet presentation
    static let deliberate: TimeInterval = 0.6 // 600ms - important confirmations
}
```

### Easing Curves

**When to use:**
- **Spring (default)** - Natural, physically-based motion
  ```swift
  .animation(.spring(response: 0.3, dampingFraction: 0.7), value: state)
  ```
  Use for: User-initiated actions, interactive elements

- **EaseInOut** - Smooth acceleration/deceleration
  ```swift
  .animation(.easeInOut(duration: Duration.normal), value: state)
  ```
  Use for: Sheet presentations, modal transitions

- **EaseOut** - Quick start, gentle landing
  ```swift
  .animation(.easeOut(duration: Duration.fast), value: state)
  ```
  Use for: Appearing elements (toasts, tooltips)

- **Linear** - Constant velocity
  ```swift
  .animation(.linear(duration: 1.5).repeatForever(autoreverses: false), value: state)
  ```
  Use for: Loading spinners, infinite loops

### Specific Animations

#### Skeleton Shimmer (Loading State)

```swift
// Implementation using SwiftUI
struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geo in
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0),
                            Color.white.opacity(0.3),
                            Color.white.opacity(0)
                        ],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(width: geo.size.width * 2)
                    .offset(x: -geo.size.width + phase * geo.size.width * 2)
                }
                .allowsHitTesting(false)
            )
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    phase = 1
                }
            }
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

// Usage
RoundedRectangle(cornerRadius: 8)
    .fill(Color.background.secondary)
    .frame(height: 60)
    .shimmer()
```

**Timing:**
- Shimmer duration: 1.5s (linear, infinite)
- Skeleton placeholder: Use .redacted(reason: .placeholder)

---

#### Price Pulse Effect

```swift
struct PricePulseModifier: ViewModifier {
    let direction: PriceDirection  // .up or .down
    @State private var opacity: Double = 1.0
    @State private var scale: CGFloat = 1.0

    func body(content: Content) -> some View {
        content
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(pulseColor.opacity(opacity), lineWidth: 2)
                    .scaleEffect(scale)
            )
            .onChange(of: direction) { _ in
                // Trigger pulse
                opacity = 1.0
                scale = 1.0

                withAnimation(.easeOut(duration: 0.3)) {
                    opacity = 0.0
                    scale = 1.1
                }
            }
    }

    var pulseColor: Color {
        direction == .up ? Color.trading.long : Color.trading.short
    }
}
```

**Timing:**
- Pulse duration: 300ms (ease-out)
- Scale: 1.0 → 1.1
- Opacity: 1.0 → 0.0
- Trigger: On price change (debounced to 100ms to avoid spam)

---

#### Success/Error Feedback

**Toast Notification:**
```swift
struct ToastView: View {
    let message: String
    let type: ToastType  // .success, .error, .info
    @State private var offset: CGFloat = -100
    @State private var opacity: Double = 0

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: type.icon)
                .font(.system(size: 16, weight: .semibold))

            Text(message)
                .font(Font.body.medium)
        }
        .foregroundColor(.white)
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.sm)
        .background(type.color)
        .cornerRadius(10)
        .shadow(color: .black.opacity(0.2), radius: 8, y: 4)
        .offset(y: offset)
        .opacity(opacity)
        .onAppear {
            // Slide in from top
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                offset = 0
                opacity = 1
            }

            // Auto-dismiss after 3s
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                withAnimation(.easeIn(duration: 0.2)) {
                    offset = -100
                    opacity = 0
                }
            }
        }
    }
}
```

**Timing:**
- Appear: 300ms (spring)
- Display: 3s
- Dismiss: 200ms (ease-in)

---

#### Pull-to-Refresh

Use native `refreshable` modifier (iOS 15+):
```swift
List {
    // Content
}
.refreshable {
    await refreshData()
}
```

**Customization (if needed):**
- Trigger threshold: 80pt drag distance
- Spinner: System default (recommended)
- Haptic: Impact feedback at trigger point

---

### Haptic Feedback Mapping

```swift
enum HapticFeedback {
    case impact(UIImpactFeedbackGenerator.FeedbackStyle)
    case notification(UINotificationFeedbackGenerator.FeedbackType)
    case selection

    func trigger() {
        switch self {
        case .impact(let style):
            UIImpactFeedbackGenerator(style: style).impactOccurred()
        case .notification(let type):
            UINotificationFeedbackGenerator().notificationOccurred(type)
        case .selection:
            UISelectionFeedbackGenerator().selectionChanged()
        }
    }
}
```

**Action → Haptic Mapping:**

| Action | Haptic Type | When |
|--------|-------------|------|
| Order placed | `notification(.success)` | Order confirmed by exchange |
| Order failed | `notification(.error)` | Order rejected |
| Risk state change | `notification(.warning)` | Entering SOFT_BRAKE/QUARANTINE |
| Position closed | `notification(.success)` | Position fully exited |
| Price update | `impact(.light)` | Significant price movement (optional, user pref) |
| Tab switch | `selection` | Bottom tab bar navigation |
| Button press | `impact(.medium)` | Primary action buttons |
| Swipe action | `impact(.light)` | Reveal swipe actions |
| Pull-to-refresh | `impact(.medium)` | At trigger threshold |

**User Preference:**
- Settings toggle: "Haptic Feedback" (on/off)
- Granular control: "Price Updates", "Trading Actions", "Navigation"

---

## 6. States & Patterns

### Empty States

**Anatomy:**
```swift
VStack(spacing: Spacing.lg) {
    // Icon
    Image(systemName: "chart.line.uptrend.xyaxis")
        .font(.system(size: 48, weight: .light))
        .foregroundColor(Color.icon.secondary)

    // Heading
    Text("No Positions Open")
        .font(Font.heading.h3)
        .foregroundColor(Color.text.primary)

    // Explanation
    Text("Positions will appear here when you enter trades. Monitor your P&L and manage risk in real-time.")
        .font(Font.body.medium)
        .foregroundColor(Color.text.secondary)
        .multilineTextAlignment(.center)
        .padding(.horizontal, Spacing.xl)

    // Action (if applicable)
    Button(action: { /* Navigate to trading */ }) {
        HStack {
            Image(systemName: "plus.circle.fill")
            Text("Open New Position")
        }
        .font(Font.body.medium.weight(.semibold))
        .foregroundColor(.white)
        .padding(.horizontal, Spacing.lg)
        .padding(.vertical, Spacing.sm)
        .background(Color.accent.primary)
        .cornerRadius(10)
    }
}
.frame(maxWidth: .infinity, maxHeight: .infinity)
.background(Color.background.primary)
```

**Principles:**
- Icon: Relevant to missing content (not generic)
- Heading: What's missing (present tense, user-focused)
- Explanation: Why it's empty + what would fill it
- Action: Next step (only if user can act)

---

### Error States

**Anatomy:**
```swift
VStack(spacing: Spacing.md) {
    // Error icon
    Image(systemName: "exclamationmark.triangle.fill")
        .font(.system(size: 40))
        .foregroundColor(Color.status.error)

    // Error title
    Text("Connection Lost")
        .font(Font.heading.h3)
        .foregroundColor(Color.text.primary)

    // Error details + action
    Text("Unable to connect to trading server. Check your network connection and try again.")
        .font(Font.body.medium)
        .foregroundColor(Color.text.secondary)
        .multilineTextAlignment(.center)
        .padding(.horizontal, Spacing.xl)

    // Retry button
    Button("Retry Connection") {
        // Retry logic
    }
    .buttonStyle(PrimaryButtonStyle())
}
.padding(Spacing.xl)
.background(Color.background.secondary)
.cornerRadius(12)
```

**Error Message Formula:**
- What went wrong: "Connection Lost"
- Why it matters: "Unable to execute trades"
- What to do: "Check network and retry"
- Never blame the user ("You lost connection" → "Connection lost")

**Common Error States:**
- Network error: Retry button
- Authentication error: Re-login prompt
- Invalid input: Inline validation with correction hint
- Rate limit: Countdown timer + retry

---

### Loading States

**Skeleton Pattern (Preferred):**
```swift
VStack(alignment: .leading, spacing: Spacing.sm) {
    // Symbol skeleton
    RoundedRectangle(cornerRadius: 4)
        .fill(Color.background.tertiary)
        .frame(width: 80, height: 16)
        .shimmer()

    // Price skeleton
    RoundedRectangle(cornerRadius: 4)
        .fill(Color.background.tertiary)
        .frame(width: 120, height: 24)
        .shimmer()

    // Delta skeleton
    RoundedRectangle(cornerRadius: 4)
        .fill(Color.background.tertiary)
        .frame(width: 60, height: 14)
        .shimmer()
}
.padding(Spacing.md)
.redacted(reason: .placeholder)  // System skeleton + shimmer
```

**Spinner Pattern (Fallback):**
```swift
ProgressView()
    .progressViewStyle(CircularProgressViewStyle(tint: Color.accent.primary))
    .scaleEffect(1.2)
```

**When to use:**
- Skeleton: List items, cards, structured content (known layout)
- Spinner: Full-screen loading, unknown duration
- Pull-to-refresh: Native refreshable modifier
- Inline loading: Small spinner next to action (e.g., "Placing order...")

---

### Toast Notifications

**Duration:**
- Success: 3s
- Error: 5s (longer to allow reading)
- Info: 3s

**Position:**
- Top: Preferred (doesn't obscure primary content)
- Bottom: Above tab bar (if top conflicts with nav)

**Max Simultaneous:** 1 (queue subsequent toasts)

**Dismissal:**
- Auto-dismiss after duration
- Swipe up to dismiss early
- Tap to dismiss (optional)

---

## 7. Accessibility

### Minimum Requirements

**Color Contrast:**
- Text on background: WCAG AA (4.5:1 for normal, 3:1 for large)
- Critical elements: WCAG AAA preferred (7:1)
- Test tool: Use Xcode Accessibility Inspector

**Touch Targets:**
- Minimum: 44x44pt (Apple HIG)
- Preferred: 48x48pt for primary actions
- Spacing between targets: 8pt minimum

**Dynamic Type:**
- Support up to XXXL size category
- Test at .accessibility5 (largest)
- Prices and critical numbers scale gracefully

**VoiceOver:**
- All interactive elements labeled
- Custom controls use `.accessibilityLabel()` and `.accessibilityHint()`
- Price updates announced: `.accessibilityLiveRegion(.polite)`
- Order confirmations: `.accessibilityLiveRegion(.assertive)`

**Reduce Motion:**
```swift
@Environment(\.accessibilityReduceMotion) var reduceMotion

var animation: Animation {
    reduceMotion ? .none : .spring(response: 0.3, dampingFraction: 0.7)
}
```

**Color Blindness:**
- Don't rely on color alone (use icons + text)
- Red/green: Always paired with up/down arrows
- Test with: Color Blindness Simulator (Xcode Instruments)

---

## 8. Dark Mode (Primary) & Light Mode (Optional)

**Current Spec: Dark-first only** (Light mode deferred to v2)

If implementing light mode:
```swift
// Adaptive colors
Color.background.primary = Color(light: #FFFFFF, dark: #0A0E14)
Color.text.primary = Color(light: #111827, dark: #F9FAFB)

// Semantic colors remain constant
Color.trading.long = #10B981  // Same in both modes
Color.trading.short = #EF4444  // Same in both modes
```

**Testing:** Use `@Environment(\.colorScheme)` to detect mode

---

## 9. Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Color system tokens in `Colors.swift`
- [ ] Typography tokens in `Typography.swift`
- [ ] Spacing tokens in `Layout.swift`
- [ ] Duration/easing tokens in `Animation.swift`
- [ ] Base component library: Card, Button, Badge

### Phase 2: Trading Components (Week 2)
- [ ] PriceTicker with pulse effect
- [ ] Sparkline chart
- [ ] OrderBook visualization
- [ ] PositionRow with swipe actions
- [ ] PnLChip
- [ ] RiskBadge
- [ ] StatusIndicator

### Phase 3: Advanced Features (Week 3)
- [ ] Candlestick chart (TradingView wrapper or native)
- [ ] Skeleton loading states
- [ ] Toast notification system
- [ ] Haptic feedback integration
- [ ] Pull-to-refresh

### Phase 4: Polish (Week 4)
- [ ] Empty states for all screens
- [ ] Error states with retry logic
- [ ] Accessibility audit (VoiceOver, Dynamic Type)
- [ ] Performance optimization (60fps target)
- [ ] User preference for haptics/animations

---

## 10. Design Tokens Export

For developers, export tokens to Swift:

```swift
// DesignTokens.swift
import SwiftUI

enum DesignTokens {
    enum Colors {
        // Background
        static let backgroundPrimary = Color(hex: "0A0E14")
        static let backgroundSecondary = Color(hex: "131820")
        static let backgroundTertiary = Color(hex: "1C2128")

        // Accent
        static let accentPrimary = Color(hex: "00D4FF")
        static let accentSecondary = Color(hex: "7C3AED")

        // Trading
        static let tradingLong = Color(hex: "10B981")
        static let tradingShort = Color(hex: "EF4444")
        static let tradingNeutral = Color(hex: "6B7280")

        // Status
        static let statusSuccess = Color(hex: "10B981")
        static let statusWarning = Color(hex: "F59E0B")
        static let statusError = Color(hex: "EF4444")
        static let statusInfo = Color(hex: "3B82F6")

        // Text
        static let textPrimary = Color(hex: "F9FAFB")
        static let textSecondary = Color(hex: "9CA3AF")
        static let textTertiary = Color(hex: "6B7280")
        static let textDisabled = Color(hex: "4B5563")

        // Border
        static let borderSubtle = Color(hex: "1F2937")
        static let borderDefault = Color(hex: "374151")
        static let borderEmphasis = Color(hex: "4B5563")
    }

    enum Typography {
        static let headingH1 = Font.system(size: 32, weight: .bold)
        static let headingH2 = Font.system(size: 24, weight: .semibold)
        static let headingH3 = Font.system(size: 20, weight: .semibold)
        static let headingH4 = Font.system(size: 16, weight: .semibold)

        static let bodyLarge = Font.system(size: 17, weight: .regular)
        static let bodyMedium = Font.system(size: 15, weight: .regular)
        static let bodySmall = Font.system(size: 13, weight: .regular)

        static let captionDefault = Font.system(size: 12, weight: .regular)
        static let captionMicro = Font.system(size: 10, weight: .medium)

        static let priceHero = Font.system(size: 40, weight: .bold, design: .monospaced)
        static let priceLarge = Font.system(size: 28, weight: .semibold, design: .monospaced)
        static let priceMedium = Font.system(size: 20, weight: .medium, design: .monospaced)
        static let priceSmall = Font.system(size: 15, weight: .regular, design: .monospaced)
    }

    enum Spacing {
        static let xxxs: CGFloat = 2
        static let xxs: CGFloat = 4
        static let xs: CGFloat = 8
        static let sm: CGFloat = 12
        static let md: CGFloat = 16
        static let lg: CGFloat = 24
        static let xl: CGFloat = 32
        static let xxl: CGFloat = 48
    }

    enum Duration {
        static let instant: TimeInterval = 0.1
        static let fast: TimeInterval = 0.15
        static let normal: TimeInterval = 0.25
        static let slow: TimeInterval = 0.4
        static let deliberate: TimeInterval = 0.6
    }
}

// Helper for hex colors
extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        scanner.currentIndex = hex.startIndex
        var rgbValue: UInt64 = 0
        scanner.scanHexInt64(&rgbValue)

        let r = Double((rgbValue & 0xFF0000) >> 16) / 255.0
        let g = Double((rgbValue & 0x00FF00) >> 8) / 255.0
        let b = Double(rgbValue & 0x0000FF) / 255.0

        self.init(red: r, green: g, blue: b)
    }
}
```

---

## 11. References & Inspiration

**Mobile Trading Apps:**
- Robinhood (micro-interactions, color psychology)
- TradingView Mobile (chart interactions, pinch-to-zoom)
- Coinbase (large type, clear hierarchy)

**Design Systems:**
- Apple Human Interface Guidelines (iOS 17)
- Material Design 3 (color system inspiration)

**Motion Libraries:**
- Lottie 4.3.0+ (for pre-built animations)
- SwiftUI Spring Animations (native, performant)

**Tools:**
- SF Symbols 5 (system icons)
- Xcode Accessibility Inspector (testing)
- Swift Charts (iOS 16+ native charting)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-31 | Initial Design DNA created |

---

**Document Owner:** HEAN Product Design
**Last Review:** 2026-01-31
**Next Review:** 2026-02-28

---

## Appendix: Quick Reference Card

**Colors:**
```
BG Primary: #0A0E14 | Accent: #00D4FF
Long: #10B981 | Short: #EF4444
Success: #10B981 | Warning: #F59E0B | Error: #EF4444
```

**Spacing:**
```
xs(8) sm(12) md(16) lg(24) xl(32) xxl(48)
```

**Typography:**
```
H1(32/bold) H2(24/semibold) H3(20/semibold)
Body(17/15/13) Caption(12/10)
Price(40/28/20/15 monospaced)
```

**Animations:**
```
instant(100ms) fast(150ms) normal(250ms) slow(400ms)
Default: .spring(response: 0.3, dampingFraction: 0.7)
```

**Touch Targets:**
```
Minimum: 44x44pt | Preferred: 48x48pt
```
