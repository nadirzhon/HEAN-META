# HEAN Platform Design Plan
## Web Dashboard + iOS App Redesign

**Version:** 2.0
**Date:** 2026-02-15
**Status:** Design Specification
**Inspiration:** Clawdbot.com professional trading aesthetic

---

## DESIGN PHILOSOPHY

### Core Principles

**1. Information Density with Clarity**
- Show maximum relevant data without clutter
- Use progressive disclosure (summary → detail on demand)
- Real-time updates without overwhelming the user

**2. Dark-First Design**
- Primary: Near-black (#0A0E27) with subtle gradients
- Accent: Electric blue (#00D9FF) for data, amber (#FFB800) for alerts
- Profit/Loss: Green (#00FF88) / Red (#FF3366) with semantic meaning

**3. Professional Trading Aesthetic**
- Bloomberg Terminal meets modern web design
- Data visualization as first-class citizen
- Zero marketing fluff, pure functionality

**4. Cross-Platform Consistency**
- Shared design system between web and iOS
- Same mental model, platform-appropriate patterns
- Unified color palette, typography, spacing

---

## COLOR SYSTEM

### Base Palette

```css
/* Dark Theme (Primary) */
--bg-primary: #0A0E27;        /* Deep space blue */
--bg-secondary: #131836;      /* Slightly lighter panels */
--bg-tertiary: #1C2347;       /* Hover states */
--bg-elevated: #242B5C;       /* Modals, dropdowns */

/* Text */
--text-primary: #E8ECFF;      /* High contrast white-blue */
--text-secondary: #9BA3C7;    /* Muted labels */
--text-tertiary: #64698A;     /* Disabled, timestamps */

/* Accent Colors */
--accent-primary: #00D9FF;    /* Electric blue - data, links */
--accent-secondary: #A78BFA;  /* Purple - AI/ML indicators */
--accent-tertiary: #FFB800;   /* Amber - warnings */

/* Semantic */
--success: #00FF88;           /* Profit, long positions */
--danger: #FF3366;            /* Loss, short positions */
--warning: #FFB800;           /* Alerts, risk */
--info: #00D9FF;              /* Neutral info */

/* Borders */
--border-subtle: rgba(155, 163, 199, 0.1);
--border-medium: rgba(155, 163, 199, 0.2);
--border-strong: rgba(155, 163, 199, 0.3);

/* Overlays */
--overlay-light: rgba(10, 14, 39, 0.8);
--overlay-heavy: rgba(10, 14, 39, 0.95);

/* Glass Effect */
--glass-bg: rgba(36, 43, 92, 0.6);
--glass-border: rgba(0, 217, 255, 0.2);
--glass-blur: blur(12px);
```

### iOS Semantic Colors

```swift
// iOS Color Extension (DesignSystem/Colors.swift)
extension Color {
    // Backgrounds
    static let bgPrimary = Color(hex: "0A0E27")
    static let bgSecondary = Color(hex: "131836")
    static let bgTertiary = Color(hex: "1C2347")
    static let bgElevated = Color(hex: "242B5C")

    // Text
    static let textPrimary = Color(hex: "E8ECFF")
    static let textSecondary = Color(hex: "9BA3C7")
    static let textTertiary = Color(hex: "64698A")

    // Accents
    static let accentPrimary = Color(hex: "00D9FF")
    static let accentSecondary = Color(hex: "A78BFA")
    static let accentWarning = Color(hex: "FFB800")

    // Semantic
    static let profit = Color(hex: "00FF88")
    static let loss = Color(hex: "FF3366")
    static let warning = Color(hex: "FFB800")
    static let info = Color(hex: "00D9FF")
}
```

---

## TYPOGRAPHY

### Web (Next.js + TailwindCSS)

```css
/* Font Stack */
--font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
--font-mono: "JetBrains Mono", "Fira Code", "Consolas", monospace;
--font-display: "Orbitron", "Rajdhani", sans-serif; /* For large numbers */

/* Scale */
--text-xs: 0.75rem;    /* 12px - timestamps, meta */
--text-sm: 0.875rem;   /* 14px - labels, secondary */
--text-base: 1rem;     /* 16px - body */
--text-lg: 1.125rem;   /* 18px - section headers */
--text-xl: 1.25rem;    /* 20px - card titles */
--text-2xl: 1.5rem;    /* 24px - page titles */
--text-3xl: 1.875rem;  /* 30px - hero numbers */
--text-4xl: 2.25rem;   /* 36px - large stats */

/* Weights */
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
```

### iOS

```swift
// Typography Extension (DesignSystem/Typography.swift)
extension Font {
    // System (San Francisco)
    static let bodySmall = Font.system(size: 12, weight: .regular)
    static let bodyRegular = Font.system(size: 14, weight: .regular)
    static let bodyMedium = Font.system(size: 14, weight: .medium)
    static let bodySemibold = Font.system(size: 14, weight: .semibold)

    // Headers
    static let heading1 = Font.system(size: 28, weight: .bold)
    static let heading2 = Font.system(size: 22, weight: .semibold)
    static let heading3 = Font.system(size: 18, weight: .semibold)

    // Monospace (for numbers, prices)
    static let mono = Font.system(.body, design: .monospaced)
    static let monoLarge = Font.system(size: 24, design: .monospaced).weight(.bold)

    // Display (for large numbers)
    static let displayHero = Font.system(size: 48, design: .rounded).weight(.heavy)
}
```

---

## WEB DASHBOARD ARCHITECTURE

### Tech Stack

**Framework:** Next.js 14 (App Router)
**Styling:** TailwindCSS + CSS Variables
**Charts:** TradingView Lightweight Charts + Recharts
**State:** Zustand (lightweight, faster than Redux)
**Real-Time:** WebSocket + SWR for polling fallback
**Deployment:** Vercel (edge functions for low latency)

### Page Structure

```
/dashboard
├── / (Overview)
├── /live (Real-time trading)
├── /strategies (Strategy performance)
├── /physics (Market thermodynamics)
├── /execution (Order flow)
├── /risk (Risk dashboard)
├── /analytics (Deep metrics)
└── /settings
```

---

## WEB DASHBOARD SCREENS

### 1. Overview Dashboard (/)

**Layout:** 4-column grid with responsive collapse to 2/1 column

```
┌─────────────────────────────────────────────────────────┐
│  HEAN  [LIVE]  $12,456.78 (+8.2%)  [⚡ LAPLACE MODE]   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ EQUITY       │ │ DAILY PNL    │ │ OPEN POSITIONS│   │
│  │ $12,456.78   │ │ +$456.78     │ │ 7 / 10        │   │
│  │ ▲ 8.2%       │ │ ▲ 8.2%       │ │ 70% util      │   │
│  │ [sparkline]  │ │ [sparkline]  │ │ [mini chart]  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ MARKET PHASE: LAPLACE (SSD Resonance Detected)  │   │
│  │ ┌──────────┬──────────┬──────────┬──────────┐   │   │
│  │ │Temp: 345 │Entropy:.3│Phase: ICE│Resonance│   │   │
│  │ │[meter]   │[meter]   │[badge]   │ 0.87    │   │   │
│  │ └──────────┴──────────┴──────────┴──────────┘   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────┐ ┌──────────────────────────────┐  │
│  │ TOP STRATEGIES  │ │ ORACLE SIGNALS              │  │
│  │ ─────────────── │ │ ──────────────────────────  │  │
│  │ ImpulseEngine   │ │ TCN: 60% ██████░░░░ (VAPOR) │  │
│  │   +$234 (52%)   │ │ FinBERT: 15% ███░░░░        │  │
│  │ FundingHarv.    │ │ Ollama: 15% ███░░░░         │  │
│  │   +$123 (27%)   │ │ Brain: 10% ██░░░░░░         │  │
│  │ BasisArb        │ │ ─────────────────────────   │  │
│  │   +$99 (21%)    │ │ Combined: 0.72 CONFIDENCE  │  │
│  └─────────────────┘ └──────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ACTIVITY FEED                                    │   │
│  │ 14:32:15 ⚡ LAPLACE MODE activated (BTC)        │   │
│  │ 14:31:42 📊 Position opened: ETHUSDT long       │   │
│  │ 14:30:18 💰 Maker order filled: +$12.34 rebate │   │
│  │ 14:29:03 🎯 Signal: SOLUSDT (impulse_engine)   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Live Badge:** Pulsing green dot when system is running
- **Phase Indicator:** Large banner showing SSD mode (Normal/Laplace/Silent)
- **Sparklines:** Inline mini-charts for equity/PnL trends (last 24h)
- **Oracle Weights:** Real-time visual representation of dynamic weights
- **Activity Feed:** WebSocket-powered real-time event stream

---

### 2. Live Trading View (/live)

**Layout:** Full-width chart with side panels

```
┌─────────────────────────────────────────────────────────┐
│  BTCUSDT  $43,256.78  ▲ $234.56 (0.54%)  [1m 5m 15m 1h]│
├───────────────────────────────────┬─────────────────────┤
│                                    │ OPEN POSITIONS (3) │
│   [TRADINGVIEW CHART - FULL SIZE] │ ─────────────────  │
│                                    │ ETHUSDT LONG       │
│   Candlesticks + Volume            │ Entry: $2,345.67   │
│   Physics overlay:                 │ Size: 5.2 ETH      │
│   - Temp line (orange)             │ PnL: +$123.45      │
│   - Entropy bands (blue)           │ TP: $2,450 (4.4%) │
│   - Phase zones (background)       │                    │
│                                    │ SOLUSDT LONG       │
│   Signal markers:                  │ Entry: $98.76      │
│   - 🟢 Buy signals                 │ Size: 120 SOL      │
│   - 🔴 Sell signals                │ PnL: +$45.67       │
│   - ⚡ SSD Laplace events          │ TP: $102 (3.3%)   │
│                                    │                    │
│   Order book depth:                │ BTCUSDT SHORT      │
│   - Iceberg detection overlay      │ Entry: $43,200     │
│                                    │ Size: 0.5 BTC      │
│                                    │ PnL: +$28.90       │
│                                    │ TP: $42,800 (0.9%)│
│                                    │                    │
│                                    │ ─────────────────  │
│                                    │ [CLOSE ALL]        │
├───────────────────────────────────┴─────────────────────┤
│ ORDER BOOK           │ RECENT TRADES  │ MY ORDERS (2)    │
│ ASK                  │ 14:32:45 BUY   │ PENDING          │
│ 43,258.50 ████░░ 2.3│ 0.5 @ 43,256   │ ETHUSDT BUY      │
│ 43,257.20 ███░░░ 1.8│ 14:32:42 SELL  │ Limit @ $2,340   │
│ 43,256.78 ██░░░░ 1.2│ 0.2 @ 43,255   │ TTL: 1.8s        │
│ ──────────────────── │ ────────────── │                  │
│ BID                  │                │ FILLED (last 10) │
│ 43,255.10 ████░░ 2.1│                │ 14:31:12 FILLED  │
│ 43,254.50 █████░ 2.8│                │ SOLUSDT @ $98.76 │
│ 43,253.00 ███░░░ 1.5│                │ Maker +$0.34     │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- **TradingView Integration:** Full-featured charting with custom indicators
- **Physics Overlay:** Temperature/entropy/phase visualized on chart
- **Signal Markers:** Entry/exit signals with strategy attribution
- **Iceberg Detection:** Visual markers for hidden large orders
- **Order Book Heat Map:** Size-weighted color intensity
- **Real-Time Updates:** WebSocket for sub-second latency

---

### 3. Strategy Performance (/strategies)

**Layout:** Strategy comparison matrix

```
┌─────────────────────────────────────────────────────────┐
│  STRATEGY ALLOCATION: $12,456.78 TOTAL                  │
├─────────────────────────────────────────────────────────┤
│  [PIE CHART: Capital Distribution]                      │
│  - ImpulseEngine: 35% ($4,359.87)                       │
│  - FundingHarvester: 25% ($3,114.20)                    │
│  - BasisArbitrage: 20% ($2,491.36)                      │
│  - Others: 20% ($2,491.35)                              │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  STRATEGY COMPARISON TABLE                              │
│  ┌────────────┬────────┬────────┬───────┬──────┬───┐   │
│  │ Strategy   │ Sharpe │ Win %  │ PnL   │ Alloc│ ••│   │
│  ├────────────┼────────┼────────┼───────┼──────┼───┤   │
│  │ Impulse    │  1.82  │ 64.2% │ +$456 │ 35%  │ ▲ │   │
│  │ [chart]    │ ██████ │ ██████ │ +8.2% │ ████ │   │   │
│  │            │        │        │       │      │   │   │
│  │ Funding    │  1.45  │ 72.1% │ +$234 │ 25%  │ ▲ │   │
│  │ [chart]    │ █████░ │ ██████ │ +5.1% │ ███░ │   │   │
│  │            │        │        │       │      │   │   │
│  │ Basis      │  1.21  │ 68.5% │ +$123 │ 20%  │ ━ │   │
│  │ [chart]    │ ████░░ │ ██████ │ +3.2% │ ██░░ │   │   │
│  │            │        │        │       │      │   │   │
│  │ HFScalp    │  0.89  │ 58.3% │  +$45 │  8%  │ ▼ │   │
│  │ [chart]    │ ███░░░ │ █████░ │ +1.1% │ █░░░ │   │   │
│  └────────────┴────────┴────────┴───────┴──────┴───┘   │
│                                                          │
│  ALLOCATION HISTORY (Last 7 Days)                       │
│  [STACKED AREA CHART]                                   │
│  - Each strategy as a colored band                      │
│  - Annotation markers for reallocation events           │
│                                                          │
│  PHASE AFFINITY MATRIX                                  │
│  ┌────────────┬─────┬─────┬──────┬────────┐            │
│  │ Strategy   │ ICE │WATER│VAPOR │MARKUP  │            │
│  ├────────────┼─────┼─────┼──────┼────────┤            │
│  │ Impulse    │ ░░  │ ███ │ ░░   │ █████ │ (markup)   │
│  │ Funding    │ ███ │ ░░  │ ░░   │ ░░░   │ (ice)      │
│  │ Basis      │ ███ │ ░░  │ ░░   │ ░░░   │ (ice)      │
│  │ Liquidity  │ ░░  │ ░░  │ ███  │ ████  │ (volatile) │
│  └────────────┴─────┴─────┴──────┴────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Sortable Table:** Click headers to sort by Sharpe/win rate/PnL/allocation
- **Inline Sparklines:** Mini performance charts per strategy
- **Allocation Trends:** Historical capital shifts over time
- **Phase Affinity:** Visual matrix showing strategy-phase compatibility
- **Drill-Down:** Click strategy row to see detailed performance

---

### 4. Physics Dashboard (/physics)

**Layout:** Thermodynamic state visualization

```
┌─────────────────────────────────────────────────────────┐
│  MARKET THERMODYNAMICS                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  CURRENT STATE: LAPLACE MODE (SSD Resonance Detected)   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ⚡ RESONANCE STRENGTH: 0.87                       │   │
│  │ Vector alignment detected - deterministic regime  │   │
│  │ ────────────────────────────────────────────────  │   │
│  │ Price momentum:    ████████░░ +0.24%            │   │
│  │ Volume momentum:   ███████░░░ +18.5%            │   │
│  │ Entropy flow:      ████████░░ -0.012 (converging)│  │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  TEMPERATURE & ENTROPY (Multi-Symbol)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ BTCUSDT  [ICE]       Temp: 345  Entropy: 0.28    │   │
│  │ [▓▓▓▓░░░░░░░░░░░░░] [▓▓▓░░░░░░░░░░░░░░░░]        │   │
│  │ Phase: ICE (conf: 0.82) ┃ Trade: ✅ SIZE: 1.2x   │   │
│  │                                                   │   │
│  │ ETHUSDT  [WATER]     Temp: 567  Entropy: 0.45    │   │
│  │ [▓▓▓▓▓▓▓░░░░░░░░░░░] [▓▓▓▓▓▓░░░░░░░░░░░]        │   │
│  │ Phase: WATER (conf: 0.71) ┃ Trade: ✅ SIZE: 1.0x │   │
│  │                                                   │   │
│  │ SOLUSDT  [VAPOR]     Temp: 1234 Entropy: 0.89    │   │
│  │ [▓▓▓▓▓▓▓▓▓▓▓▓░░░░░] [▓▓▓▓▓▓▓▓▓▓▓░░░░░]        │   │
│  │ Phase: VAPOR (conf: 0.93) ┃ Trade: ⚠️ SIZE: 0.5x│   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  PHASE TRANSITION TIMELINE                              │
│  [GANTT-STYLE CHART]                                    │
│  BTCUSDT  ░░░[ICE]░░░[WATER]░░░░[ICE]░░░░░░           │
│  ETHUSDT  ░[WATER]░░░░░░░░░[VAPOR]░[WATER]░░         │
│  SOLUSDT  [ICE]░░░[WATER]░░░░░░░░░░░[VAPOR]          │
│           12:00  13:00  14:00  15:00  NOW              │
│                                                          │
│  SZILARD PROFIT EXTRACTION                              │
│  [GAUGE CHART: 0 → Max Profit]                         │
│  Current: $12.34 extractable                            │
│  Optimal entry point: BTCUSDT @ $43,180 (ICE→WATER)    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Resonance Indicator:** Large badge when SSD Laplace mode active
- **Vector Alignment:** Visual representation of price/volume/entropy momentum
- **Multi-Symbol State:** Sortable/filterable list of all tracked symbols
- **Phase Timeline:** Historical phase transitions (Gantt chart style)
- **Szilard Gauge:** Thermodynamic profit extraction opportunity meter

---

### 5. Execution Quality (/execution)

**Layout:** Order flow analysis

```
┌─────────────────────────────────────────────────────────┐
│  EXECUTION ANALYTICS                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  COST SAVINGS (vs Pure Taker)                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Today: -$45.67 saved (-18.3% execution cost)     │   │
│  │ [WATERFALL CHART]                                │   │
│  │ Baseline: $250 ┃ Maker rebates: -$78 ┃ TWAP: -$12│  │
│  │               ┗━━━━━━━━━━━━━━━━━━━━━┛            │   │
│  │ Final cost: $160  (36% maker, 64% taker)         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  MAKER vs TAKER BREAKDOWN (Last 24h)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │ [DONUT CHART]                                    │   │
│  │ Maker Orders: 36% (142 fills) +$78.34 rebates   │   │
│  │ Taker Orders: 64% (251 fills) -$238.34 fees     │   │
│  │                                                   │   │
│  │ Maker Fill Rate: 52% (up from 30% baseline)     │   │
│  │ [PROGRESS BAR] ████████████░░░░░░░░░░ 52%       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  TWAP EXECUTIONS (Active + Recent)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ACTIVE (1)                                       │   │
│  │ Session: twap-4a8f                               │   │
│  │ BTCUSDT BUY 2.5 → 10 slices @ 12s intervals     │   │
│  │ Progress: ████████░░░░░░░░░░ 3/10 (30%)         │   │
│  │ Filled: 0.75 BTC @ avg $43,245.67               │   │
│  │ Next slice in: 8.2s                             │   │
│  │                                                   │   │
│  │ COMPLETED (last 5)                               │   │
│  │ 14:12 ETHUSDT 15.2 → 6 slices, avg $2,345.67    │   │
│  │ 13:45 SOLUSDT 250 → 8 slices, avg $98.76        │   │
│  │ 12:30 BTCUSDT 1.8 → 5 slices, avg $43,120.45    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  LIMIT ORDER PERFORMANCE                                │
│  [LINE CHART: Fill Rate over Time]                     │
│  - Target: 50% fill rate                                │
│  - Current: 52% (above target)                          │
│  - TTL distribution histogram                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Cost Waterfall:** Visual breakdown of execution cost savings
- **Maker Rebate Tracking:** Cumulative rebates earned
- **TWAP Progress:** Real-time execution status for active TWAP sessions
- **Fill Rate Trends:** Historical limit order fill rate over time

---

## iOS APP REDESIGN

### Architecture

**SwiftUI + MVVM**
**Minimum iOS:** 17.0
**Design Language:** Apple HIG + Custom Trading UI
**Networking:** Combine + URLSession (WebSocket for real-time)

### Navigation Structure

```
TabView (5 tabs at bottom)
├── 🏠 Overview
├── 📊 Live
├── 🎯 Strategies
├── ⚛️ Physics
└── ⚙️ Settings

Each tab uses NavigationStack for drill-down navigation
```

---

## iOS APP SCREENS

### 1. Overview Tab

**Layout:** Scrollable card-based layout (iOS native)

```
┌─────────────────────────┐
│  HEAN                   │ ← Navigation Bar
│  [LIVE] $12,456.78      │
├─────────────────────────┤
│                         │
│  ┌───────────────────┐  │ ← Hero Card (glass effect)
│  │ EQUITY            │  │
│  │ $12,456.78        │  │
│  │ +$456.78 (8.2%)   │  │
│  │ [mini sparkline]  │  │
│  └───────────────────┘  │
│                         │
│  ┌─────────┬─────────┐  │ ← Split Cards
│  │ DAILY   │ OPEN    │  │
│  │ +$456   │ 7 / 10  │  │
│  │ +8.2%   │ 70%     │  │
│  └─────────┴─────────┘  │
│                         │
│  ┌───────────────────┐  │ ← Physics Card
│  │ 📊 MARKET PHASE   │  │
│  │ LAPLACE MODE      │  │
│  │ Resonance: 0.87   │  │
│  │ [circular gauge]  │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │ ← Strategies Card
│  │ 🎯 TOP STRATEGIES │  │
│  │ ─────────────     │  │
│  │ Impulse +$234 52% │  │
│  │ Funding +$123 27% │  │
│  │ Basis    +$99 21% │  │
│  │ [See All →]       │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │ ← Activity Card
│  │ 📋 RECENT ACTIVITY│  │
│  │ ─────────────     │  │
│  │ 14:32 ⚡ Laplace  │  │
│  │ 14:31 📊 ETH Long │  │
│  │ 14:30 💰 Maker    │  │
│  │ [See All →]       │  │
│  └───────────────────┘  │
│                         │
└─────────────────────────┘
```

**iOS-Specific Features:**
- **Pull to Refresh:** Standard iOS gesture
- **Haptic Feedback:** On card tap, alerts
- **Dynamic Type:** Supports accessibility text sizes
- **Dark Mode Native:** Uses iOS system dark mode API

---

### 2. Live Trading Tab

**Layout:** Full-screen chart with overlay controls

```
┌─────────────────────────┐
│ BTCUSDT  $43,256.78     │ ← Collapsible header
│ ▲ $234.56 (0.54%)       │
├─────────────────────────┤
│                         │
│  [CHART - FULL HEIGHT]  │ ← Interactive TradingView chart
│                         │ ← Swipe up for order book
│  [Volume bars]          │ ← Swipe down for positions
│                         │
│  ⚡ SSD Event marker    │ ← Tap markers for details
│  🟢 Buy signal          │
│                         │
├─────────────────────────┤ ← Sheet (drag to expand)
│  ▔▔▔ (drag handle)      │
│  OPEN POSITIONS (3)     │
│                         │
│  [List view]            │
│  ETHUSDT LONG           │
│  Entry: $2,345.67       │
│  PnL: +$123.45 (+5.2%)  │
│  [Swipe to Close]       │
│                         │
│  SOLUSDT LONG           │
│  Entry: $98.76          │
│  PnL: +$45.67 (+2.1%)   │
│  [Swipe to Close]       │
│                         │
└─────────────────────────┘
```

**iOS-Specific Features:**
- **Bottom Sheet:** Native `.sheet()` or `.halfSheet()` modifier
- **Swipe Gestures:** Left/right to close position, up/down to switch sheets
- **Context Menus:** Long-press on position for quick actions
- **Charts:** SwiftUI Charts for simple views, WebView for TradingView

---

### 3. Strategies Tab

**Layout:** List with drill-down

```
┌─────────────────────────┐
│  Strategies             │
│  [Filter: All ▾]        │
├─────────────────────────┤
│                         │
│  ┌───────────────────┐  │ ← Card per strategy
│  │ 🚀 Impulse Engine │  │
│  │ Allocated: $4,360 │  │
│  │ ───────────────   │  │
│  │ PnL: +$456 (8.2%)│  │
│  │ Sharpe: 1.82 █████│  │
│  │ Win: 64.2% █████  │  │
│  │ [chart mini]      │  │
│  │ ─────────         │  │
│  │ Phase: MARKUP ✓   │  │ ← Affinity indicator
│  │ [Details →]       │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ 💎 Funding Harv.  │  │
│  │ Allocated: $3,114 │  │
│  │ ───────────────   │  │
│  │ PnL: +$234 (5.1%)│  │
│  │ Sharpe: 1.45 ████ │  │
│  │ Win: 72.1% ██████ │  │
│  │ [chart mini]      │  │
│  │ ─────────         │  │
│  │ Phase: ICE ✓      │  │
│  │ [Details →]       │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ [+ More (6)]      │  │
│  └───────────────────┘  │
│                         │
└─────────────────────────┘
```

**Drill-Down Detail View:**
```
┌─────────────────────────┐
│ ← Impulse Engine        │
├─────────────────────────┤
│                         │
│  PERFORMANCE CHART      │
│  [Full-size line chart] │
│  Last 30 days           │
│                         │
│  ┌───────────────────┐  │
│  │ METRICS           │  │
│  │ ─────────────     │  │
│  │ Sharpe: 1.82      │  │
│  │ Win Rate: 64.2%   │  │
│  │ Profit Factor: 2.4│  │
│  │ Avg Trade: +$12.3 │  │
│  │ Trade Count: 37   │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ ALLOCATION        │  │
│  │ ─────────────     │  │
│  │ Current: $4,360   │  │
│  │ Target: $4,500    │  │
│  │ [Progress bar]    │  │
│  │ Next realloc: 45m │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ RECENT TRADES     │  │
│  │ ─────────────     │  │
│  │ [List of trades]  │  │
│  └───────────────────┘  │
│                         │
└─────────────────────────┘
```

---

### 4. Physics Tab

**Layout:** Thermodynamic visualizations

```
┌─────────────────────────┐
│  Market Physics         │
│  [Symbol: BTCUSDT ▾]    │
├─────────────────────────┤
│                         │
│  ┌───────────────────┐  │ ← Hero Status Card
│  │ ⚡ LAPLACE MODE   │  │
│  │ Resonance: 0.87   │  │
│  │ [Circular gauge]  │  │
│  │                   │  │
│  │ 📊 Phase: ICE     │  │
│  │ Confidence: 82%   │  │
│  └───────────────────┘  │
│                         │
│  ┌─────────┬─────────┐  │ ← Temperature & Entropy
│  │ TEMP    │ ENTROPY │  │
│  │ 345     │ 0.28    │  │
│  │ [meter] │ [meter] │  │
│  │ COLD    │ COMPRESS│  │
│  └─────────┴─────────┘  │
│                         │
│  ┌───────────────────┐  │ ← Trading Recommendation
│  │ 🎯 TRADING ADVICE │  │
│  │ ─────────────     │  │
│  │ ✅ Trade: YES     │  │
│  │ Size: 1.2x        │  │
│  │ Reason: ICE phase │  │
│  │ stable, low vol   │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │ ← Phase Timeline
│  │ 📈 PHASE HISTORY  │  │
│  │ [Gantt chart]     │  │
│  │ ICE → WATER → ICE │  │
│  │ 12h  13h  14h NOW │  │
│  └───────────────────┘  │
│                         │
└─────────────────────────┘
```

**iOS-Specific Features:**
- **Gauges:** Native SwiftUI `Gauge` view
- **SF Symbols:** Use system icons (chart.xyaxis.line, flame.fill)
- **Animations:** Spring animations for state transitions

---

## DESIGN SYSTEM COMPONENTS

### Web Components (React + TailwindCSS)

```tsx
// Card.tsx
<Card variant="glass" size="md">
  <CardHeader>
    <CardTitle icon={IconFlame}>Temperature</CardTitle>
  </CardHeader>
  <CardBody>
    <Meter value={345} max={1000} color="warning" />
  </CardBody>
</Card>

// Meter.tsx (custom gauge)
<div className="relative h-32 w-32">
  <svg viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="45" className="stroke-bg-tertiary" />
    <circle
      cx="50" cy="50" r="45"
      className="stroke-accent-primary"
      strokeDasharray={`${percent * 283} 283`}
    />
  </svg>
  <span className="absolute inset-0 flex items-center justify-center text-2xl font-mono">
    {value}
  </span>
</div>

// PriceDisplay.tsx
<span className={cn(
  "font-mono text-lg",
  isPositive ? "text-success" : "text-danger"
)}>
  ${price.toFixed(2)}
  <span className="text-sm ml-1">
    {isPositive ? "▲" : "▼"} {percent}%
  </span>
</span>

// StatusBadge.tsx
<Badge
  variant={mode === "laplace" ? "success" : "default"}
  icon={mode === "laplace" ? IconZap : undefined}
  pulse={mode === "laplace"}
>
  {mode.toUpperCase()}
</Badge>
```

### iOS Components (SwiftUI)

```swift
// MetricCard.swift
struct MetricCard: View {
    let title: String
    let value: String
    let change: Double
    let chart: [Double]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.bodySmall)
                .foregroundColor(.textSecondary)

            Text(value)
                .font(.displayHero)
                .foregroundColor(.textPrimary)

            HStack {
                Image(systemName: change > 0 ? "arrow.up" : "arrow.down")
                Text("\(abs(change), specifier: "%.2f")%")
            }
            .font(.bodyMedium)
            .foregroundColor(change > 0 ? .profit : .loss)

            Sparkline(data: chart)
                .stroke(Color.accentPrimary, lineWidth: 1.5)
                .frame(height: 40)
        }
        .padding()
        .background(Color.bgSecondary)
        .cornerRadius(12)
    }
}

// PhysicsGauge.swift
struct PhysicsGauge: View {
    let value: Double
    let max: Double
    let label: String

    var body: some View {
        Gauge(value: value, in: 0...max) {
            Text(label)
        } currentValueLabel: {
            Text("\(value, specifier: "%.0f")")
                .font(.mono)
        }
        .gaugeStyle(.accessoryCircular)
        .tint(gaugeColor)
    }

    var gaugeColor: Color {
        switch value / max {
        case 0..<0.3: return .info
        case 0.3..<0.7: return .accentPrimary
        default: return .warning
        }
    }
}

// StatusPill.swift
struct StatusPill: View {
    let mode: String
    let isActive: Bool

    var body: some View {
        HStack(spacing: 4) {
            if isActive {
                Circle()
                    .fill(Color.profit)
                    .frame(width: 8, height: 8)
            }
            Text(mode.uppercased())
                .font(.bodySmall.weight(.semibold))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.bgElevated)
        .cornerRadius(16)
    }
}
```

---

## ANIMATION & MICRO-INTERACTIONS

### Web

```css
/* Transition base */
.transition-base {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Card hover */
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 24px rgba(0, 217, 255, 0.15);
}

/* Number counter animation (uses react-spring) */
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 10px rgba(0, 217, 255, 0.3); }
  50% { box-shadow: 0 0 20px rgba(0, 217, 255, 0.6); }
}

/* Live indicator */
.live-indicator {
  animation: pulse-glow 2s ease-in-out infinite;
}

/* Sparkline draw animation */
@keyframes draw-sparkline {
  to { stroke-dashoffset: 0; }
}
```

### iOS

```swift
// Appear animation
.onAppear {
    withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
        isVisible = true
    }
}

// Number change animation
Text("\(equity, specifier: "%.2f")")
    .contentTransition(.numericText())
    .animation(.spring(response: 0.3), value: equity)

// Phase transition animation
withAnimation(.easeInOut(duration: 0.5)) {
    currentPhase = newPhase
}

// Haptic feedback
let generator = UIImpactFeedbackGenerator(style: .medium)
generator.impactOccurred()
```

---

## RESPONSIVE DESIGN

### Web Breakpoints

```css
/* Mobile: 0-640px */
@media (max-width: 640px) {
  .grid-4 { grid-template-columns: 1fr; }
  .hide-mobile { display: none; }
}

/* Tablet: 641-1024px */
@media (min-width: 641px) and (max-width: 1024px) {
  .grid-4 { grid-template-columns: repeat(2, 1fr); }
}

/* Desktop: 1025px+ */
@media (min-width: 1025px) {
  .grid-4 { grid-template-columns: repeat(4, 1fr); }
  .sidebar { display: block; }
}

/* Ultra-wide: 1920px+ */
@media (min-width: 1920px) {
  .container { max-width: 1600px; }
}
```

### iOS Adaptive Layout

```swift
// Size classes
@Environment(\.horizontalSizeClass) var sizeClass

var body: some View {
    if sizeClass == .compact {
        // iPhone portrait: single column
        VStack { /* ... */ }
    } else {
        // iPhone landscape / iPad: two columns
        HStack { /* ... */ }
    }
}

// iPad split view support
NavigationSplitView {
    SidebarView()
} detail: {
    DetailView()
}
```

---

## ACCESSIBILITY

### Web (WCAG 2.1 AA)

```tsx
// Contrast ratios
- Text on bg-primary: 14.2:1 (AAA)
- Accent on bg-primary: 7.8:1 (AA Large)
- Success/danger on bg-primary: 4.9:1 (AA)

// ARIA labels
<button aria-label="Close position ETHUSDT">
  <XIcon />
</button>

// Keyboard navigation
<Card tabIndex={0} onKeyDown={handleKeyPress}>

// Screen reader announcements
<div aria-live="polite" aria-atomic="true">
  Position opened: BTCUSDT long @ $43,256.78
</div>
```

### iOS (VoiceOver)

```swift
// Accessibility labels
.accessibilityLabel("Equity: \(equity) dollars")
.accessibilityValue("Up \(change)%")

// Grouping
.accessibilityElement(children: .combine)

// Traits
.accessibilityAddTraits(.isButton)
.accessibilityAddTraits(.updatesFrequently) // For live data

// Dynamic Type support
Text("Title").font(.headline.dynamic())
```

---

## PERFORMANCE TARGETS

### Web
- **First Contentful Paint:** < 1.2s
- **Time to Interactive:** < 2.5s
- **Lighthouse Score:** > 90
- **WebSocket latency:** < 100ms
- **Chart render:** 60fps for smooth animations

### iOS
- **App launch:** < 2s cold start
- **Frame rate:** 120fps on ProMotion displays
- **Memory usage:** < 200MB average
- **Battery impact:** < 5% per hour (background refresh off)

---

## IMPLEMENTATION PHASES

### Phase 1: Foundation (2 weeks)
- [ ] Set up Next.js project with TailwindCSS
- [ ] Implement design system components (Card, Button, Badge, etc.)
- [ ] Create color scheme and typography system
- [ ] Set up WebSocket client for real-time data
- [ ] Implement iOS SwiftUI design system

### Phase 2: Core Screens (3 weeks)
- [ ] Web: Overview dashboard
- [ ] Web: Live trading view
- [ ] iOS: Overview tab
- [ ] iOS: Live trading tab
- [ ] Integrate TradingView charts (web)
- [ ] Integrate chart library (iOS)

### Phase 3: Advanced Features (3 weeks)
- [ ] Web: Strategy performance dashboard
- [ ] Web: Physics visualization
- [ ] Web: Execution analytics
- [ ] iOS: Strategies tab
- [ ] iOS: Physics tab
- [ ] Implement all data fetching hooks

### Phase 4: Polish (2 weeks)
- [ ] Animations and micro-interactions
- [ ] Responsive design testing (all breakpoints)
- [ ] Accessibility audit and fixes
- [ ] Performance optimization
- [ ] Error states and loading skeletons

### Phase 5: Testing & Launch (1 week)
- [ ] E2E testing (Playwright for web, XCTest for iOS)
- [ ] User acceptance testing
- [ ] Deploy web to Vercel
- [ ] Submit iOS app to TestFlight
- [ ] Production rollout

---

## TECH STACK SUMMARY

### Web Dashboard
```yaml
Framework: Next.js 14 (App Router)
Styling: TailwindCSS + CSS Variables
State: Zustand
Charts: TradingView Lightweight Charts + Recharts
Real-Time: WebSocket + SWR
Deployment: Vercel
Testing: Playwright + Vitest
```

### iOS App
```yaml
Framework: SwiftUI
Min iOS: 17.0
Architecture: MVVM + Combine
Charts: SwiftUI Charts + custom views
Real-Time: URLSession WebSocket
Testing: XCTest + UI Tests
Distribution: TestFlight → App Store
```

---

## DESIGN FILES STRUCTURE

```
/design
├── /web
│   ├── /components       # Storybook components
│   ├── /screens          # Full screen mockups (Figma)
│   ├── /assets           # Icons, illustrations
│   └── /design-tokens    # JSON export for Tailwind config
│
├── /ios
│   ├── /screens          # iOS screen mockups (Sketch/Figma)
│   ├── /assets           # SF Symbols, custom icons
│   └── /color-sets       # Xcode color asset catalog
│
└── /shared
    ├── /brand            # Logo, brand guidelines
    ├── /icons            # Shared iconography
    └── /typography       # Font files (Inter, JetBrains Mono)
```

---

## NEXT STEPS

1. **Design Review:** Present mockups to stakeholders
2. **Technical Spike:** Prototype TradingView integration
3. **API Contract:** Finalize WebSocket/REST API spec
4. **Sprint Planning:** Break phases into 2-week sprints
5. **Hire/Assign:** Frontend dev (React) + iOS dev (SwiftUI)

---

**END OF DESIGN PLAN**

This comprehensive design plan provides a production-ready blueprint for the HEAN web dashboard and iOS app redesign, following best practices from leading trading platforms while maintaining a modern, professional aesthetic inspired by Clawdbot.
