# HEAN iOS App - World-Class Trading Application Analysis

## Part 1: Competitive Research - World's Best Trading Apps

### 1.1 Apps Analyzed

| # | App | Category | Key Strength | Monthly Users |
|---|-----|----------|-------------|---------------|
| 1 | **Binance** | Crypto Exchange | Real-time data, widget system, comprehensive tools | 120M+ |
| 2 | **Bybit** | Derivatives | Copy trading, position management, derivatives UX | 30M+ |
| 3 | **Bloomberg Terminal** | Professional | Data density, analytics depth, institutional tools | 325K+ |
| 4 | **Robinhood** | Retail Trading | Simplicity, onboarding, gamification | 23M+ |
| 5 | **TradingView** | Charts/Social | Chart customization, social features, indicators | 50M+ |
| 6 | **Interactive Brokers** | Multi-Asset | Professional tools, global markets, low cost | 2.5M+ |
| 7 | **Coinbase** | Crypto | Clean UX, portfolio tracking, education | 110M+ |
| 8 | **WeBull** | Retail Trading | Advanced charts, free trading, paper trading | 12M+ |
| 9 | **thinkorswim** | Options | Options analytics, probability analysis, paper trading | 11M+ |
| 10 | **Revolut** | Banking+Trading | Modern UX, multi-product, accessibility | 35M+ |

---

### 1.2 Best Practices Extracted

#### From Binance (Real-time Excellence)
- **Sub-100ms price updates** via WebSocket with visual flash animations
- **Color-coded orderbook** with depth visualization (green bids, red asks)
- **Widget system** allowing users to customize home screen with trading pairs
- **Notification hierarchy**: Critical (liquidation) > Important (fill) > Info (signal)
- **Gesture-based trading**: Swipe to close position, long-press for quick actions

#### From Bybit (Derivatives UX)
- **Position card design**: Entry price, mark price, PnL, ROI%, margin all visible at glance
- **TP/SL visualization**: Drag handles on chart to set take-profit and stop-loss
- **Copy trading**: Follow top traders with configurable allocation
- **Risk metrics inline**: Available balance, position margin, maintenance margin always visible
- **One-tap close**: Instant position closure with confirmation haptic

#### From Bloomberg Terminal Mobile
- **Information density**: 40+ data points per screen without feeling cluttered
- **Data hierarchy**: Primary metrics large, secondary small, tertiary on demand
- **Professional typography**: Monospaced numbers, tabular alignment, consistent precision
- **Watchlist system**: Multi-column sortable lists with custom column configuration
- **Alert system**: Price alerts, volume alerts, custom formula alerts

#### From Robinhood (Consumer UX)
- **Onboarding**: 3-screen walkthrough, first trade within 60 seconds
- **Emotion design**: Confetti on first trade, color psychology (green = good)
- **Progressive disclosure**: Basic view by default, swipe/tap for advanced
- **Single-number focus**: One hero metric per card (price, PnL, etc.)
- **Storytelling**: "Your portfolio is up 3.2% this week" narrative approach

#### From TradingView (Charts)
- **Multi-timeframe charts**: 1m, 5m, 15m, 1h, 4h, 1D, 1W
- **150+ indicators**: MA, RSI, MACD, Bollinger, Ichimoku, Volume Profile
- **Drawing tools**: Trendlines, Fibonacci, horizontal lines, rectangles
- **Chart types**: Candles, Heikin-Ashi, Renko, Line, Area, Bars
- **Social features**: Share chart snapshots, follow analysts, comment threads

#### From Interactive Brokers TWS
- **Multi-account support**: Switch between accounts seamlessly
- **Order types**: 50+ order types (Market, Limit, Stop, Trailing, OCO, Bracket)
- **Risk navigator**: Real-time risk analytics with Greeks, scenario analysis
- **Margin calculator**: Pre-trade margin impact estimation
- **Audit trail**: Complete order history with timestamps and status changes

#### From Coinbase (Clean UX)
- **Portfolio pie chart**: Visual allocation breakdown by asset
- **Price alerts**: Simple "Notify me when BTC reaches $50,000"
- **Transaction history**: Clean timeline with filter/search
- **Education integration**: "Learn" cards explaining concepts
- **Security**: Biometric auth, 2FA, withdrawal confirmations

#### From WeBull (Advanced Retail)
- **Paper trading mode**: Full simulation without risk
- **Extended hours data**: Pre-market and after-hours visibility
- **Technical analysis**: Automatic pattern recognition
- **News feed**: Integrated real-time news per symbol
- **Community features**: Comments, predictions, sentiment polls

#### From thinkorswim (Analytics)
- **Options chain**: Vertical spreads, probability cones, Greeks overlay
- **Probability analysis**: Expected move, probability of profit
- **Backtesting**: Strategy backtester with historical data
- **Scripting**: Custom indicator scripting language
- **Multi-screen**: Support for external displays

#### From Revolut (Modern Banking)
- **Glassmorphism**: Frosted glass effects, depth through blur
- **Micro-interactions**: Every tap has feedback (haptic + visual)
- **Bottom sheet navigation**: Swipe-up for details, swipe-down to dismiss
- **Currency conversion**: Inline real-time conversion
- **Spending analytics**: Category-based breakdown with trends

---

### 1.3 Feature Comparison Matrix

| Feature | Binance | Bybit | Bloomberg | Robinhood | TradingView | HEAN (Target) |
|---------|---------|-------|-----------|-----------|-------------|---------------|
| Real-time prices | Y | Y | Y | Y | Y | **Y** |
| WebSocket updates | Y | Y | Y | Y | Y | **Y** |
| Candlestick charts | Y | Y | Y | Y | Y | **Y** |
| Technical indicators | 30+ | 20+ | 100+ | 5 | 150+ | **15+** |
| Position management | Y | Y | Y | Y | N | **Y** |
| Order types | 10+ | 8+ | 50+ | 3 | N | **5+** |
| Risk visualization | N | Basic | Y | N | N | **Y (Advanced)** |
| Strategy control | N | N | N | N | N | **Y (Unique)** |
| AI Assistant | N | N | N | N | N | **Y (Unique)** |
| Signal feed | N | N | N | N | N | **Y (Unique)** |
| Trading transparency | N | N | N | N | N | **Y (Unique)** |
| Widgets | Y | Y | N | Y | N | **Y** |
| Watch App | N | N | Y | Y | N | **Y** |
| Live Activities | N | N | N | Y | N | **Y** |
| Haptic feedback | Basic | Basic | N | Y | N | **Y (Advanced)** |
| Push notifications | Y | Y | Y | Y | Y | **Y** |
| Biometric auth | Y | Y | N | Y | N | **Y** |
| Dark mode | Y | Y | Y | Y | Y | **Y (Primary)** |
| Accessibility | Basic | Basic | Y | Y | Basic | **Y (Full)** |

---

## Part 2: HEAN Competitive Advantages (Unique Features)

### 2.1 AI Assistant (No Competitor Has This)
- Natural language queries: "Why am I not trading right now?"
- Context-aware responses using HEAN's `/trading/why` endpoint
- Quick suggestion chips for common questions
- Conversation history with timestamps
- Links diagnostics data to human-readable explanations

### 2.2 Trading Transparency (Unique to HEAN)
- Real-time signal pipeline visualization: Signal -> Decision -> Order -> Fill
- Every blocked signal shows the exact reason (risk limit, confidence too low, etc.)
- Order decision audit trail with reason codes
- Strategy-level decision explanations
- "Why Not Trading" diagnostic panel

### 2.3 Strategy Control (Unique to HEAN)
- Live enable/disable strategies with one tap
- Per-strategy performance metrics (PnL, win rate, signal count)
- Parameter tuning interface with sliders
- Strategy comparison view
- Real-time strategy event feed

### 2.4 Risk State Machine Visualization (Advanced)
- Visual state machine: NORMAL -> SOFT_BRAKE -> QUARANTINE -> HARD_STOP
- Animated transitions between states
- Drawdown progress bar with warning/critical thresholds
- Killswitch controls with biometric confirmation
- Per-symbol quarantine management

### 2.5 Signal Feed with Reasoning (Unique)
- Real-time signal stream with confidence scores
- Entry/exit reasoning for each signal
- Side indicator (LONG/SHORT) with color coding
- Strategy attribution
- Signal-to-order correlation tracking

---

## Part 3: Architecture Decisions

### 3.1 Pattern: MVVM + Clean Architecture

```
┌─────────────────────────────────────────────────┐
│                    Views (SwiftUI)                │
│  DashboardView, TradeView, StrategiesView, ...   │
├─────────────────────────────────────────────────┤
│                ViewModels (@MainActor)            │
│  DashboardVM, TradeVM, StrategiesVM, ...         │
│  - @Published properties for UI binding           │
│  - Combine subscriptions for real-time            │
│  - async/await for API calls                      │
├─────────────────────────────────────────────────┤
│              Service Layer (Protocols)            │
│  TradingService, PortfolioService, RiskService   │
│  - Protocol-based for testability                 │
│  - Live + Mock implementations                    │
├─────────────────────────────────────────────────┤
│              Core Infrastructure                  │
│  APIClient (Actor), WebSocketManager, Storage     │
│  - Thread-safe networking                         │
│  - Exponential backoff reconnection               │
│  - Keychain for credentials                       │
├─────────────────────────────────────────────────┤
│              Data Models (Codable)                │
│  Market, Position, Order, Signal, RiskState       │
│  - Immutable structs                              │
│  - Identifiable, Hashable                         │
└─────────────────────────────────────────────────┘
```

### 3.2 Technology Choices

| Technology | Purpose | Rationale |
|-----------|---------|-----------|
| **SwiftUI** | UI Framework | Declarative, reactive, iOS 17+ features |
| **Combine** | Reactive Streams | Native Apple framework, WebSocket integration |
| **Swift Concurrency** | Async Operations | async/await, actors for thread safety |
| **URLSession** | Networking | Native, no dependencies, WebSocket support |
| **OSLog** | Logging | System-level, categorized, performant |
| **Keychain** | Secure Storage | API keys, tokens, sensitive data |
| **UserDefaults** | Preferences | Settings, UI state, non-sensitive data |
| **WidgetKit** | Home Screen Widgets | Portfolio, PnL at a glance |
| **ActivityKit** | Live Activities | Position tracking on lock screen |
| **WatchKit** | Apple Watch | Glanceable dashboard and alerts |

### 3.3 Why No External Dependencies

| Common Dependency | Why We Skip It | Our Alternative |
|-------------------|---------------|-----------------|
| Alamofire | URLSession is sufficient | Actor-based APIClient |
| Socket.IO | HEAN uses plain WebSocket | URLSessionWebSocketTask |
| Kingfisher | No remote images needed | N/A |
| SwiftyJSON | Codable is sufficient | Codable structs |
| Charts (3rd party) | SwiftUI Charts is native | Swift Charts framework |
| Realm/CoreData | Simple caching needs | UserDefaults + files |

**Zero dependencies = faster builds, smaller binary, no supply chain risk.**

---

## Part 4: Design System

### 4.1 Color Palette

```
Background:
  Primary:    #0A0A0F (Deep space black)
  Secondary:  #12121A (Elevated surface)
  Tertiary:   #1A1A2E (Card background)

Accent:
  Primary:    #00D4FF (Cyan - brand color)
  Secondary:  #7B61FF (Purple - AI/premium)
  Tertiary:   #FF6B35 (Orange - attention)

Semantic:
  Success:    #22C55E (Green - profit, buy)
  Error:      #EF4444 (Red - loss, sell)
  Warning:    #F59E0B (Amber - caution)
  Info:       #3B82F6 (Blue - information)

Text:
  Primary:    #FFFFFF (White)
  Secondary:  #94A3B8 (Gray-400)
  Tertiary:   #64748B (Gray-500)
  Disabled:   #334155 (Gray-700)
```

### 4.2 Typography

```
Hero:       SF Pro Rounded, 34pt, Bold
Title1:     SF Pro Display, 28pt, Bold
Title2:     SF Pro Display, 22pt, Bold
Title3:     SF Pro Display, 20pt, Semibold
Headline:   SF Pro Text, 17pt, Semibold
Body:       SF Pro Text, 17pt, Regular
Callout:    SF Pro Text, 16pt, Regular
Subhead:    SF Pro Text, 15pt, Regular
Footnote:   SF Pro Text, 13pt, Regular
Caption1:   SF Pro Text, 12pt, Regular
Caption2:   SF Pro Text, 11pt, Regular

Numbers:    SF Mono, various sizes, Medium (tabular lining)
```

### 4.3 Component Library

| Component | Purpose | States |
|-----------|---------|--------|
| **GlassCard** | Primary container | Default, Highlighted, Error |
| **PriceTicker** | Animated price display | Rising, Falling, Neutral |
| **CandlestickChart** | OHLCV chart | Loading, Data, Empty, Error |
| **Sparkline** | Mini trend line | Up, Down, Flat |
| **PnLBadge** | Profit/Loss indicator | Positive, Negative, Zero |
| **RiskBadge** | Risk state badge | Normal, SoftBrake, Quarantine, HardStop |
| **StatusIndicator** | Connection status | Connected, Reconnecting, Disconnected |
| **SkeletonView** | Loading placeholder | Animating |
| **ConfidenceBadge** | Signal confidence | High (>0.7), Medium (0.4-0.7), Low (<0.4) |
| **MetricView** | Label + value pair | Default, Highlighted |
| **ErrorBanner** | Error message | Warning, Error, Critical |
| **LoadingOverlay** | Full-screen loader | Active |

### 4.4 Animation Principles

```
Spring Animations:
  Quick:     response: 0.3, dampingFraction: 0.7
  Standard:  response: 0.5, dampingFraction: 0.8
  Slow:      response: 0.8, dampingFraction: 0.85

Transitions:
  Screen:    .asymmetric(insertion: .push(from: .trailing), removal: .push(from: .leading))
  Sheet:     .spring(response: 0.4, dampingFraction: 0.85)
  Card:      .scale.combined(with: .opacity)

Timing:
  Price flash:     200ms
  State change:    300ms
  Screen transition: 400ms
  Chart animation:   600ms
```

### 4.5 Haptic Feedback Map

| Action | Haptic Type | Intensity |
|--------|-------------|-----------|
| Order placed | .success | Strong |
| Order filled | .success | Medium |
| Order cancelled | .warning | Light |
| Order rejected | .error | Strong |
| Position opened | .impact(.heavy) | Strong |
| Position closed | .impact(.medium) | Medium |
| Killswitch activated | .error + vibrate | Maximum |
| Risk state change | .warning | Medium |
| Button tap | .selection | Light |
| Pull to refresh | .impact(.light) | Light |
| Signal received | .impact(.light) | Light |

---

## Part 5: API Integration Guide

### 5.1 REST Endpoints Used by iOS App

#### Dashboard Screen
```
GET /engine/status          -> Engine running/stopped
GET /orders/positions       -> Active positions
GET /trading/metrics        -> Trading funnel stats
GET /analytics/summary      -> Win rate, profit factor
GET /risk/governor/status   -> Risk state
```

#### Trade Screen
```
POST /orders/test           -> Place order
POST /orders/close-position -> Close position
POST /orders/cancel-all     -> Cancel all
GET  /orders                -> Order list
```

#### Strategies Screen
```
GET  /strategies                     -> List strategies
POST /strategies/{id}/enable         -> Toggle strategy
POST /strategies/{id}/params         -> Update params
```

#### Risk Screen
```
GET  /risk/status              -> Risk overview
GET  /risk/limits              -> Current limits
GET  /risk/governor/status     -> Governor state
GET  /risk/killswitch/status   -> Killswitch status
POST /risk/killswitch/reset    -> Reset killswitch
```

#### AI Assistant Screen
```
GET /trading/why               -> Why not trading diagnostics
GET /trading/state             -> Current trading state
GET /analytics/summary         -> Analytics for AI context
```

### 5.2 WebSocket Topics

```swift
enum WebSocketTopic: String {
    // Real-time data
    case systemStatus = "system_status"
    case systemHeartbeat = "system_heartbeat"
    case marketData = "market_data"
    case marketTicks = "market_ticks"

    // Trading
    case orders = "orders"
    case positions = "positions"
    case tradingMetrics = "trading_metrics"
    case tradingEvents = "trading_events"

    // Strategy & Signals
    case signals = "signals"
    case strategyEvents = "strategy_events"

    // Account
    case accountState = "account_state"
    case orderDecisions = "order_decisions"

    // Risk
    case riskEvents = "risk_events"

    // Full snapshot
    case snapshot = "snapshot"
}
```

### 5.3 Connection Protocol

```swift
// Subscribe to topic
let message = ["action": "subscribe", "topic": "positions"]
websocket.send(json: message)

// Unsubscribe
let message = ["action": "unsubscribe", "topic": "positions"]
websocket.send(json: message)

// Ping/Pong (30s interval)
let ping = ["action": "ping"]
websocket.send(json: ping)
// Server responds: {"type": "pong", "timestamp": "..."}
```

### 5.4 Error Handling Strategy

```
HTTP Errors:
  401 -> Re-authenticate, show login
  403 -> Show "access denied" banner
  404 -> Show "not found" state
  429 -> Rate limited, retry after delay
  500 -> Show error, retry with backoff

WebSocket Errors:
  Connection lost -> Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, 16s, max 30s)
  Invalid message -> Log and skip
  Auth failed -> Re-authenticate

Business Errors:
  RISK_BLOCKED -> Show risk state, explain reason
  ORDER_REJECTED -> Show rejection reason, suggest fix
  KILLSWITCH -> Show emergency state, disable trading UI
```

---

## Part 6: Implementation Roadmap

### Week 1: Foundation & Services
- [ ] Service protocols (Trading, Portfolio, Market, Strategy, Risk, Signal)
- [ ] Live service implementations with APIClient integration
- [ ] WebSocket topic subscriptions in services
- [ ] Data models for all API responses
- [ ] DIContainer updates for service injection

### Week 2: Core Features
- [ ] Dashboard with real data (equity, PnL, positions, risk state)
- [ ] Position management (view, close, TP/SL)
- [ ] Order management (place, cancel, history)
- [ ] Market overview with live prices
- [ ] Settings with API configuration

### Week 3: Advanced Features
- [ ] Strategy control (list, enable/disable, performance)
- [ ] Risk dashboard (state machine, killswitch, limits)
- [ ] Signal feed (real-time, confidence, reasoning)
- [ ] Trading transparency ("Why" diagnostics)
- [ ] AI Assistant (natural language queries)

### Week 4: Charts & Analytics
- [ ] Candlestick chart with indicators (MA, RSI, MACD, Bollinger)
- [ ] Portfolio analytics (Sharpe, Sortino, drawdown, equity curve)
- [ ] Per-strategy PnL breakdown
- [ ] Market heatmap
- [ ] Order flow visualization

### Week 5: iOS Platform Features & Polish
- [ ] Home screen widgets (Portfolio, PnL)
- [ ] Apple Watch companion app
- [ ] Push notifications (APNs)
- [ ] Live Activities for active positions
- [ ] Biometric authentication
- [ ] Accessibility audit (VoiceOver, Dynamic Type)
- [ ] Performance optimization
- [ ] Onboarding flow with TipKit

---

## Part 7: Key Implementation Patterns

### 7.1 ViewModel Pattern

```swift
@MainActor
class DashboardViewModel: ObservableObject {
    // Published state for UI binding
    @Published var equity: Double = 0
    @Published var pnl: Double = 0
    @Published var positions: [Position] = []
    @Published var riskState: RiskState = .normal
    @Published var isLoading = false
    @Published var error: AppError?

    // Service dependencies (protocol-based)
    private let portfolioService: PortfolioServiceProtocol
    private let tradingService: TradingServiceProtocol
    private let riskService: RiskServiceProtocol
    private var cancellables = Set<AnyCancellable>()

    // Combine subscriptions for real-time updates
    func subscribeToUpdates() {
        portfolioService.equityPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] equity in
                self?.equity = equity
            }
            .store(in: &cancellables)
    }

    // Async/await for initial/refresh loads
    func refresh() async {
        isLoading = true
        defer { isLoading = false }

        do {
            async let e = portfolioService.fetchEquity()
            async let p = tradingService.fetchPositions()
            async let r = riskService.fetchRiskState()

            let (equity, positions, risk) = try await (e, p, r)
            self.equity = equity
            self.positions = positions
            self.riskState = risk
        } catch {
            self.error = AppError(error)
        }
    }
}
```

### 7.2 WebSocket Integration Pattern

```swift
class LiveTradingService: TradingServiceProtocol {
    private let websocket: WebSocketManager
    private let positionsSubject = CurrentValueSubject<[Position], Never>([])

    var positionsPublisher: AnyPublisher<[Position], Never> {
        positionsSubject.eraseToAnyPublisher()
    }

    init(websocket: WebSocketManager) {
        self.websocket = websocket

        // Subscribe to positions topic
        websocket.subscribe(to: .positions)

        // Process incoming position events
        websocket.eventPublisher
            .compactMap { event -> [Position]? in
                guard event.topic == "positions" else { return nil }
                return try? JSONDecoder().decode([Position].self, from: event.data)
            }
            .sink { [weak self] positions in
                self?.positionsSubject.send(positions)
            }
            .store(in: &cancellables)
    }
}
```

### 7.3 Error State Handling Pattern

```swift
struct ContentWithStates<Content: View, Empty: View>: View {
    let isLoading: Bool
    let error: AppError?
    let isEmpty: Bool
    let content: () -> Content
    let emptyView: () -> Empty
    let onRetry: () async -> Void

    var body: some View {
        Group {
            if isLoading {
                SkeletonView()
            } else if let error = error {
                ErrorView(error: error, onRetry: { Task { await onRetry() } })
            } else if isEmpty {
                emptyView()
            } else {
                content()
            }
        }
    }
}
```

---

## Part 8: Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cold start | < 1s | Time to first meaningful content |
| WebSocket connect | < 500ms | Connection establishment |
| Screen transition | < 200ms | 60fps animation |
| Chart render (200 candles) | < 100ms | Canvas draw time |
| List scroll | 60fps | Lazy loading, no drops |
| Memory (typical) | < 100MB | Instruments measurement |
| Memory (peak) | < 200MB | Under heavy data load |
| Battery per hour | < 5% | Active use measurement |

---

## Part 9: Accessibility Requirements

### VoiceOver
- All interactive elements labeled with `.accessibilityLabel()`
- Financial numbers with context: "Bitcoin price: forty-two thousand three hundred fifty dollars"
- Position cards navigable with swipe gestures
- Custom actions for close position, modify order

### Dynamic Type
- All text scales with system font size
- Layout adapts to larger text (vertical stack instead of horizontal)
- Minimum touch target: 44x44 points

### Color Contrast
- WCAG AA compliant (4.5:1 for body text)
- Color never sole indicator (always paired with icon/text)
- Reduced motion support (cross-fade instead of slide)

---

## Part 10: File Structure

```
ios/HEAN/
├── App/
│   ├── HEANApp.swift
│   └── ContentView.swift
├── Core/
│   ├── Networking/
│   │   ├── APIClient.swift
│   │   ├── APIEndpoints.swift
│   │   └── APIError.swift
│   ├── WebSocket/
│   │   ├── WebSocketManager.swift
│   │   └── WebSocketEvent.swift
│   ├── DI/
│   │   └── DIContainer.swift
│   ├── Storage/
│   │   ├── UserDefaultsStore.swift
│   │   └── KeychainStore.swift
│   ├── Notifications/
│   │   ├── NotificationService.swift
│   │   └── PushNotificationHandler.swift
│   └── Logger/
│       └── Logger.swift
├── DesignSystem/
│   ├── Theme/
│   │   ├── Theme.swift
│   │   ├── AppColors.swift
│   │   └── AppTypography.swift
│   ├── Components/
│   │   ├── GlassCard.swift
│   │   ├── PriceTicker.swift
│   │   ├── CandlestickChart.swift
│   │   ├── Sparkline.swift
│   │   ├── PnLBadge.swift
│   │   ├── RiskBadge.swift
│   │   ├── StatusIndicator.swift
│   │   ├── SkeletonView.swift
│   │   ├── ConfidenceBadge.swift
│   │   ├── MetricView.swift
│   │   ├── ErrorBanner.swift
│   │   └── LoadingOverlay.swift
│   ├── Motion/
│   │   ├── Animations.swift
│   │   └── Haptics.swift
│   └── Modifiers/
│       └── ViewModifiers.swift
├── Features/
│   ├── Dashboard/
│   │   ├── DashboardView.swift
│   │   ├── DashboardViewModel.swift
│   │   └── Components/
│   │       ├── EquityCard.swift
│   │       ├── PnLCard.swift
│   │       ├── PositionsList.swift
│   │       └── QuickActions.swift
│   ├── Markets/
│   │   ├── MarketsView.swift
│   │   ├── MarketsViewModel.swift
│   │   ├── MarketDetailView.swift
│   │   └── Components/
│   │       ├── MarketCard.swift
│   │       └── MarketHeatmap.swift
│   ├── Trade/
│   │   ├── TradeView.swift
│   │   ├── TradeViewModel.swift
│   │   ├── OrderEntrySheet.swift
│   │   └── Components/
│   │       ├── OrderTypeSelector.swift
│   │       ├── SideSelector.swift
│   │       └── QuantityInput.swift
│   ├── Positions/
│   │   ├── PositionsView.swift
│   │   ├── PositionsViewModel.swift
│   │   ├── PositionDetailView.swift
│   │   └── Components/
│   │       ├── PositionCard.swift
│   │       └── TPSLSheet.swift
│   ├── Strategies/
│   │   ├── StrategiesView.swift
│   │   ├── StrategiesViewModel.swift
│   │   ├── StrategyDetailView.swift
│   │   └── Components/
│   │       ├── StrategyCard.swift
│   │       └── StrategyPerformanceChart.swift
│   ├── Risk/
│   │   ├── RiskDashboardView.swift
│   │   ├── RiskViewModel.swift
│   │   └── Components/
│   │       ├── RiskStateView.swift
│   │       ├── RiskProgressBar.swift
│   │       └── KillSwitchButton.swift
│   ├── Analytics/
│   │   ├── AnalyticsView.swift
│   │   ├── AnalyticsViewModel.swift
│   │   └── Components/
│   │       ├── PerformanceMetrics.swift
│   │       ├── EquityCurveChart.swift
│   │       └── TradeDistribution.swift
│   ├── Signals/
│   │   ├── SignalFeedView.swift
│   │   ├── SignalFeedViewModel.swift
│   │   ├── SignalDetailView.swift
│   │   └── Components/
│   │       ├── SignalCard.swift
│   │       └── SignalReasoningView.swift
│   ├── AIAssistant/
│   │   ├── AIAssistantView.swift
│   │   ├── AIAssistantViewModel.swift
│   │   └── Components/
│   │       ├── MessageBubble.swift
│   │       └── QuerySuggestions.swift
│   ├── Activity/
│   │   ├── ActivityView.swift
│   │   ├── ActivityViewModel.swift
│   │   └── Components/
│   │       ├── TradeHistoryCard.swift
│   │       └── EventTimeline.swift
│   └── Settings/
│       ├── SettingsView.swift
│       ├── SettingsViewModel.swift
│       └── Components/
│           ├── SettingRow.swift
│           └── APIConfigView.swift
├── Models/
│   ├── Market.swift
│   ├── Position.swift
│   ├── Order.swift
│   ├── Portfolio.swift
│   ├── Signal.swift
│   ├── RiskState.swift
│   ├── Strategy.swift
│   ├── TradingEvent.swift
│   ├── SystemStatus.swift
│   ├── WhyDiagnostics.swift
│   └── AnalyticsData.swift
├── Services/
│   ├── Protocols/
│   │   ├── MarketServiceProtocol.swift
│   │   ├── TradingServiceProtocol.swift
│   │   ├── PortfolioServiceProtocol.swift
│   │   ├── StrategyServiceProtocol.swift
│   │   ├── RiskServiceProtocol.swift
│   │   ├── SignalServiceProtocol.swift
│   │   └── EventServiceProtocol.swift
│   ├── Live/
│   │   ├── LiveMarketService.swift
│   │   ├── LiveTradingService.swift
│   │   ├── LivePortfolioService.swift
│   │   ├── LiveStrategyService.swift
│   │   ├── LiveRiskService.swift
│   │   ├── LiveSignalService.swift
│   │   └── LiveEventService.swift
│   └── Mock/
│       └── (mock implementations)
├── Widgets/
│   ├── HEANWidget.swift
│   ├── PnLWidget.swift
│   └── PositionsWidget.swift
├── WatchApp/
│   ├── HEANWatchApp.swift
│   ├── WatchDashboardView.swift
│   └── WatchPositionsView.swift
├── Extensions/
│   ├── Color+Hex.swift
│   ├── Date+Formatting.swift
│   ├── Double+Currency.swift
│   └── View+Modifiers.swift
└── Assets.xcassets/
```

---

**This is not just another trading app. This is the future of mobile trading.**
