# Architecture Overview

## 🏗 Архитектура Trading Analytics Pro

Этот документ описывает архитектуру приложения на всех уровнях.

---

## 📐 Общая структура

```
┌─────────────────────────────────────────────────┐
│          Trading Analytics App                   │
│                   (SwiftUI)                      │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Views     │ │  ViewModels  │ │   Models     │
│   (SwiftUI)  │ │ (Observable) │ │  (Structs)   │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │       Data Layer              │
        │  (Files / API / Mock Data)    │
        └───────────────────────────────┘
```

---

## 🎯 MVVM Pattern

```
┌────────────────────────────────────────────────────┐
│                    View Layer                       │
│  ┌──────────────┐  ┌──────────────┐               │
│  │ DashboardView│  │AnalyticsView │  ...           │
│  └──────────────┘  └──────────────┘               │
│         │                  │                        │
│         └──────────────────┼────────────────┐      │
└────────────────────────────│────────────────┘      │
                             │ @EnvironmentObject    │
┌────────────────────────────│────────────────┐      │
│                    ViewModel Layer          │      │
│  ┌──────────────────────┐  ┌──────────────┐│      │
│  │ TradingDataManager   │  │AITradingAsst.││      │
│  │  @ObservableObject   │  │@Observable   ││      │
│  └──────────────────────┘  └──────────────┘│      │
│         │                          │        │      │
│         └──────────────────────────┼────────┘      │
└────────────────────────────────────│────────────┐  │
                                     │            │  │
┌────────────────────────────────────│────────────┘  │
│                    Model Layer     │               │
│  ┌─────────┐ ┌──────────────┐ ┌──────────┐       │
│  │  Trade  │ │BacktestResult│ │Performance│       │
│  └─────────┘ └──────────────┘ └──────────┘       │
└───────────────────────────────────────────────────┘
```

---

## 📊 Data Flow

### User Interaction Flow

```
User Tap
   │
   ▼
┌─────────┐
│  View   │ ──────┐
└─────────┘       │
                  │ Action
   ┌──────────────┘
   │
   ▼
┌──────────────┐
│  ViewModel   │ ──────┐
└──────────────┘       │
                       │ Update State
   ┌───────────────────┘
   │
   ▼
┌──────────────┐
│    Model     │
└──────────────┘
   │
   │ @Published
   ▼
┌──────────────┐
│  View Update │ (SwiftUI Auto-refresh)
└──────────────┘
```

### Data Loading Flow

```
App Launch
   │
   ▼
┌──────────────────┐
│  .task modifier  │
└──────────────────┘
   │
   ▼
┌─────────────────────────┐
│ TradingDataManager      │
│  .loadBacktestData()    │
└─────────────────────────┘
   │
   ├─────────────────┬─────────────────┐
   ▼                 ▼                 ▼
┌─────────┐    ┌─────────┐      ┌──────────┐
│Parse Log│    │Load Mock│      │Call API  │
└─────────┘    └─────────┘      └──────────┘
   │                 │                 │
   └─────────────────┼─────────────────┘
                     ▼
           ┌──────────────────┐
           │ Update @Published│
           │   Properties     │
           └──────────────────┘
                     │
                     ▼
           ┌──────────────────┐
           │  SwiftUI Update  │
           └──────────────────┘
```

---

## 🧩 Component Architecture

### Dashboard View Components

```
DashboardView
│
├── PerformanceMetricsGrid
│   ├── MetricCard (Return)
│   ├── MetricCard (Win Rate)
│   ├── MetricCard (Profit Factor)
│   └── MetricCard (Sharpe Ratio)
│
├── EquityCurveCard
│   ├── Header
│   ├── Chart (Line + Area)
│   └── Legend
│
├── DetailedStatsCard
│   ├── StatRow × 9
│   └── Dividers
│
└── TradeDistributionCard
    ├── DonutChart
    └── LegendItems
```

### Analytics View Components

```
AnalyticsView
│
├── VisualizationSelector
│   └── Button × 4 (with matched geometry)
│
├── MainVisualization
│   ├── Performance3DView
│   │   ├── Chart3D
│   │   │   └── SurfacePlot
│   │   └── PoseControls
│   │
│   ├── TradeHeatmapView
│   │   └── Chart (RectangleMark)
│   │
│   ├── ReturnsDistributionView
│   │   └── Chart (BarMark)
│   │
│   └── DrawdownAnalysisView
│       ├── Chart (Line + Area)
│       └── Stats
│
├── Chart3DControls
│   └── Button × 4 (Front/Top/Side/Default)
│
├── AdvancedMetricsCard
│   └── GridItem × 4
│
└── CorrelationMatrixView
    └── Matrix (4×4)
```

### Trades List Components

```
TradesListView
│
├── SearchBar
│   ├── TextField
│   └── Clear Button
│
├── FilterSelector
│   └── FilterChip × 5
│
├── TradesSummaryCard
│   └── SummaryItem × 3
│
├── TradesList (ScrollView)
│   └── TradeCard × N
│       ├── Symbol + Type
│       ├── Date
│       └── P&L
│
└── TradeDetailSheet (Modal)
    ├── Header
    ├── PriceCard
    ├── QuantityCard
    ├── DatesCard
    └── PerformanceCard
```

### AI Assistant Components

```
AIAssistantView
│
├── AIStatusCard
│   ├── AI Icon (animated)
│   ├── Status
│   └── Progress
│
├── InsightSelector
│   └── Button × 4
│
├── MainContent
│   ├── AnalysisOverview
│   │   ├── AIAnalysisCard
│   │   ├── KeyInsightsGrid
│   │   └── PerformanceGradeCard
│   │
│   ├── Recommendations
│   │   └── RecommendationCard × N
│   │
│   ├── PatternAnalysis
│   │   └── PatternCard × 3
│   │
│   └── RiskAssessment
│       ├── RiskLevelCard
│       └── RiskFactorsList
│
└── QuickActionsCard
    └── ActionButton × 2
```

---

## 🎨 Design System Architecture

### Glass Effect System

```
GlassEffectContainer
│
├── Configuration
│   ├── spacing: CGFloat
│   └── Content: View
│
├── Children (Glass Views)
│   └── Each has .glassEffect()
│       ├── .regular
│       ├── .tint(Color)
│       └── .interactive()
│
└── Merging Logic
    └── Based on spacing
```

### Animation System

```
Animation System
│
├── Spring Animations
│   ├── response: 0.3
│   ├── dampingFraction: 0.7
│   └── Use Cases:
│       ├── Tab switching
│       ├── Card appearance
│       └── State changes
│
├── Matched Geometry
│   ├── @Namespace
│   └── .matchedGeometryEffect()
│       ├── Tab indicators
│       └── Filter selectors
│
├── Symbol Effects
│   ├── .pulse
│   ├── .bounce
│   └── .rotate
│
└── Transitions
    ├── .opacity
    ├── .scale
    └── .slide
```

---

## 🔄 State Management

### Observable Pattern

```
┌──────────────────────────────────────┐
│      TradingDataManager              │
│      @MainActor @ObservableObject    │
│                                      │
│  @Published Properties:              │
│  ├── backtestResults: [BacktestR.]  │
│  ├── trades: [Trade]                │
│  ├── equityCurve: [EquityCurveP.]   │
│  ├── isLoading: Bool                │
│  └── errorMessage: String?          │
│                                      │
│  Computed:                           │
│  ├── currentBacktest               │
│  ├── openTrades                    │
│  └── closedTrades                  │
└──────────────────────────────────────┘
         │
         │ Changes trigger
         ▼
┌──────────────────────────────────────┐
│      SwiftUI View Hierarchy          │
│                                      │
│  Automatically re-renders when       │
│  @Published properties change        │
└──────────────────────────────────────┘
```

### Environment Objects

```
App Level
│
├── @StateObject dataManager
│   └── Injected via .environmentObject()
│
└── @StateObject aiAssistant
    └── Injected via .environmentObject()
        │
        ├── DashboardView
        │   └── @EnvironmentObject var dataManager
        │
        ├── AnalyticsView
        │   └── @EnvironmentObject var dataManager
        │
        ├── TradesListView
        │   └── @EnvironmentObject var dataManager
        │
        └── AIAssistantView
            ├── @EnvironmentObject var dataManager
            └── @EnvironmentObject var aiAssistant
```

---

## 📡 AI Integration Architecture

### Foundation Models Flow

```
AITradingAssistant
│
├── Initialization
│   ├── Check model.availability
│   └── Create LanguageModelSession
│       └── instructions: String
│
├── Analysis Request
│   ├── analyzeBacktestResults()
│   │   ├── Build prompt
│   │   ├── session.respond(to: prompt)
│   │   └── Parse response
│   │
│   └── analyzeEquityCurve()
│       ├── Calculate volatility
│       ├── Calculate trend
│       └── Generate insights
│
├── Recommendation Generation
│   ├── Check metrics
│   ├── Apply rules
│   └── Create TradingRecommendation
│       ├── title
│       ├── description
│       ├── priority
│       └── category
│
└── Output
    ├── currentAnalysis: String?
    └── recommendations: [TradingRecommendation]
```

### On-Device Processing

```
User Request
   │
   ▼
┌─────────────────────┐
│  AITradingAssistant │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  SystemLanguageModel│ (Apple Intelligence)
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  On-Device LLM      │ (No cloud!)
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  Response           │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  Update @Published  │
└─────────────────────┘
```

---

## 📊 Chart Architecture

### Swift Charts Integration

```
Chart Components
│
├── Line Charts
│   ├── LineMark
│   ├── AreaMark (gradient fill)
│   └── Customization
│       ├── .foregroundStyle()
│       ├── .lineStyle()
│       └── Gradient
│
├── Bar Charts
│   ├── BarMark
│   ├── .cornerRadius()
│   └── Gradient fills
│
├── Donut Charts
│   ├── SectorMark
│   ├── .innerRadius()
│   └── .angularInset()
│
└── Heatmaps
    ├── RectangleMark
    ├── .foregroundStyle(by:)
    └── Color scale
```

### 3D Chart System

```
Chart3D
│
├── SurfacePlot
│   ├── x: String
│   ├── y: String
│   ├── z: String
│   └── function: (Double, Double) -> Double
│
├── Customization
│   ├── .roughness()
│   ├── .foregroundStyle()
│   └── Gradient
│
├── Interaction
│   ├── .chart3DPose($pose)
│   │   ├── .default
│   │   ├── .front
│   │   ├── .top
│   │   └── .right
│   │
│   └── .chart3DCameraProjection()
│       ├── .automatic
│       ├── .perspective
│       └── .orthographic
│
└── User Interaction
    ├── Drag → Rotate
    ├── Pinch → Zoom
    └── Buttons → Preset views
```

---

## 🔐 Security Architecture

### Privacy-First Design

```
Data Flow
│
├── User Device
│   ├── Local Storage
│   │   ├── BacktestResults
│   │   ├── Trades
│   │   └── EquityCurve
│   │
│   ├── On-Device Processing
│   │   ├── AI Analysis
│   │   ├── Chart Rendering
│   │   └── Calculations
│   │
│   └── No Cloud Upload
│       └── All data stays local
│
└── Privacy Compliance
    ├── No tracking
    ├── No analytics
    └── User controls data
```

---

## 🚀 Performance Architecture

### Optimization Strategies

```
Performance Layer
│
├── View Optimization
│   ├── LazyVStack/LazyVGrid
│   ├── @ViewBuilder
│   └── Conditional rendering
│
├── Data Optimization
│   ├── Async loading
│   ├── Pagination
│   └── Data limiting
│
├── Chart Optimization
│   ├── Point decimation
│   ├── GPU rendering
│   └── Caching
│
└── Memory Management
    ├── Weak references
    ├── Unowned captures
    └── Value types (structs)
```

### Async/Await Pattern

```
UI Thread
   │
   │ .task { }
   ▼
┌──────────────┐
│ async method │
└──────────────┘
   │
   ├─ Background Task
   │
   ▼
await MainActor.run {
   Update @Published
}
   │
   ▼
UI Update (on main thread)
```

---

## 🧪 Testing Architecture

### Test Structure

```
Tests
│
├── Unit Tests
│   ├── Model Tests
│   │   ├── Trade calculations
│   │   └── Performance grading
│   │
│   └── ViewModel Tests
│       ├── Data loading
│       └── Filtering logic
│
├── Integration Tests
│   ├── Data flow
│   ├── AI integration
│   └── Complete workflows
│
└── Edge Cases
    ├── Empty data
    ├── Invalid inputs
    └── Extreme values
```

---

## 📦 Module Dependencies

```
TradingAnalyticsApp
│
├── Foundation
│   └── FoundationModels (AI)
│
├── SwiftUI
│   ├── Charts (2D + 3D)
│   └── Symbols
│
├── Combine
│   └── Publishers
│
└── Testing
    └── Swift Testing
```

---

## 🔄 Update Cycle

### View Lifecycle

```
View Appears
   │
   ▼
.onAppear { }
   │
   ▼
.task { }
   │ async
   ▼
Load Data
   │
   ▼
Update @Published
   │
   ▼
SwiftUI re-renders
   │
   ▼
Animations
   │
   ▼
User Interaction
   │
   ▼
State Change
   │
   └─→ (repeat)
```

---

## 🎯 Best Practices Applied

```
Architecture Principles
│
├── Single Responsibility
│   └── Each component has one job
│
├── Separation of Concerns
│   ├── Views → UI only
│   ├── ViewModels → Logic
│   └── Models → Data
│
├── DRY (Don't Repeat Yourself)
│   ├── Reusable components
│   └── Shared utilities
│
├── Dependency Injection
│   └── @EnvironmentObject
│
└── Testability
    ├── Mockable data
    └── Isolated components
```

---

**Документация актуальна на: 31 января 2026**  
**Версия приложения: 1.0.0**
