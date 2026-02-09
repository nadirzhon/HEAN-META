# HEAN iOS - Architecture Diagrams

Visual representations of the iOS application architecture.

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         iOS Application                              │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Presentation Layer                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │Dashboard │  │ Markets  │  │  Trade   │  │ Activity │   │   │
│  │  │   View   │  │   View   │  │   View   │  │   View   │   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │   │
│  │       │             │             │             │           │   │
│  │       ▼             ▼             ▼             ▼           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ViewModel │  │ViewModel │  │ViewModel │  │ViewModel │   │   │
│  │  │@Observable│  │@Observable│  │@Observable│  │@Observable│   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │   │
│  └───────┼──────────────┼──────────────┼──────────────┼────────┘   │
│          │              │              │              │            │
│  ┌───────┴──────────────┴──────────────┴──────────────┴────────┐   │
│  │                     Business Logic Layer                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Market   │  │  Order   │  │ Position │  │Portfolio │   │   │
│  │  │ Service  │  │ Service  │  │ Service  │  │ Service  │   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │   │
│  └───────┼──────────────┼──────────────┼──────────────┼────────┘   │
│          │              │              │              │            │
│  ┌───────┴──────────────┴──────────────┴──────────────┴────────┐   │
│  │                     Data Layer                               │   │
│  │                  ┌──────────────────┐                        │   │
│  │                  │  DataProvider    │                        │   │
│  │                  │  (Protocol)      │                        │   │
│  │                  └────────┬─────────┘                        │   │
│  │                           │                                  │   │
│  │          ┌────────────────┼────────────────┐                │   │
│  │          ▼                ▼                ▼                │   │
│  │    ┌──────────┐    ┌──────────┐    ┌──────────┐           │   │
│  │    │   API    │    │WebSocket │    │  Cache   │           │   │
│  │    │  Client  │    │  Client  │    │ Manager  │           │   │
│  │    └────┬─────┘    └────┬─────┘    └──────────┘           │   │
│  └─────────┼───────────────┼─────────────────────────────────┘   │
└────────────┼───────────────┼──────────────────────────────────────┘
             │               │
             ▼               ▼
      ┌───────────┐   ┌───────────┐
      │  Backend  │   │  Backend  │
      │   REST    │   │WebSocket  │
      │    API    │   │  Server   │
      └───────────┘   └───────────┘
```

---

## 2. Dependency Injection Flow

```
App Launch (HEANApp.swift)
         │
         ▼
  Check Environment
         │
    ┌────┴────┐
    ▼         ▼
 Mock?     Production?
    │         │
    ▼         ▼
configureMock()  configureProduction()
    │                  │
    └─────────┬────────┘
              ▼
      DIContainer.shared
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Logger  APIClient  WSClient
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  Cache   Keychain  UserDefaults
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
MockProvider  RealProvider
              │
    ┌─────────┴─────────┐
    ▼                   ▼
  Services        DataProvider
    │                   │
    └─────────┬─────────┘
              ▼
        ViewModels
              │
              ▼
           Views
```

---

## 3. Data Flow - REST API Request

```
User Interaction
    │
    ▼
View (SwiftUI)
    │
    ▼
ViewModel (call async method)
    │
    ▼
Service Layer (e.g., MarketService)
    │
    ├─> Check Cache? ──> Cache Hit ──> Return
    │                         │
    │                    Cache Miss
    ▼                         │
DataProvider                  │
    │                         │
    ▼                         │
APIClient.request()           │
    │                         │
    ├─> Check Deduplicator ──┘
    │       │
    │   In-flight? ──> Yes ──> Await existing
    │       │
    │      No
    ▼       │
Build URLRequest
    │
    ├─> Apply Interceptors
    │   ├─> AuthInterceptor (add token)
    │   ├─> LoggingInterceptor (log request)
    │   └─> CorrelationIDInterceptor (add correlation ID)
    │
    ▼
URLSession.data(for:)
    │
    ▼
Network Request ───────────> Backend Server
    │                               │
    ▼                               ▼
Response <─────────────────── HTTP Response
    │
    ▼
Handle Status Code
    │
    ├─> 200-299: Success
    │   └─> Decode JSON
    │       └─> Return Model
    │
    ├─> 401: Unauthorized
    │   └─> Throw APIError.unauthorized
    │
    ├─> 429: Rate Limited
    │   └─> Parse Retry-After
    │       └─> Throw APIError.rateLimited
    │
    ├─> 500-599: Server Error
    │   └─> Check RetryPolicy
    │       ├─> Retry? ──> Exponential Backoff ──> Retry
    │       └─> Max Retries ──> Throw Error
    │
    └─> Other: Unexpected
        └─> Throw APIError.unexpectedStatusCode
    │
    ▼
Return to Service
    │
    ├─> Cache Result
    │
    ▼
Return to ViewModel
    │
    ├─> Update @Observable state
    │
    ▼
SwiftUI View Updates (automatic via @Observable)
    │
    ▼
UI Renders
```

---

## 4. WebSocket Event Flow

```
WebSocket Server
    │
    │ (push event)
    ▼
WebSocketClient
    │
    ├─> State: connected?
    │   ├─> Yes: Receive message
    │   └─> No: Queue or drop
    │
    ▼
URLSessionWebSocketTask.receive()
    │
    ▼
Convert to WebSocketMessage
    │
    ▼
Yield to AsyncStream
    │
    ├─> MessageRouter (in DataProvider)
    │       │
    │       ├─> Decode EventEnvelope
    │       │
    │       ├─> Route by event.type
    │       │   ├─> "ticker" ──> TickerEvent
    │       │   ├─> "candle" ──> CandleEvent
    │       │   ├─> "order_update" ──> OrderUpdateEvent
    │       │   ├─> "position_update" ──> PositionUpdateEvent
    │       │   └─> "telemetry" ──> TelemetryEvent
    │       │
    │       └─> Yield to DataProvider.eventStream
    │
    ▼
Services observe eventStream
    │
    ├─> MarketService
    │   └─> Update ticker cache
    │
    ├─> OrderService
    │   └─> Notify order observers
    │
    └─> PositionService
        └─> Refresh positions
    │
    ▼
ViewModels observe Service streams
    │
    ├─> DashboardViewModel
    │   └─> for await event in eventService.eventStream
    │       └─> Update @Observable properties
    │
    └─> MarketsViewModel
        └─> for await ticker in marketService.observeTicker()
            └─> Update market prices
    │
    ▼
SwiftUI Views auto-update
    │
    ▼
UI Renders with new data
```

---

## 5. Error Handling Flow

```
Error Occurs (e.g., Network Failure)
    │
    ▼
APIClient catches URLError
    │
    ├─> Map to APIError
    │   ├─> NSURLErrorNotConnectedToInternet ──> APIError.networkUnavailable
    │   ├─> NSURLErrorTimedOut ──> APIError.timeout
    │   └─> Other ──> APIError.networkError
    │
    ▼
Add Correlation ID
    │
    ▼
Log Error (with context)
    │
    ├─> Logger.error("Request failed", context: [
    │       "correlation_id": uuid,
    │       "endpoint": path,
    │       "error": description
    │   ])
    │
    ▼
Throw APIError
    │
    ▼
Service Layer catches
    │
    ├─> Check RetryPolicy
    │   ├─> Retryable + attempts left?
    │   │   └─> Sleep (exponential backoff)
    │   │       └─> Retry operation
    │   │
    │   └─> Not retryable or max retries
    │       └─> Propagate error
    │
    ▼
ViewModel catches
    │
    ├─> Update state to .failed(error)
    │
    ├─> Map to UserFacingError
    │   ├─> Title: "Network Error"
    │   ├─> Message: "No internet connection"
    │   ├─> Severity: .medium
    │   └─> Actions: [.retry, .openSettings]
    │
    ├─> Set @Published presentedError
    │
    └─> Check severity
        ├─> .medium ──> scheduleRetry()
        └─> .high/.critical ──> No auto-retry
    │
    ▼
View observes presentedError
    │
    ▼
Show Alert
    │
    ├─> Display title + message
    │
    └─> Present action buttons
        ├─> "Retry" ──> viewModel.retry()
        ├─> "Settings" ──> openAppSettings()
        └─> "Dismiss" ──> dismiss alert
```

---

## 6. State Management (ViewModel)

```
┌────────────────────────────────────────────────────┐
│              ViewModel (@Observable)               │
│                                                    │
│  State Properties (auto-published):                │
│  ┌──────────────────────────────────────────┐    │
│  │ var portfolioState: LoadingState         │    │
│  │ var positionsState: LoadingState         │    │
│  │ var presentedError: UserFacingError?     │    │
│  │ var isLoading: Bool                      │    │
│  └──────────────────────────────────────────┘    │
│                     │                             │
│                     │ (changes trigger)           │
│                     ▼                             │
│  ┌──────────────────────────────────────────┐    │
│  │      SwiftUI View Auto-Update            │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  Dependencies (injected):                         │
│  ┌──────────────────────────────────────────┐    │
│  │ @Injected(\.portfolioService)            │    │
│  │ @Injected(\.eventService)                │    │
│  │ @Injected(\.logger)                      │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  Methods:                                         │
│  ┌──────────────────────────────────────────┐    │
│  │ func loadData() async                    │    │
│  │ func refresh() async                     │    │
│  │ func handleEvent(_ event: Event)         │    │
│  └──────────────────────────────────────────┘    │
└────────────────────────────────────────────────────┘
```

### LoadingState Enum

```
enum LoadingState<T> {
    case idle           // Initial state
    case loading        // Request in progress
    case loaded(T)      // Success with data
    case failed(Error)  // Error occurred
}

State Transitions:
    idle ──loadData()──> loading ──success──> loaded(data)
      │                    │
      │                    └──error──> failed(error)
      │                                      │
      └──────────refresh()─────────────────┘
```

---

## 7. Caching Architecture

```
┌────────────────────────────────────────────────────┐
│                 CacheManager                       │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │         Memory Cache (NSCache)           │    │
│  │  ┌────────────────────────────────┐     │    │
│  │  │ Key: "ticker:BTCUSDT"          │     │    │
│  │  │ Value: Ticker                  │     │    │
│  │  │ TTL: 10 seconds                │     │    │
│  │  └────────────────────────────────┘     │    │
│  │  ┌────────────────────────────────┐     │    │
│  │  │ Key: "markets"                 │     │    │
│  │  │ Value: [Market]                │     │    │
│  │  │ TTL: 5 minutes                 │     │    │
│  │  └────────────────────────────────┘     │    │
│  └──────────────────────────────────────────┘    │
│                     │                             │
│                     │ (on eviction/expiry)        │
│                     ▼                             │
│  ┌──────────────────────────────────────────┐    │
│  │          Disk Cache (FileManager)        │    │
│  │  ┌────────────────────────────────┐     │    │
│  │  │ File: candles_BTCUSDT_1h.json  │     │    │
│  │  │ Data: [Candle]                 │     │    │
│  │  │ TTL: 1 hour                    │     │    │
│  │  └────────────────────────────────┘     │    │
│  └──────────────────────────────────────────┘    │
└────────────────────────────────────────────────────┘

Cache Lookup Flow:
Request
    │
    ▼
Check Memory Cache
    ├─> Hit ──> Return (fast, ~1ms)
    │
    └─> Miss
        │
        ▼
    Check Disk Cache
        ├─> Hit ──> Load + Cache in Memory ──> Return
        │
        └─> Miss
            │
            ▼
        Fetch from Network
            │
            ▼
        Store in both caches
            │
            ▼
        Return
```

---

## 8. MVVM + Coordinator Pattern

```
┌────────────────────────────────────────────────────┐
│                  AppCoordinator                    │
│  (Root navigation coordinator)                     │
│                                                    │
│  TabView:                                         │
│  ┌──────────┬──────────┬──────────┬──────────┐   │
│  │Dashboard │ Markets  │  Trade   │ Activity │   │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘   │
└───────┼──────────┼──────────┼──────────┼──────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │Dashboard││ Markets ││  Trade  ││Activity │
   │Coordi-  ││Coordi-  ││Coordi-  ││Coordi-  │
   │nator    ││nator    ││nator    ││nator    │
   └────┬────┘└────┬────┘└────┬────┘└────┬────┘
        │          │          │          │
        │          │          │          │
        ▼          ▼          ▼          ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │Dashboard││ Markets ││  Trade  ││Activity │
   │ViewModel││ViewModel││ViewModel││ViewModel│
   └────┬────┘└────┬────┘└────┬────┘└────┬────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │Dashboard││ Markets ││  Trade  ││Activity │
   │  View   ││  View   ││  View   ││  View   │
   └─────────┘└─────────┘└─────────┘└─────────┘

Coordinator responsibilities:
  - Navigation flow
  - Screen transitions
  - Deep linking
  - Modal presentation

ViewModel responsibilities:
  - Business logic
  - State management
  - Data fetching
  - Event handling

View responsibilities:
  - UI rendering
  - User input
  - Animations
  - Layout
```

---

## 9. Thread Safety Model

```
┌────────────────────────────────────────────────────┐
│                  Main Thread                       │
│              (MainActor)                           │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │          UI Components                   │    │
│  │  - Views                                 │    │
│  │  - ViewModels                            │    │
│  │  - Coordinators                          │    │
│  └──────────────────────────────────────────┘    │
│                     │                             │
│                     │ (async calls)               │
│                     ▼                             │
└────────────────────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Actor   │    │  Actor   │    │  Actor   │
│  Cache   │    │Dedup     │    │  Router  │
│  Manager │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘
      │               │               │
      └───────────────┼───────────────┘
                      │
                      │ (isolated state)
                      ▼
              ┌────────────────┐
              │  Background    │
              │  Execution     │
              └────────────────┘

Rules:
1. All UI updates on MainActor
2. Actors protect shared mutable state
3. Sendable types for cross-actor data
4. No manual locks/semaphores needed
```

---

## 10. Testing Pyramid

```
                   ┌────────┐
                   │   UI   │ (10%)
                   │  Tests │
                   └────────┘
                  ┌──────────┐
                  │Integration│ (20%)
                  │   Tests   │
                  └──────────┘
               ┌──────────────┐
               │     Unit      │ (70%)
               │     Tests     │
               └──────────────┘

Unit Tests:
  - ViewModels (with mock services)
  - Services (with mock providers)
  - Utilities, Extensions
  - Pure functions

Integration Tests:
  - Full feature flows
  - Service + ViewModel interaction
  - WebSocket event handling
  - Multi-component scenarios

UI Tests:
  - Critical user paths
  - Order placement flow
  - Error recovery scenarios
  - Accessibility validation

SwiftUI Previews (not counted):
  - Visual testing
  - All states (loading, success, error, empty)
  - Design validation
```

---

## 11. Build & Deploy Pipeline

```
┌────────────────────────────────────────────────────┐
│             Developer Workflow                     │
└────────────────────────────────────────────────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Code Changes  │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   SwiftLint    │ (code style)
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   Unit Tests   │ (pass?)
              └────────┬───────┘
                       │
                       ├─> Fail ──> Fix Issues
                       │
                       └─> Pass
                           │
                           ▼
              ┌────────────────────┐
              │ Integration Tests  │ (pass?)
              └────────┬───────────┘
                       │
                       ├─> Fail ──> Fix Issues
                       │
                       └─> Pass
                           │
                           ▼
              ┌────────────────────┐
              │    UI Tests        │ (pass?)
              └────────┬───────────┘
                       │
                       ├─> Fail ──> Fix Issues
                       │
                       └─> Pass
                           │
                           ▼
              ┌────────────────────┐
              │  Build Archive     │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │ Upload TestFlight  │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  Internal Testing  │
              └────────┬───────────┘
                       │
                       ├─> Issues ──> Fix
                       │
                       └─> Approved
                           │
                           ▼
              ┌────────────────────┐
              │ External TestFlight│
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  App Store Review  │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  Production Release│
              └────────────────────┘
```

---

## Summary

These diagrams illustrate:

1. **High-Level Architecture**: Multi-layer design with clear separation
2. **Dependency Injection**: How components are wired together
3. **Data Flow**: Request/response lifecycle with caching
4. **WebSocket Events**: Real-time updates from server to UI
5. **Error Handling**: Comprehensive error propagation and recovery
6. **State Management**: Observable pattern with SwiftUI integration
7. **Caching**: Multi-level cache strategy
8. **MVVM+Coordinator**: Navigation and business logic separation
9. **Thread Safety**: Actor model for data race prevention
10. **Testing**: Pyramid approach with different test types
11. **Build Pipeline**: From code to production

All diagrams use ASCII art for easy viewing in any text editor or markdown viewer.
