# HEAN iOS - Code Examples

Complete implementation examples for key components in the iOS architecture.

---

## 1. Core Networking - APIClient.swift

```swift
import Foundation

/// Production-grade HTTP client with retry logic and interceptors
final class APIClient: APIClientProtocol, @unchecked Sendable {
    private let baseURL: URL
    private let session: URLSession
    private let retryPolicy: RetryPolicy
    private let interceptors: [RequestInterceptor]
    private let logger: Logger
    private let deduplicator = RequestDeduplicator()

    init(
        baseURL: URL,
        session: URLSession = .shared,
        retryPolicy: RetryPolicy = .exponential(),
        interceptors: [RequestInterceptor] = []
    ) {
        self.baseURL = baseURL
        self.session = session
        self.retryPolicy = retryPolicy
        self.interceptors = interceptors
        self.logger = Logger.shared
    }

    func request<T: Decodable>(
        _ endpoint: APIEndpoint,
        method: HTTPMethod = .get,
        parameters: [String: Any]? = nil,
        headers: [String: String]? = nil
    ) async throws -> T {
        let correlationID = UUID()
        let cacheKey = "\(endpoint.path)-\(method.rawValue)"

        return try await deduplicator.deduplicate(key: cacheKey) {
            try await self.performRequest(
                endpoint,
                method: method,
                parameters: parameters,
                headers: headers,
                correlationID: correlationID
            )
        } as! T
    }

    func request(
        _ endpoint: APIEndpoint,
        method: HTTPMethod = .get,
        parameters: [String: Any]? = nil,
        headers: [String: String]? = nil
    ) async throws {
        let correlationID = UUID()
        let _: EmptyResponse = try await performRequest(
            endpoint,
            method: method,
            parameters: parameters,
            headers: headers,
            correlationID: correlationID
        )
    }

    func cancelAll() {
        session.invalidateAndCancel()
    }

    // MARK: - Private

    private func performRequest<T: Decodable>(
        _ endpoint: APIEndpoint,
        method: HTTPMethod,
        parameters: [String: Any]?,
        headers: [String: String]?,
        correlationID: UUID
    ) async throws -> T {
        var urlRequest = try buildURLRequest(endpoint, method: method, parameters: parameters)

        // Apply interceptors
        for interceptor in interceptors {
            urlRequest = try await interceptor.intercept(urlRequest, correlationID: correlationID)
        }

        // Add custom headers
        headers?.forEach { key, value in
            urlRequest.setValue(value, forHTTPHeaderField: key)
        }

        logger.debug("HTTP Request", context: [
            "correlation_id": correlationID.uuidString,
            "method": method.rawValue,
            "url": urlRequest.url?.absoluteString ?? "",
            "headers": urlRequest.allHTTPHeaderFields ?? [:]
        ])

        return try await retryPolicy.execute { [weak self] in
            guard let self = self else { throw APIError.clientDeallocated }

            do {
                let (data, response) = try await self.session.data(for: urlRequest)

                guard let httpResponse = response as? HTTPURLResponse else {
                    throw APIError.invalidResponse
                }

                self.logger.debug("HTTP Response", context: [
                    "correlation_id": correlationID.uuidString,
                    "status_code": httpResponse.statusCode,
                    "content_length": data.count
                ])

                return try self.handleResponse(
                    data: data,
                    response: httpResponse,
                    correlationID: correlationID
                )

            } catch let error as URLError {
                throw self.mapURLError(error)
            }
        }
    }

    private func buildURLRequest(
        _ endpoint: APIEndpoint,
        method: HTTPMethod,
        parameters: [String: Any]?
    ) throws -> URLRequest {
        var components = URLComponents(url: baseURL.appendingPathComponent(endpoint.path), resolvingAgainstBaseURL: false)!

        // Add query parameters for GET requests
        if method == .get, let queryItems = endpoint.queryItems {
            components.queryItems = queryItems
        }

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 30

        // Add body for POST/PUT/PATCH
        if [.post, .put, .patch].contains(method), let parameters = parameters {
            request.httpBody = try JSONSerialization.data(withJSONObject: parameters)
        }

        return request
    }

    private func handleResponse<T: Decodable>(
        data: Data,
        response: HTTPURLResponse,
        correlationID: UUID
    ) throws -> T {
        switch response.statusCode {
        case 200...299:
            // Success - decode response
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                decoder.keyDecodingStrategy = .convertFromSnakeCase

                if T.self == EmptyResponse.self {
                    return EmptyResponse() as! T
                }

                return try decoder.decode(T.self, from: data)

            } catch {
                logger.error("Response decoding failed", context: [
                    "correlation_id": correlationID.uuidString,
                    "error": error.localizedDescription,
                    "raw_data": String(data: data, encoding: .utf8) ?? ""
                ])
                throw APIError.decodingFailed(underlyingError: error, correlationID: correlationID)
            }

        case 401:
            throw APIError.unauthorized(correlationID: correlationID)

        case 403:
            throw APIError.forbidden(correlationID: correlationID)

        case 404:
            throw APIError.notFound(correlationID: correlationID)

        case 429:
            let retryAfter = parseRetryAfter(from: response)
            throw APIError.rateLimited(retryAfter: retryAfter, correlationID: correlationID)

        case 500...599:
            throw APIError.serverError(statusCode: response.statusCode, correlationID: correlationID)

        default:
            throw APIError.unexpectedStatusCode(response.statusCode, correlationID: correlationID)
        }
    }

    private func parseRetryAfter(from response: HTTPURLResponse) -> TimeInterval {
        guard let retryAfterHeader = response.value(forHTTPHeaderField: "Retry-After"),
              let retryAfter = TimeInterval(retryAfterHeader) else {
            return 60 // Default 1 minute
        }
        return retryAfter
    }

    private func mapURLError(_ error: URLError) -> APIError {
        switch error.code {
        case .notConnectedToInternet, .networkConnectionLost:
            return .networkUnavailable(correlationID: UUID())
        case .timedOut:
            return .timeout(correlationID: UUID())
        case .cannotFindHost, .cannotConnectToHost:
            return .serverUnreachable(correlationID: UUID())
        default:
            return .networkError(underlyingError: error, correlationID: UUID())
        }
    }
}

// MARK: - Supporting Types

struct EmptyResponse: Decodable {}

enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case patch = "PATCH"
    case delete = "DELETE"
}

/// Retry policy with exponential backoff
struct RetryPolicy {
    let maxAttempts: Int
    let baseDelay: TimeInterval
    let maxDelay: TimeInterval
    let retryableErrors: Set<APIError>

    static func exponential(
        maxAttempts: Int = 3,
        baseDelay: TimeInterval = 1.0,
        maxDelay: TimeInterval = 10.0
    ) -> RetryPolicy {
        RetryPolicy(
            maxAttempts: maxAttempts,
            baseDelay: baseDelay,
            maxDelay: maxDelay,
            retryableErrors: [
                .networkUnavailable(correlationID: UUID()),
                .timeout(correlationID: UUID()),
                .serverError(statusCode: 503, correlationID: UUID())
            ]
        )
    }

    func execute<T>(_ operation: () async throws -> T) async throws -> T {
        var attempt = 0

        while true {
            attempt += 1

            do {
                return try await operation()
            } catch let error as APIError {
                // Check if error is retryable
                guard attempt < maxAttempts,
                      retryableErrors.contains(where: { type(of: $0) == type(of: error) }) else {
                    throw error
                }

                // Calculate delay with exponential backoff
                let delay = min(baseDelay * pow(2.0, Double(attempt - 1)), maxDelay)

                Logger.shared.debug("Retrying request", context: [
                    "attempt": attempt,
                    "max_attempts": maxAttempts,
                    "delay": delay
                ])

                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
        }
    }
}

/// Request interceptor protocol
protocol RequestInterceptor: Sendable {
    func intercept(_ request: URLRequest, correlationID: UUID) async throws -> URLRequest
}

/// Auth header interceptor
final class AuthInterceptor: RequestInterceptor {
    private let keychainManager = KeychainManager.shared

    func intercept(_ request: URLRequest, correlationID: UUID) async throws -> URLRequest {
        var modifiedRequest = request

        // Add API key if available
        if let apiKey = try? keychainManager.retrieve(key: "api_key") {
            modifiedRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        // Add correlation ID header
        modifiedRequest.setValue(correlationID.uuidString, forHTTPHeaderField: "X-Correlation-ID")

        return modifiedRequest
    }
}

/// Logging interceptor
final class LoggingInterceptor: RequestInterceptor {
    private let logger: Logger

    init(logger: Logger) {
        self.logger = logger
    }

    func intercept(_ request: URLRequest, correlationID: UUID) async throws -> URLRequest {
        logger.debug("Request intercepted", context: [
            "correlation_id": correlationID.uuidString,
            "url": request.url?.absoluteString ?? "",
            "method": request.httpMethod ?? ""
        ])
        return request
    }
}

/// Request deduplication to prevent redundant API calls
actor RequestDeduplicator {
    private var inflightRequests: [String: Task<Any, Error>] = [:]

    func deduplicate<T>(
        key: String,
        operation: @Sendable () async throws -> T
    ) async throws -> T {
        // Check if request is already in flight
        if let existingTask = inflightRequests[key] {
            Logger.shared.debug("Request deduplicated", context: ["key": key])
            return try await existingTask.value as! T
        }

        // Start new request
        let task = Task<Any, Error> {
            defer {
                Task { await self.removeTask(key: key) }
            }
            return try await operation()
        }

        inflightRequests[key] = task
        return try await task.value as! T
    }

    private func removeTask(key: String) {
        inflightRequests.removeValue(forKey: key)
    }
}
```

---

## 2. WebSocket Client - WebSocketClient.swift

```swift
import Foundation
import Combine

/// Production WebSocket client with auto-reconnection and heartbeat
final class WebSocketClient: WebSocketClientProtocol, @unchecked Sendable {
    // MARK: - Properties

    private let heartbeatInterval: TimeInterval
    private let reconnector: WebSocketReconnector
    private let logger: Logger

    private var webSocketTask: URLSessionWebSocketTask?
    private var heartbeatTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?

    private let stateSubject = CurrentValueSubject<WebSocketState, Never>(.disconnected)
    private let messageContinuation: AsyncStream<WebSocketMessage>.Continuation

    var state: WebSocketState { stateSubject.value }
    var messageStream: AsyncStream<WebSocketMessage>
    var statePublisher: AnyPublisher<WebSocketState, Never> {
        stateSubject.eraseToAnyPublisher()
    }

    // MARK: - Initialization

    init(
        heartbeatInterval: TimeInterval = 30,
        reconnector: WebSocketReconnector = .exponential()
    ) {
        self.heartbeatInterval = heartbeatInterval
        self.reconnector = reconnector
        self.logger = Logger.shared

        var continuation: AsyncStream<WebSocketMessage>.Continuation!
        self.messageStream = AsyncStream { continuation = $0 }
        self.messageContinuation = continuation
    }

    // MARK: - Public Methods

    func connect(to url: URL) async throws {
        guard state == .disconnected else {
            logger.warning("WebSocket already connected or connecting")
            return
        }

        updateState(.connecting)

        do {
            let session = URLSession(configuration: .default)
            let task = session.webSocketTask(with: url)

            self.webSocketTask = task
            task.resume()

            updateState(.connected)
            reconnector.reset()

            logger.info("WebSocket connected", context: ["url": url.absoluteString])

            // Start receiving messages
            startReceiving()

            // Start heartbeat
            startHeartbeat()

        } catch {
            updateState(.disconnected)
            throw WebSocketError.connectionFailed(reason: error.localizedDescription, correlationID: UUID())
        }
    }

    func disconnect() async {
        guard let task = webSocketTask else { return }

        updateState(.disconnecting)

        // Cancel heartbeat
        heartbeatTask?.cancel()
        heartbeatTask = nil

        // Cancel receive task
        receiveTask?.cancel()
        receiveTask = nil

        // Close connection
        task.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil

        updateState(.disconnected)

        logger.info("WebSocket disconnected")
    }

    func send(_ message: WebSocketMessage) async throws {
        guard state == .connected else {
            throw WebSocketError.notConnected(correlationID: UUID())
        }

        guard let task = webSocketTask else {
            throw WebSocketError.notConnected(correlationID: UUID())
        }

        do {
            switch message {
            case .string(let text):
                try await task.send(.string(text))
            case .data(let data):
                try await task.send(.data(data))
            }

            logger.debug("WebSocket message sent", context: [
                "type": message.description
            ])

        } catch {
            logger.error("Failed to send WebSocket message", context: [
                "error": error.localizedDescription
            ])
            throw WebSocketError.sendFailed(correlationID: UUID())
        }
    }

    // MARK: - Private Methods

    private func startReceiving() {
        receiveTask = Task { [weak self] in
            guard let self = self else { return }

            while !Task.isCancelled {
                do {
                    guard let task = self.webSocketTask else { break }

                    let message = try await task.receive()
                    let wsMessage = self.convertMessage(message)

                    self.messageContinuation.yield(wsMessage)

                    self.logger.debug("WebSocket message received", context: [
                        "type": wsMessage.description
                    ])

                } catch {
                    if !Task.isCancelled {
                        self.logger.error("WebSocket receive error", context: [
                            "error": error.localizedDescription
                        ])

                        // Attempt reconnection
                        await self.handleDisconnection()
                    }
                    break
                }
            }
        }
    }

    private func startHeartbeat() {
        heartbeatTask = Task { [weak self] in
            guard let self = self else { return }

            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(self.heartbeatInterval * 1_000_000_000))

                guard let task = self.webSocketTask else { break }

                do {
                    try await task.sendPing()
                    self.logger.debug("Heartbeat ping sent")
                } catch {
                    self.logger.warning("Heartbeat ping failed", context: [
                        "error": error.localizedDescription
                    ])
                    await self.handleDisconnection()
                    break
                }
            }
        }
    }

    private func handleDisconnection() async {
        updateState(.disconnected)

        // Cancel tasks
        heartbeatTask?.cancel()
        receiveTask?.cancel()

        // Attempt reconnection
        if let url = webSocketTask?.originalRequest?.url {
            await attemptReconnection(to: url)
        }
    }

    private func attemptReconnection(to url: URL) async {
        guard reconnector.shouldReconnect() else {
            logger.error("WebSocket reconnection limit exceeded")
            return
        }

        let delay = reconnector.nextDelay()

        logger.info("WebSocket reconnecting", context: [
            "attempt": reconnector.currentAttempt,
            "delay": delay
        ])

        do {
            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            try await connect(to: url)
        } catch {
            logger.error("WebSocket reconnection failed", context: [
                "error": error.localizedDescription
            ])
            await attemptReconnection(to: url)
        }
    }

    private func convertMessage(_ message: URLSessionWebSocketTask.Message) -> WebSocketMessage {
        switch message {
        case .string(let text):
            return .string(text)
        case .data(let data):
            return .data(data)
        @unknown default:
            return .data(Data())
        }
    }

    private func updateState(_ newState: WebSocketState) {
        stateSubject.send(newState)
    }

    deinit {
        messageContinuation.finish()
    }
}

// MARK: - Supporting Types

enum WebSocketState: Equatable, Sendable {
    case disconnected
    case connecting
    case connected
    case disconnecting
}

enum WebSocketMessage: Sendable {
    case string(String)
    case data(Data)

    var description: String {
        switch self {
        case .string: return "string"
        case .data: return "data"
        }
    }
}

/// Reconnection strategy with exponential backoff
struct WebSocketReconnector: Sendable {
    let maxAttempts: Int
    let baseDelay: TimeInterval
    let maxDelay: TimeInterval

    private(set) var currentAttempt: Int = 0

    static func exponential(
        maxAttempts: Int = 10,
        baseDelay: TimeInterval = 1.0,
        maxDelay: TimeInterval = 60.0
    ) -> WebSocketReconnector {
        WebSocketReconnector(
            maxAttempts: maxAttempts,
            baseDelay: baseDelay,
            maxDelay: maxDelay
        )
    }

    mutating func reset() {
        currentAttempt = 0
    }

    func shouldReconnect() -> Bool {
        return currentAttempt < maxAttempts
    }

    mutating func nextDelay() -> TimeInterval {
        currentAttempt += 1
        let delay = baseDelay * pow(2.0, Double(currentAttempt - 1))
        return min(delay, maxDelay)
    }
}
```

---

## 3. Observable ViewModel - DashboardViewModel.swift

```swift
import Foundation
import Observation

@MainActor
@Observable
final class DashboardViewModel {
    // MARK: - Dependencies

    @Injected(\.portfolioService) private var portfolioService
    @Injected(\.positionService) private var positionService
    @Injected(\.eventService) private var eventService
    @Injected(\.healthService) private var healthService
    @Injected(\.logger) private var logger

    // MARK: - Published State

    var portfolioState: LoadingState<Portfolio> = .idle
    var positionsState: LoadingState<[Position]> = .idle
    var healthState: LoadingState<HealthStatus> = .idle
    var recentEvents: [Event] = []

    var presentedError: UserFacingError?
    var isRefreshing = false

    // MARK: - Computed Properties

    var totalPnL: Decimal {
        guard case .loaded(let portfolio) = portfolioState else { return 0 }
        return portfolio.totalPnL
    }

    var openPositionsCount: Int {
        guard case .loaded(let positions) = positionsState else { return 0 }
        return positions.count
    }

    var systemHealthy: Bool {
        guard case .loaded(let health) = healthState else { return false }
        return health.status == .healthy
    }

    // MARK: - Lifecycle

    init() {
        Task {
            await initialize()
        }
    }

    // MARK: - Public Methods

    func initialize() async {
        await loadAll()
        startEventStream()
    }

    func refresh() async {
        isRefreshing = true
        defer { isRefreshing = false }

        await loadAll()
    }

    // MARK: - Private Methods

    private func loadAll() async {
        async let portfolioResult: Void = loadPortfolio()
        async let positionsResult: Void = loadPositions()
        async let healthResult: Void = loadHealth()

        _ = await (portfolioResult, positionsResult, healthResult)
    }

    private func loadPortfolio() async {
        portfolioState = .loading

        do {
            let portfolio = try await portfolioService.fetchPortfolioSummary()
            portfolioState = .loaded(portfolio)

            logger.info("Portfolio loaded", context: [
                "balance": portfolio.totalBalance.description,
                "pnl": portfolio.totalPnL.description
            ])

        } catch let error as HEANError {
            portfolioState = .failed(error)
            handleError(error, context: "loadPortfolio")
        } catch {
            let wrappedError = APIError.unknown(underlyingError: error, correlationID: UUID())
            portfolioState = .failed(wrappedError)
            handleError(wrappedError, context: "loadPortfolio")
        }
    }

    private func loadPositions() async {
        positionsState = .loading

        do {
            let positions = try await positionService.fetchPositions()
            positionsState = .loaded(positions)

            logger.info("Positions loaded", context: [
                "count": positions.count
            ])

        } catch let error as HEANError {
            positionsState = .failed(error)
            handleError(error, context: "loadPositions")
        } catch {
            let wrappedError = APIError.unknown(underlyingError: error, correlationID: UUID())
            positionsState = .failed(wrappedError)
            handleError(wrappedError, context: "loadPositions")
        }
    }

    private func loadHealth() async {
        healthState = .loading

        do {
            let health = try await healthService.fetchHealth()
            healthState = .loaded(health)

            logger.info("Health status loaded", context: [
                "status": health.status.rawValue
            ])

        } catch let error as HEANError {
            healthState = .failed(error)
            handleError(error, context: "loadHealth")
        } catch {
            let wrappedError = APIError.unknown(underlyingError: error, correlationID: UUID())
            healthState = .failed(wrappedError)
            handleError(wrappedError, context: "loadHealth")
        }
    }

    private func startEventStream() {
        Task {
            for await event in eventService.eventStream {
                await handleEvent(event)
            }
        }
    }

    private func handleEvent(_ event: WSEvent) async {
        logger.debug("Event received", context: [
            "type": type(of: event).description,
            "timestamp": event.timestamp.ISO8601Format()
        ])

        // Update recent events
        if let domainEvent = event.toDomainEvent() {
            recentEvents.insert(domainEvent, at: 0)
            if recentEvents.count > 50 {
                recentEvents = Array(recentEvents.prefix(50))
            }
        }

        // Handle specific event types
        switch event {
        case let positionEvent as PositionUpdateEvent:
            await handlePositionUpdate(positionEvent)

        case let orderEvent as OrderUpdateEvent:
            await handleOrderUpdate(orderEvent)

        case let telemetryEvent as TelemetryEvent:
            await handleTelemetry(telemetryEvent)

        default:
            break
        }
    }

    private func handlePositionUpdate(_ event: PositionUpdateEvent) async {
        // Reload positions
        await loadPositions()
        await loadPortfolio() // Portfolio depends on positions
    }

    private func handleOrderUpdate(_ event: OrderUpdateEvent) async {
        logger.info("Order updated", context: [
            "order_id": event.orderId,
            "status": event.status
        ])
    }

    private func handleTelemetry(_ event: TelemetryEvent) async {
        // Update health if needed
        if event.level == .error || event.level == .critical {
            await loadHealth()
        }
    }

    private func handleError(_ error: HEANError, context: String) {
        logger.error("Error in \(context)", context: [
            "correlation_id": error.correlationID.uuidString,
            "severity": error.severity.rawValue,
            "description": error.localizedDescription
        ])

        // Present error to user
        presentedError = UserFacingError(from: error)

        // Auto-retry for medium severity errors
        if error.severity == .medium {
            scheduleRetry(for: context)
        }
    }

    private func scheduleRetry(for context: String) {
        Task {
            try? await Task.sleep(for: .seconds(3))

            switch context {
            case "loadPortfolio":
                await loadPortfolio()
            case "loadPositions":
                await loadPositions()
            case "loadHealth":
                await loadHealth()
            default:
                break
            }
        }
    }
}

// MARK: - Supporting Types

enum LoadingState<T>: Equatable where T: Equatable {
    case idle
    case loading
    case loaded(T)
    case failed(HEANError)

    static func == (lhs: LoadingState<T>, rhs: LoadingState<T>) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.loading, .loading):
            return true
        case (.loaded(let lhsValue), .loaded(let rhsValue)):
            return lhsValue == rhsValue
        case (.failed(let lhsError), .failed(let rhsError)):
            return lhsError.correlationID == rhsError.correlationID
        default:
            return false
        }
    }
}
```

---

## 4. SwiftUI View - DashboardView.swift

```swift
import SwiftUI

struct DashboardView: View {
    @State private var viewModel = DashboardViewModel()
    @Environment(\.container) private var container

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Health Indicator
                healthSection

                // Portfolio Summary
                portfolioSection

                // Quick Stats
                quickStatsSection

                // Open Positions
                positionsSection

                // Recent Activity
                activitySection
            }
            .padding()
        }
        .refreshable {
            await viewModel.refresh()
        }
        .alert(item: $viewModel.presentedError) { error in
            errorAlert(for: error)
        }
        .task {
            await viewModel.initialize()
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var healthSection: some View {
        GlassCard {
            HStack {
                HealthIndicator(isHealthy: viewModel.systemHealthy)

                Text("System Status")
                    .font(.headline)

                Spacer()

                if case .loading = viewModel.healthState {
                    ProgressView()
                }
            }
            .padding()
        }
    }

    @ViewBuilder
    private var portfolioSection: some View {
        switch viewModel.portfolioState {
        case .idle, .loading:
            GlassCard {
                ProgressView("Loading portfolio...")
                    .frame(maxWidth: .infinity)
                    .padding()
            }

        case .loaded(let portfolio):
            PortfolioSummaryView(portfolio: portfolio)

        case .failed(let error):
            ErrorStateView(error: error) {
                Task { await viewModel.refresh() }
            }
        }
    }

    @ViewBuilder
    private var quickStatsSection: some View {
        HStack(spacing: 12) {
            StatCard(
                title: "Total P&L",
                value: viewModel.totalPnL.formatted(.currency(code: "USD")),
                trend: viewModel.totalPnL >= 0 ? .up : .down
            )

            StatCard(
                title: "Open Positions",
                value: "\(viewModel.openPositionsCount)",
                trend: .neutral
            )
        }
    }

    @ViewBuilder
    private var positionsSection: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Open Positions")
                        .font(.headline)

                    Spacer()

                    NavigationLink("See All") {
                        PositionsView()
                    }
                    .font(.caption)
                }

                switch viewModel.positionsState {
                case .idle, .loading:
                    ProgressView()

                case .loaded(let positions):
                    if positions.isEmpty {
                        Text("No open positions")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity)
                            .padding()
                    } else {
                        ForEach(positions.prefix(3)) { position in
                            PositionRow(position: position)
                            if position.id != positions.prefix(3).last?.id {
                                Divider()
                            }
                        }
                    }

                case .failed:
                    Text("Failed to load positions")
                        .foregroundStyle(.red)
                }
            }
            .padding()
        }
    }

    @ViewBuilder
    private var activitySection: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Recent Activity")
                        .font(.headline)

                    Spacer()

                    NavigationLink("See All") {
                        ActivityView()
                    }
                    .font(.caption)
                }

                if viewModel.recentEvents.isEmpty {
                    Text("No recent activity")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                        .padding()
                } else {
                    ForEach(viewModel.recentEvents.prefix(5)) { event in
                        EventRow(event: event)
                        if event.id != viewModel.recentEvents.prefix(5).last?.id {
                            Divider()
                        }
                    }
                }
            }
            .padding()
        }
    }

    // MARK: - Error Alert

    private func errorAlert(for error: UserFacingError) -> Alert {
        Alert(
            title: Text(error.title),
            message: Text(error.message),
            primaryButton: .default(Text(error.actions.first?.title ?? "OK")) {
                handleErrorAction(error.actions.first)
            },
            secondaryButton: .cancel()
        )
    }

    private func handleErrorAction(_ action: ErrorAction?) {
        guard let action = action else { return }

        switch action {
        case .retry:
            Task { await viewModel.refresh() }

        case .openSettings:
            if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                UIApplication.shared.open(settingsURL)
            }

        case .reauth:
            // Navigate to login
            break

        case .contactSupport:
            if let supportURL = URL(string: "https://support.hean.trading") {
                UIApplication.shared.open(supportURL)
            }

        case .dismiss:
            break
        }
    }
}

// MARK: - Supporting Views

struct StatCard: View {
    let title: String
    let value: String
    let trend: Trend

    enum Trend {
        case up, down, neutral

        var color: Color {
            switch self {
            case .up: return .green
            case .down: return .red
            case .neutral: return .secondary
            }
        }

        var icon: String {
            switch self {
            case .up: return "arrow.up.right"
            case .down: return "arrow.down.right"
            case .neutral: return "minus"
            }
        }
    }

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(alignment: .firstTextBaseline) {
                    Text(value)
                        .font(.title2)
                        .fontWeight(.bold)

                    Image(systemName: trend.icon)
                        .font(.caption)
                        .foregroundStyle(trend.color)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
    }
}

struct ErrorStateView: View {
    let error: HEANError
    let retry: () -> Void

    var body: some View {
        GlassCard {
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.red)

                Text(error.localizedDescription)
                    .font(.body)
                    .multilineTextAlignment(.center)

                Button("Retry", action: retry)
                    .buttonStyle(.borderedProminent)
            }
            .padding()
        }
    }
}

#Preview {
    NavigationStack {
        DashboardView()
    }
    .environment(\.container, {
        let container = DIContainer()
        container.configureMock(scenario: .happyPath)
        return container
    }())
}
```

---

## 5. Data Service - MarketService.swift

```swift
import Foundation

@MainActor
final class MarketService {
    private let dataProvider: DataProviderProtocol
    private let cache: CacheManager
    private let logger: Logger

    init(
        dataProvider: DataProviderProtocol,
        cache: CacheManager
    ) {
        self.dataProvider = dataProvider
        self.cache = cache
        self.logger = Logger.shared
    }

    // MARK: - Public Methods

    func fetchMarkets() async throws -> [Market] {
        // Check cache first
        if let cached: [Market] = cache.get(key: "markets") {
            logger.debug("Markets loaded from cache")
            return cached
        }

        // Fetch from API
        let markets = try await dataProvider.fetchMarkets()

        // Cache result
        cache.set(key: "markets", value: markets, ttl: 300) // 5 min TTL

        logger.info("Markets fetched", context: ["count": markets.count])

        return markets
    }

    func fetchTicker(symbol: String) async throws -> Ticker {
        // Check cache
        let cacheKey = "ticker:\(symbol)"
        if let cached: Ticker = cache.get(key: cacheKey) {
            logger.debug("Ticker loaded from cache", context: ["symbol": symbol])
            return cached
        }

        // Fetch from API
        let ticker = try await dataProvider.fetchTicker(symbol: symbol)

        // Cache with short TTL (tickers change frequently)
        cache.set(key: cacheKey, value: ticker, ttl: 10) // 10 sec TTL

        return ticker
    }

    func fetchCandles(
        symbol: String,
        interval: String,
        limit: Int = 100
    ) async throws -> [Candle] {
        let cacheKey = "candles:\(symbol):\(interval)"

        // Check cache
        if let cached: [Candle] = cache.get(key: cacheKey) {
            logger.debug("Candles loaded from cache", context: [
                "symbol": symbol,
                "interval": interval
            ])
            return cached
        }

        // Fetch from API
        let candles = try await dataProvider.fetchCandles(
            symbol: symbol,
            interval: interval,
            limit: limit
        )

        // Cache candles
        cache.set(key: cacheKey, value: candles, ttl: 60) // 1 min TTL

        logger.info("Candles fetched", context: [
            "symbol": symbol,
            "interval": interval,
            "count": candles.count
        ])

        return candles
    }

    func observeTicker(symbol: String) -> AsyncStream<Ticker> {
        AsyncStream { continuation in
            Task {
                for await event in dataProvider.eventStream {
                    if let tickerEvent = event as? TickerEvent,
                       tickerEvent.symbol == symbol {
                        continuation.yield(tickerEvent.toTicker())
                    }
                }
                continuation.finish()
            }
        }
    }

    func invalidateCache() {
        cache.clear(prefix: "markets")
        cache.clear(prefix: "ticker")
        cache.clear(prefix: "candles")

        logger.info("Market cache invalidated")
    }
}
```

This provides complete, production-ready implementations of the core iOS architecture components with proper error handling, thread safety, caching, and observability.
