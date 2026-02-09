//
//  WebSocketManager.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import Foundation
import Combine
import OSLog

/// WebSocket message with topic for routing
struct WebSocketMessage {
    let topic: String
    let data: Any?
    let rawData: Data

    init(topic: String, data: Any?, rawData: Data) {
        self.topic = topic
        self.data = data
        self.rawData = rawData
    }

    /// Parse from raw JSON data
    static func parse(from data: Data) -> WebSocketMessage? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let topic = json["topic"] as? String else {
            return nil
        }
        return WebSocketMessage(topic: topic, data: json["data"], rawData: data)
    }
}

enum WebSocketEvent {
    case connected
    case disconnected
    case message(Data)
    case error(Error)
}

@MainActor
class WebSocketManager: ObservableObject {
    @Published private(set) var isConnected = false
    @Published private(set) var lastError: Error?

    private let url: String
    private var webSocketTask: URLSessionWebSocketTask?
    private let eventSubject = PassthroughSubject<WebSocketEvent, Never>()
    private let messageSubject = PassthroughSubject<WebSocketMessage, Never>()
    private var reconnectTimer: Timer?
    private let maxReconnectDelay: TimeInterval = 30
    private var currentReconnectDelay: TimeInterval = 5

    /// Active topic subscriptions
    private var subscriptions: Set<String> = []

    var eventPublisher: AnyPublisher<WebSocketEvent, Never> {
        eventSubject.eraseToAnyPublisher()
    }

    /// Publisher for parsed WebSocket messages with topics
    var messagePublisher: AnyPublisher<WebSocketMessage, Never> {
        messageSubject.eraseToAnyPublisher()
    }

    init(url: String) {
        self.url = url
    }

    private lazy var session: URLSession = {
        URLSession(configuration: .default)
    }()

    func connect() {
        guard webSocketTask == nil else {
            Logger.websocket.info("WebSocket already connected")
            return
        }

        guard let url = URL(string: url) else {
            Logger.websocket.error("Invalid WebSocket URL: \(self.url)")
            return
        }

        Logger.websocket.info("Connecting to WebSocket: \(url.absoluteString)")

        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()

        // Start receiving — the first successful receive confirms the connection
        receiveMessage()
    }

    func disconnect() {
        Logger.websocket.info("Disconnecting WebSocket")

        reconnectTimer?.invalidate()
        reconnectTimer = nil

        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil

        isConnected = false
        eventSubject.send(.disconnected)
    }

    /// Subscribe to a topic for receiving messages
    func subscribe(topic: String) {
        subscriptions.insert(topic)

        if isConnected {
            sendSubscription(topic: topic)
        }

        Logger.websocket.info("Subscribed to topic: \(topic)")
    }

    /// Unsubscribe from a topic
    func unsubscribe(topic: String) {
        subscriptions.remove(topic)

        if isConnected {
            sendUnsubscription(topic: topic)
        }

        Logger.websocket.info("Unsubscribed from topic: \(topic)")
    }

    private func sendSubscription(topic: String) {
        let message: [String: Any] = [
            "action": "subscribe",
            "topic": topic
        ]

        if let data = try? JSONSerialization.data(withJSONObject: message) {
            send(data)
        }
    }

    private func sendUnsubscription(topic: String) {
        let message: [String: Any] = [
            "action": "unsubscribe",
            "topic": topic
        ]

        if let data = try? JSONSerialization.data(withJSONObject: message) {
            send(data)
        }
    }

    func send(_ message: String) {
        guard let data = message.data(using: .utf8) else { return }
        send(data)
    }

    func send(_ data: Data) {
        // Send as TEXT frame (required by FastAPI WebSocket)
        guard let text = String(data: data, encoding: .utf8) else { return }
        let message = URLSessionWebSocketTask.Message.string(text)
        webSocketTask?.send(message) { [weak self] error in
            if let error = error {
                Logger.websocket.error("Failed to send message: \(error.localizedDescription)")
                Task { @MainActor in
                    self?.lastError = error
                }
            }
        }
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self else { return }

            Task { @MainActor in
                switch result {
                case .success(let message):
                    // First successful receive = connection confirmed
                    if !self.isConnected {
                        self.isConnected = true
                        self.currentReconnectDelay = 5 // Reset backoff on success
                        self.eventSubject.send(.connected)
                        Logger.websocket.info("WebSocket connected successfully")

                        // Re-subscribe to all topics
                        for topic in self.subscriptions {
                            self.sendSubscription(topic: topic)
                        }
                    }

                    switch message {
                    case .data(let data):
                        self.handleReceivedData(data)
                    case .string(let text):
                        if let data = text.data(using: .utf8) {
                            self.handleReceivedData(data)
                        }
                    @unknown default:
                        Logger.websocket.warning("Unknown WebSocket message type")
                    }

                    // Continue receiving
                    self.receiveMessage()

                case .failure(let error):
                    Logger.websocket.error("WebSocket error: \(error.localizedDescription)")
                    self.lastError = error
                    self.eventSubject.send(.error(error))
                    self.handleDisconnection()
                }
            }
        }
    }

    private func handleReceivedData(_ data: Data) {
        // Send raw event
        eventSubject.send(.message(data))

        // Try to parse as WebSocketMessage with topic
        if let wsMessage = WebSocketMessage.parse(from: data) {
            // Only forward messages for subscribed topics
            if subscriptions.contains(wsMessage.topic) {
                messageSubject.send(wsMessage)
                Logger.websocket.debug("Received message for topic: \(wsMessage.topic)")
            }
        }
    }

    private func handleDisconnection() {
        isConnected = false
        webSocketTask = nil
        eventSubject.send(.disconnected)

        // Schedule reconnection with exponential backoff
        scheduleReconnect()
    }

    private func scheduleReconnect() {
        reconnectTimer?.invalidate()

        Logger.websocket.info("Scheduling reconnect in \(self.currentReconnectDelay)s")

        reconnectTimer = Timer.scheduledTimer(withTimeInterval: currentReconnectDelay, repeats: false) { [weak self] _ in
            guard let self = self else { return }
            Task { @MainActor in
                self.connect()
            }
        }

        // Exponential backoff
        currentReconnectDelay = min(currentReconnectDelay * 2, maxReconnectDelay)
    }
}
