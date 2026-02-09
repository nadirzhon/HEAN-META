//
//  AIAssistantView.swift
//  HEAN
//
//  Natural language trading assistant - unique feature no competitor has
//

import SwiftUI

struct AIAssistantView: View {
    @StateObject var viewModel: AIAssistantViewModel
    @State private var query = ""
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            if viewModel.messages.isEmpty {
                                welcomeView
                            }

                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }

                            if viewModel.isLoading {
                                HStack(spacing: 8) {
                                    ProgressView().tint(Color(hex: "00D4FF")).scaleEffect(0.8)
                                    Text("Analyzing...").font(.caption).foregroundColor(.gray)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal)
                                .id("loading")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.messages.count) {
                        withAnimation(.spring(response: 0.3)) {
                            proxy.scrollTo(viewModel.messages.last?.id ?? "loading", anchor: .bottom)
                        }
                    }
                }

                // Quick Suggestions
                if viewModel.messages.isEmpty {
                    quickSuggestions
                }

                // Input Bar
                inputBar
            }
            .background(Color(hex: "0A0A0F").ignoresSafeArea())
            .navigationTitle("AI Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.clearMessages()
                    } label: {
                        Image(systemName: "trash")
                            .foregroundColor(.gray)
                    }
                }
            }
        }
    }

    // MARK: - Welcome

    private var welcomeView: some View {
        VStack(spacing: 20) {
            Spacer().frame(height: 40)

            Image(systemName: "sparkles")
                .font(.system(size: 50))
                .foregroundColor(Color(hex: "7B61FF"))
                .symbolEffect(.pulse)

            Text("HEAN AI Assistant")
                .font(.title2.bold())
                .foregroundColor(.white)

            Text("Ask me anything about your trading system.\nI analyze real-time data to give you actionable insights.")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer().frame(height: 20)
        }
    }

    // MARK: - Quick Suggestions

    private var quickSuggestions: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Quick Questions")
                .font(.caption.bold())
                .foregroundColor(.gray)
                .padding(.horizontal)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.suggestions, id: \.self) { suggestion in
                        Button {
                            query = suggestion
                            submitQuery()
                        } label: {
                            Text(suggestion)
                                .font(.caption)
                                .foregroundColor(Color(hex: "00D4FF"))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(Color(hex: "00D4FF").opacity(0.1))
                                .cornerRadius(16)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16)
                                        .stroke(Color(hex: "00D4FF").opacity(0.3), lineWidth: 0.5)
                                )
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding(.bottom, 8)
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        HStack(spacing: 12) {
            TextField("Ask about your trading...", text: $query)
                .textFieldStyle(.plain)
                .padding(12)
                .background(Color(hex: "1A1A2E"))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.gray.opacity(0.2), lineWidth: 0.5)
                )
                .focused($isInputFocused)
                .submitLabel(.send)
                .onSubmit { submitQuery() }

            Button {
                submitQuery()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .foregroundColor(query.isEmpty ? .gray : Color(hex: "00D4FF"))
            }
            .disabled(query.isEmpty || viewModel.isLoading)
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(Color(hex: "12121A"))
    }

    private func submitQuery() {
        guard !query.isEmpty else { return }
        let q = query
        query = ""
        Task { await viewModel.sendQuery(q) }
        Haptics.light()
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: AIMessage

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if message.isUser {
                Spacer(minLength: 60)
            } else {
                Image(systemName: "sparkles")
                    .font(.caption)
                    .foregroundColor(Color(hex: "7B61FF"))
                    .frame(width: 24, height: 24)
                    .background(Color(hex: "7B61FF").opacity(0.15))
                    .cornerRadius(12)
            }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .font(.subheadline)
                    .foregroundColor(.white)
                    .padding(12)
                    .background(
                        message.isUser
                            ? Color(hex: "00D4FF").opacity(0.2)
                            : Color(hex: "1A1A2E")
                    )
                    .cornerRadius(16)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(
                                message.isUser
                                    ? Color(hex: "00D4FF").opacity(0.3)
                                    : Color.gray.opacity(0.15),
                                lineWidth: 0.5
                            )
                    )

                Text(message.timestamp.formatted(.dateTime.hour().minute()))
                    .font(.caption2)
                    .foregroundColor(.gray.opacity(0.6))
            }

            if !message.isUser {
                Spacer(minLength: 40)
            }
        }
    }
}

// MARK: - AI Message Model

struct AIMessage: Identifiable {
    let id = UUID()
    let content: String
    let isUser: Bool
    let timestamp: Date
}

// MARK: - AI Assistant ViewModel

@MainActor
final class AIAssistantViewModel: ObservableObject {
    @Published var messages: [AIMessage] = []
    @Published var isLoading = false

    private let tradingService: TradingServiceProtocol

    let suggestions = [
        "Why am I not trading?",
        "What's my risk exposure?",
        "Show my best strategy",
        "Why were signals blocked?",
        "Explain my last trade"
    ]

    init(tradingService: TradingServiceProtocol) {
        self.tradingService = tradingService
    }

    func sendQuery(_ query: String) async {
        messages.append(AIMessage(content: query, isUser: true, timestamp: Date()))
        isLoading = true

        do {
            let diagnostics = try await tradingService.fetchWhyDiagnostics()
            let metrics = try await tradingService.fetchTradingMetrics()
            let response = generateResponse(query: query, diagnostics: diagnostics, metrics: metrics)
            messages.append(AIMessage(content: response, isUser: false, timestamp: Date()))
        } catch {
            messages.append(AIMessage(
                content: "I'm having trouble connecting to the trading system. Error: \(error.localizedDescription)",
                isUser: false,
                timestamp: Date()
            ))
        }

        isLoading = false
    }

    func clearMessages() {
        messages.removeAll()
    }

    private func generateResponse(query: String, diagnostics: WhyDiagnostics, metrics: TradingMetrics) -> String {
        let q = query.lowercased()

        if q.contains("not trading") || q.contains("why") {
            var response = ""
            if !diagnostics.engineRunning {
                response = "The trading engine is currently stopped. Start the engine to begin trading."
            } else if diagnostics.signalsBlocked > 0 {
                let reasons = diagnostics.blockReasons.map { "\($0.reason) (\($0.count)x)" }.joined(separator: "\n- ")
                response = "The engine is running but \(diagnostics.signalsBlocked) signals were blocked.\n\nBlock reasons:\n- \(reasons)"
            } else if diagnostics.signalsGenerated == 0 {
                response = "The engine is running but no signals have been generated yet. This could mean:\n- Market conditions don't meet strategy criteria\n- Strategies are still warming up\n- Confidence thresholds are too high"
            } else {
                response = "Trading is active! \(diagnostics.signalsGenerated) signals generated, \(diagnostics.ordersPlaced) orders placed, \(diagnostics.ordersFilled) filled."
            }

            if !diagnostics.recommendations.isEmpty {
                response += "\n\nRecommendations:\n" + diagnostics.recommendations.map { "- \($0)" }.joined(separator: "\n")
            }
            return response
        }

        if q.contains("risk") || q.contains("exposure") {
            return "Trading Funnel Status:\n- Signals: \(metrics.signalsDetected)\n- Orders: \(metrics.ordersCreated)\n- Filled: \(metrics.ordersFilled)\n- Blocked: \(metrics.signalsBlocked)\n\nCheck the Risk tab for detailed exposure metrics and risk state."
        }

        if q.contains("strategy") || q.contains("best") {
            return "Strategy performance:\n- Signals detected: \(metrics.signalsDetected)\n- Conversion rate: \(metrics.ordersCreated > 0 ? String(format: "%.1f%%", Double(metrics.ordersFilled) / Double(metrics.ordersCreated) * 100) : "N/A")\n\nGo to Strategies tab for per-strategy breakdown."
        }

        if q.contains("block") || q.contains("reject") {
            if diagnostics.signalsBlocked > 0 {
                let reasons = diagnostics.blockReasons.map { "- \($0.reason): \($0.count) times (\($0.category))" }.joined(separator: "\n")
                return "Blocked signals:\n\(reasons)\n\nTotal: \(diagnostics.signalsBlocked) blocked out of \(diagnostics.signalsGenerated) generated."
            }
            return "No signals have been blocked recently."
        }

        return "Here's your current trading status:\n- Engine: \(diagnostics.engineRunning ? "Running" : "Stopped")\n- Signals: \(metrics.signalsDetected)\n- Orders: \(metrics.ordersCreated)\n- Filled: \(metrics.ordersFilled)\n- Blocked: \(metrics.signalsBlocked)\n\nAsk me specifics like \"Why am I not trading?\" or \"What's my risk exposure?\" for detailed analysis."
    }
}
