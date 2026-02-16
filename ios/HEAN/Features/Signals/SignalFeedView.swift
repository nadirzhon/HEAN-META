//
//  SignalFeedView.swift
//  HEAN
//
//  Real-time signal feed with confidence scores and reasoning
//

import SwiftUI

struct SignalFeedView: View {
    @StateObject var viewModel: SignalFeedViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 12) {
                    if viewModel.signals.isEmpty {
                        emptyState
                    } else {
                        ForEach(viewModel.signals) { signal in
                            SignalCardView(signal: signal)
                                .transition(.asymmetric(
                                    insertion: .move(edge: .top).combined(with: .opacity),
                                    removal: .opacity
                                ))
                        }
                    }
                }
                .padding()
                .animation(.spring(response: 0.4, dampingFraction: 0.8), value: viewModel.signals.count)
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle("Signals")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(viewModel.isListening ? Theme.Colors.success : Theme.Colors.error)
                            .frame(width: 8, height: 8)
                        Text(viewModel.isListening ? "Live" : "Offline")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
            }
            .task {
                viewModel.startListening()
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer().frame(height: 60)
            Image(systemName: "antenna.radiowaves.left.and.right")
                .font(.system(size: 50))
                .foregroundColor(.gray)
                .symbolEffect(.variableColor, isActive: viewModel.isListening)

            Text(L.waitingForSignals)
                .font(.headline)
                .foregroundColor(.gray)

            Text(L.signalsWillAppear)
                .font(.caption)
                .foregroundColor(.gray.opacity(0.7))
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - Signal Card

struct SignalCardView: View {
    let signal: Signal
    @State private var isExpanded = false

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                // Header
                HStack {
                    // Side indicator
                    Text(signal.side.uppercased())
                        .font(.caption.bold())
                        .foregroundColor(signal.isBuy ? Theme.Colors.success : Theme.Colors.error)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background((signal.isBuy ? Theme.Colors.success : Theme.Colors.error).opacity(0.15))
                        .cornerRadius(6)

                    Text(signal.symbol)
                        .font(.system(.headline, design: .monospaced))
                        .foregroundColor(.white)

                    Spacer()

                    // Confidence badge
                    if let conf = signal.confidence {
                        ConfidenceBadgeView(confidence: conf)
                    }
                }

                // Strategy info
                HStack(spacing: 16) {
                    if let strategy = signal.strategy {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(L.strategy).font(.caption2).foregroundColor(.gray)
                            Text(strategy)
                                .font(.subheadline.weight(.semibold)).monospacedDigit()
                                .foregroundColor(.white)
                        }
                    }
                }

                // Reasoning
                if let reason = signal.reason, !reason.isEmpty {
                    Text(reason)
                        .font(.caption)
                        .foregroundColor(.gray)
                        .lineLimit(isExpanded ? nil : 2)
                        .onTapGesture { withAnimation { isExpanded.toggle() } }
                }

                // Footer
                HStack {
                    if let strategy = signal.strategy {
                        Label(strategy, systemImage: "brain.head.profile")
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.purple)
                    }

                    Spacer()

                    if let ts = signal.timestamp {
                        Text(ts.formatted(.relative(presentation: .named)))
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Confidence Badge

enum ConfidenceLevel {
    case high, medium, low

    var color: Color {
        switch self {
        case .high: return Theme.Colors.success
        case .medium: return Theme.Colors.warning
        case .low: return Theme.Colors.error
        }
    }
}

struct ConfidenceBadgeView: View {
    let confidence: Double

    var level: ConfidenceLevel {
        if confidence >= 0.7 { return .high }
        if confidence >= 0.4 { return .medium }
        return .low
    }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: level == .high ? "checkmark.circle.fill" :
                    level == .medium ? "minus.circle.fill" : "exclamationmark.circle.fill")
                .font(.caption2)
            Text(String(format: "%.0f%%", confidence * 100))
                .font(.caption.weight(.bold)).monospacedDigit()
        }
        .foregroundColor(level.color)
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(level.color.opacity(0.15))
        .cornerRadius(8)
    }
}

// MARK: - Signal Feed ViewModel

@MainActor
final class SignalFeedViewModel: ObservableObject {
    @Published var signals: [Signal] = []
    @Published var isListening = false

    private let signalService: SignalServiceProtocol

    init(signalService: SignalServiceProtocol) {
        self.signalService = signalService
    }

    func startListening() {
        signalService.subscribeToSignals()
        isListening = true

        // Observe new signals
        Task {
            for await signal in signalService.signalPublisher.values {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    signals.insert(signal, at: 0)
                    if signals.count > 50 { signals.removeLast() }
                }
                Haptics.light()
            }
        }
    }
}
