//
//  StatusIndicator.swift
//  HEAN
//
//  Connection status indicator
//

import SwiftUI

/// Connection status indicator with optional latency display
struct StatusIndicator: View {
    let status: ConnectionStatus
    let latency: Int?
    let showLabel: Bool

    @State private var isPulsing = false

    enum ConnectionStatus {
        case connected
        case disconnected
        case reconnecting

        var color: Color {
            switch self {
            case .connected: return Theme.Colors.success
            case .disconnected: return Theme.Colors.error
            case .reconnecting: return Theme.Colors.warning
            }
        }

        var label: String {
            switch self {
            case .connected: return "Connected"
            case .disconnected: return "Disconnected"
            case .reconnecting: return "Reconnecting"
            }
        }

        var shouldPulse: Bool {
            self == .reconnecting
        }
    }

    init(
        status: ConnectionStatus,
        latency: Int? = nil,
        showLabel: Bool = true
    ) {
        self.status = status
        self.latency = latency
        self.showLabel = showLabel
    }

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            // Status dot
            ZStack {
                // Pulse ring for reconnecting
                if status.shouldPulse {
                    Circle()
                        .stroke(status.color.opacity(0.3), lineWidth: 2)
                        .frame(width: 12, height: 12)
                        .scaleEffect(isPulsing ? 1.8 : 1.0)
                        .opacity(isPulsing ? 0 : 1)
                }

                // Solid dot
                Circle()
                    .fill(status.color)
                    .frame(width: 8, height: 8)
            }

            if showLabel {
                // Status label
                Text(status.label)
                    .font(Theme.Typography.caption(13, weight: .medium))
                    .foregroundColor(Theme.Colors.textPrimary)

                // Latency (if available and connected)
                if let latency = latency, status == .connected {
                    Text("(\(latency)ms)")
                        .font(Theme.Typography.caption(11, weight: .regular))
                        .foregroundColor(Theme.Colors.textTertiary)
                }
            }
        }
        .onAppear {
            if status.shouldPulse {
                startPulseAnimation()
            }
        }
        .onChange(of: status) { _, _ in
            if status.shouldPulse {
                startPulseAnimation()
            } else {
                isPulsing = false
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        var text = "Status: \(status.label)"
        if let latency = latency, status == .connected {
            text += ", latency \(latency) milliseconds"
        }
        return text
    }

    private func startPulseAnimation() {
        withAnimation(Animation.easeOut(duration: 1.5).repeatForever(autoreverses: false)) {
            isPulsing = true
        }
    }
}

// MARK: - Preview
#Preview("StatusIndicator Variants") {
    ZStack {
        Theme.Colors.background
            .ignoresSafeArea()

        VStack(spacing: 20) {
            // With labels
            VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                Text("With Labels")
                    .font(Theme.Typography.headlineFont())
                    .foregroundColor(Theme.Colors.textPrimary)

                StatusIndicator(status: .connected, latency: 45, showLabel: true)
                StatusIndicator(status: .reconnecting, showLabel: true)
                StatusIndicator(status: .disconnected, showLabel: true)
            }
            .padding(Theme.Spacing.lg)
            .background(Theme.Colors.card)
            .cornerRadius(Theme.CornerRadius.md)

            // Without labels (dots only)
            VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                Text("Compact (No Labels)")
                    .font(Theme.Typography.headlineFont())
                    .foregroundColor(Theme.Colors.textPrimary)

                HStack(spacing: 20) {
                    StatusIndicator(status: .connected, showLabel: false)
                    StatusIndicator(status: .reconnecting, showLabel: false)
                    StatusIndicator(status: .disconnected, showLabel: false)
                }
            }
            .padding(Theme.Spacing.lg)
            .background(Theme.Colors.card)
            .cornerRadius(Theme.CornerRadius.md)

            // In context (header bar simulation)
            HStack {
                Text("HEAN")
                    .font(Theme.Typography.headlineFont(18, weight: .bold))
                    .foregroundColor(Theme.Colors.textPrimary)

                Spacer()

                StatusIndicator(status: .connected, latency: 32, showLabel: true)
            }
            .padding(Theme.Spacing.lg)
            .background(Theme.Colors.card)
            .cornerRadius(Theme.CornerRadius.md)
        }
        .padding(Theme.Spacing.xl)
    }
}
