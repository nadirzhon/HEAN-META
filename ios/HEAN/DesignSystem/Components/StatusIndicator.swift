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
            case .connected: return AppColors.success
            case .disconnected: return AppColors.error
            case .reconnecting: return AppColors.warning
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
        HStack(spacing: AppTypography.xs) {
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
                    .font(AppTypography.caption(13, weight: .medium))
                    .foregroundColor(AppColors.textPrimary)

                // Latency (if available and connected)
                if let latency = latency, status == .connected {
                    Text("(\(latency)ms)")
                        .font(AppTypography.caption(11, weight: .regular))
                        .foregroundColor(AppColors.textTertiary)
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
        AppColors.backgroundPrimary
            .ignoresSafeArea()

        VStack(spacing: AppTypography.lg) {
            // With labels
            VStack(alignment: .leading, spacing: AppTypography.md) {
                Text("With Labels")
                    .font(AppTypography.headline())
                    .foregroundColor(AppColors.textPrimary)

                StatusIndicator(status: .connected, latency: 45, showLabel: true)
                StatusIndicator(status: .reconnecting, showLabel: true)
                StatusIndicator(status: .disconnected, showLabel: true)
            }
            .padding(AppTypography.md)
            .background(AppColors.backgroundSecondary)
            .cornerRadius(AppTypography.radiusMd)

            // Without labels (dots only)
            VStack(alignment: .leading, spacing: AppTypography.md) {
                Text("Compact (No Labels)")
                    .font(AppTypography.headline())
                    .foregroundColor(AppColors.textPrimary)

                HStack(spacing: AppTypography.lg) {
                    StatusIndicator(status: .connected, showLabel: false)
                    StatusIndicator(status: .reconnecting, showLabel: false)
                    StatusIndicator(status: .disconnected, showLabel: false)
                }
            }
            .padding(AppTypography.md)
            .background(AppColors.backgroundSecondary)
            .cornerRadius(AppTypography.radiusMd)

            // In context (header bar simulation)
            HStack {
                Text("HEAN")
                    .font(AppTypography.headline(18, weight: .bold))
                    .foregroundColor(AppColors.textPrimary)

                Spacer()

                StatusIndicator(status: .connected, latency: 32, showLabel: true)
            }
            .padding(AppTypography.md)
            .background(AppColors.backgroundSecondary)
            .cornerRadius(AppTypography.radiusMd)
        }
        .padding(AppTypography.xl)
    }
}
