//
//  RiskBadge.swift
//  HEAN
//
//  Risk state indicator badge
//

import SwiftUI

/// Risk state indicator with animated pulse for alerts
struct RiskBadge: View {
    let state: RiskState
    let variant: BadgeVariant

    @State private var isPulsing = false

    enum RiskState: String {
        case normal = "NORMAL"
        case softBrake = "SOFT_BRAKE"
        case quarantine = "QUARANTINE"
        case hardStop = "HARD_STOP"

        var color: Color {
            switch self {
            case .normal: return Theme.Colors.success
            case .softBrake: return Theme.Colors.warning
            case .quarantine: return Color.orange
            case .hardStop: return Theme.Colors.error
            }
        }

        var icon: String {
            switch self {
            case .normal: return "checkmark.shield.fill"
            case .softBrake: return "exclamationmark.triangle.fill"
            case .quarantine: return "pause.circle.fill"
            case .hardStop: return "xmark.octagon.fill"
            }
        }

        var description: String {
            switch self {
            case .normal: return "All systems operational"
            case .softBrake: return "Reduced position sizing"
            case .quarantine: return "Trading paused for review"
            case .hardStop: return "Trading halted"
            }
        }

        var shouldPulse: Bool {
            self != .normal
        }
    }

    enum BadgeVariant {
        case compact  // Icon only
        case expanded // Icon + label + description
    }

    init(state: RiskState, variant: BadgeVariant = .compact) {
        self.state = state
        self.variant = variant
    }

    var body: some View {
        Group {
            if variant == .compact {
                compactView
            } else {
                expandedView
            }
        }
        .onAppear {
            if state.shouldPulse {
                startPulseAnimation()
            }
        }
        .onChange(of: state) { _, _ in
            if state.shouldPulse {
                startPulseAnimation()
            } else {
                isPulsing = false
            }
        }
    }

    private var compactView: some View {
        Image(systemName: state.icon)
            .font(.system(size: 16, weight: .bold))
            .foregroundColor(state.color)
            .padding(Theme.Spacing.sm)
            .background(
                Circle()
                    .fill(state.color.opacity(0.2))
                    .scaleEffect(isPulsing ? 1.2 : 1.0)
                    .opacity(isPulsing ? 0 : 1)
            )
            .accessibilityLabel("Risk state: \(state.rawValue)")
    }

    private var expandedView: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            // Header with icon and label
            HStack(spacing: Theme.Spacing.md) {
                Image(systemName: state.icon)
                    .font(.system(size: 20, weight: .bold))

                Text(state.rawValue.replacingOccurrences(of: "_", with: " "))
                    .font(Theme.Typography.headlineFont(16, weight: .bold))

                Spacer()
            }
            .foregroundColor(state.color)

            // Description
            Text(state.description)
                .font(Theme.Typography.caption(13))
                .foregroundColor(Theme.Colors.textSecondary)
        }
        .padding(Theme.Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Theme.CornerRadius.md)
                .fill(state.color.opacity(0.15))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.CornerRadius.md)
                        .strokeBorder(state.color.opacity(isPulsing ? 0.6 : 0.3), lineWidth: 2)
                        .scaleEffect(isPulsing ? 1.05 : 1.0)
                )
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Risk state: \(state.rawValue). \(state.description)")
    }

    private func startPulseAnimation() {
        withAnimation(Animation.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
            isPulsing = true
        }
    }
}

// MARK: - Preview
#Preview("RiskBadge Variants") {
    ZStack {
        Theme.Colors.background
            .ignoresSafeArea()

        ScrollView {
            VStack(spacing: 20) {
                // Compact variants
                VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                    Text("Compact Badges")
                        .font(Theme.Typography.headlineFont())
                        .foregroundColor(Theme.Colors.textPrimary)

                    HStack(spacing: Theme.Spacing.lg) {
                        RiskBadge(state: .normal, variant: .compact)
                        RiskBadge(state: .softBrake, variant: .compact)
                        RiskBadge(state: .quarantine, variant: .compact)
                        RiskBadge(state: .hardStop, variant: .compact)
                    }
                }

                Divider()
                    .background(Theme.Colors.textTertiary)

                // Expanded variants
                VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                    Text("Expanded Badges")
                        .font(Theme.Typography.headlineFont())
                        .foregroundColor(Theme.Colors.textPrimary)

                    RiskBadge(state: .normal, variant: .expanded)
                    RiskBadge(state: .softBrake, variant: .expanded)
                    RiskBadge(state: .quarantine, variant: .expanded)
                    RiskBadge(state: .hardStop, variant: .expanded)
                }
            }
            .padding(Theme.Spacing.xl)
        }
    }
}
