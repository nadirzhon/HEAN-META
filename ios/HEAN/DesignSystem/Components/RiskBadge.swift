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
            case .normal: return AppColors.success
            case .softBrake: return AppColors.warning
            case .quarantine: return Color.orange
            case .hardStop: return AppColors.error
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
            .padding(AppTypography.xs)
            .background(
                Circle()
                    .fill(state.color.opacity(0.2))
                    .scaleEffect(isPulsing ? 1.2 : 1.0)
                    .opacity(isPulsing ? 0 : 1)
            )
            .accessibilityLabel("Risk state: \(state.rawValue)")
    }

    private var expandedView: some View {
        VStack(alignment: .leading, spacing: AppTypography.sm) {
            // Header with icon and label
            HStack(spacing: AppTypography.sm) {
                Image(systemName: state.icon)
                    .font(.system(size: 20, weight: .bold))

                Text(state.rawValue.replacingOccurrences(of: "_", with: " "))
                    .font(AppTypography.headline(16, weight: .bold))

                Spacer()
            }
            .foregroundColor(state.color)

            // Description
            Text(state.description)
                .font(AppTypography.caption(13))
                .foregroundColor(AppColors.textSecondary)
        }
        .padding(AppTypography.md)
        .background(
            RoundedRectangle(cornerRadius: AppTypography.radiusMd)
                .fill(state.color.opacity(0.15))
                .overlay(
                    RoundedRectangle(cornerRadius: AppTypography.radiusMd)
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
        AppColors.backgroundPrimary
            .ignoresSafeArea()

        ScrollView {
            VStack(spacing: AppTypography.lg) {
                // Compact variants
                VStack(alignment: .leading, spacing: AppTypography.sm) {
                    Text("Compact Badges")
                        .font(AppTypography.headline())
                        .foregroundColor(AppColors.textPrimary)

                    HStack(spacing: AppTypography.md) {
                        RiskBadge(state: .normal, variant: .compact)
                        RiskBadge(state: .softBrake, variant: .compact)
                        RiskBadge(state: .quarantine, variant: .compact)
                        RiskBadge(state: .hardStop, variant: .compact)
                    }
                }

                Divider()
                    .background(AppColors.textTertiary)

                // Expanded variants
                VStack(alignment: .leading, spacing: AppTypography.md) {
                    Text("Expanded Badges")
                        .font(AppTypography.headline())
                        .foregroundColor(AppColors.textPrimary)

                    RiskBadge(state: .normal, variant: .expanded)
                    RiskBadge(state: .softBrake, variant: .expanded)
                    RiskBadge(state: .quarantine, variant: .expanded)
                    RiskBadge(state: .hardStop, variant: .expanded)
                }
            }
            .padding(AppTypography.xl)
        }
    }
}
