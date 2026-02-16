//
//  PnLBadge.swift
//  HEAN
//
//  Profit/Loss indicator badge
//

import SwiftUI

/// Profit/Loss indicator chip with auto-coloring
struct PnLBadge: View {
    let value: Double
    let format: BadgeFormat
    let size: BadgeSize

    enum BadgeFormat {
        case dollar       // $1,234.56
        case percent      // +12.34%
        case combined     // $1,234.56 (+12.34%)
    }

    enum BadgeSize {
        case compact, expanded
    }

    init(
        value: Double,
        format: BadgeFormat = .dollar,
        size: BadgeSize = .compact
    ) {
        self.value = value
        self.format = format
        self.size = size
    }

    /// Convenience initializer with percentage for combined display
    init(value: Double, percentage: Double, size: BadgeSize = .compact) {
        self.value = value
        self.format = .combined
        self.size = size
    }

    private var isPositive: Bool {
        value >= 0
    }

    private var color: Color {
        if value == 0 {
            return Theme.Colors.textSecondary
        }
        return isPositive ? Theme.Colors.success : Theme.Colors.error
    }

    private var backgroundColor: Color {
        if value == 0 {
            return Theme.Colors.textTertiary.opacity(0.1)
        }
        return color.opacity(0.15)
    }

    private var icon: String {
        if value == 0 {
            return "minus"
        }
        return isPositive ? "arrow.up.right" : "arrow.down.right"
    }

    private var formattedValue: String {
        switch format {
        case .dollar:
            return formatDollar(value)
        case .percent:
            return formatPercent(value)
        case .combined:
            // Assuming value is dollar amount and we calculate a mock percentage
            return "\(formatDollar(value)) (\(formatPercent(value / 100)))"
        }
    }

    private func formatDollar(_ val: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencySymbol = "$"
        formatter.maximumFractionDigits = 2
        return (val >= 0 ? "+" : "") + (formatter.string(from: NSNumber(value: val)) ?? "$0.00")
    }

    private func formatPercent(_ val: Double) -> String {
        String(format: "%+.2f%%", val)
    }

    var body: some View {
        HStack(spacing: Theme.Spacing.sm / 2) {
            Image(systemName: icon)
                .font(.system(size: size == .compact ? 10 : 12, weight: .bold))

            Text(formattedValue)
                .font(size == .compact ? Theme.Typography.caption(12, weight: .semibold) : Theme.Typography.bodyFont(14, weight: .semibold))
        }
        .foregroundColor(color)
        .padding(.horizontal, size == .compact ? Theme.Spacing.md : Theme.Spacing.lg)
        .padding(.vertical, size == .compact ? 6 : Theme.Spacing.sm)
        .background(
            Capsule()
                .fill(backgroundColor)
        )
        .accessibilityLabel("PnL: \(formattedValue)")
    }
}

// MARK: - Preview
#Preview("PnLBadge Variants") {
    ZStack {
        Theme.Colors.background
            .ignoresSafeArea()

        VStack(spacing: 20) {
            // Positive dollar
            HStack(spacing: Theme.Spacing.lg) {
                PnLBadge(value: 1234.56, format: .dollar, size: .compact)
                PnLBadge(value: 1234.56, format: .dollar, size: .expanded)
            }

            // Negative dollar
            HStack(spacing: Theme.Spacing.lg) {
                PnLBadge(value: -567.89, format: .dollar, size: .compact)
                PnLBadge(value: -567.89, format: .dollar, size: .expanded)
            }

            // Percent
            HStack(spacing: Theme.Spacing.lg) {
                PnLBadge(value: 12.34, format: .percent, size: .compact)
                PnLBadge(value: -8.76, format: .percent, size: .expanded)
            }

            // Combined
            VStack(spacing: Theme.Spacing.md) {
                PnLBadge(value: 2500.00, format: .combined, size: .compact)
                PnLBadge(value: -1500.00, format: .combined, size: .expanded)
            }

            // Zero
            PnLBadge(value: 0, format: .dollar, size: .compact)
        }
        .padding(Theme.Spacing.xl)
    }
}
