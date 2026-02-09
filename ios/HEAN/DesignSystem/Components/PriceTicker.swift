//
//  PriceTicker.swift
//  HEAN
//
//  Real-time price ticker with animations
//

import SwiftUI

/// Real-time price display with pulse and flash animations
struct PriceTicker: View {
    let symbol: String
    let price: Double
    let changePercent: Double
    let size: TickerSize

    init(
        symbol: String = "",
        price: Double,
        changePercent: Double = 0,
        size: TickerSize = .medium
    ) {
        self.symbol = symbol
        self.price = price
        self.changePercent = changePercent
        self.size = size
    }

    @State private var isPulsing = false
    @State private var flashColor: Color? = nil

    enum TickerSize {
        case small, medium, large

        var priceFont: Font {
            switch self {
            case .small: return AppTypography.mono(18, weight: .semibold)
            case .medium: return AppTypography.mono(24, weight: .bold)
            case .large: return AppTypography.mono(32, weight: .bold)
            }
        }

        var symbolFont: Font {
            switch self {
            case .small: return AppTypography.caption(12)
            case .medium: return AppTypography.body(14)
            case .large: return AppTypography.headline(16)
            }
        }

        var changeFont: Font {
            switch self {
            case .small: return AppTypography.caption(11, weight: .medium)
            case .medium: return AppTypography.caption(13, weight: .medium)
            case .large: return AppTypography.body(15, weight: .medium)
            }
        }
    }

    private var isPositive: Bool {
        changePercent >= 0
    }

    private var changeColor: Color {
        isPositive ? AppColors.success : AppColors.error
    }

    private var formattedPrice: String {
        String(format: "$%.2f", price)
    }

    private var formattedChange: String {
        String(format: "%+.2f%%", changePercent)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: AppTypography.xs / 2) {
            // Symbol label (only if not empty)
            if !symbol.isEmpty {
                Text(symbol)
                    .font(size.symbolFont)
                    .foregroundColor(AppColors.textSecondary)
            }

            // Price
            Text(formattedPrice)
                .font(size.priceFont)
                .foregroundColor(AppColors.textPrimary)
                .scaleEffect(isPulsing ? 1.05 : 1.0)
                .animation(AppAnimation.spring, value: isPulsing)

            // Change percentage with arrow
            HStack(spacing: 4) {
                Image(systemName: isPositive ? "arrow.up.right" : "arrow.down.right")
                    .font(.system(size: size == .small ? 10 : 12, weight: .bold))

                Text(formattedChange)
                    .font(size.changeFont)
            }
            .foregroundColor(changeColor)
        }
        .padding(size == .small ? AppTypography.sm : AppTypography.md)
        .background(
            RoundedRectangle(cornerRadius: AppTypography.radiusSm)
                .fill(flashColor ?? Color.clear)
                .animation(.easeOut(duration: 0.3), value: flashColor)
        )
        .onChange(of: price) { oldValue, newValue in
            triggerPriceChange(oldPrice: oldValue, newPrice: newValue)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(symbol) price \(formattedPrice), change \(formattedChange)")
    }

    private func triggerPriceChange(oldPrice: Double, newPrice: Double) {
        // Pulse animation
        isPulsing = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            isPulsing = false
        }

        // Flash background
        let direction = newPrice > oldPrice
        flashColor = direction ? AppColors.success.opacity(0.2) : AppColors.error.opacity(0.2)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            flashColor = nil
        }
    }
}

// MARK: - Preview
#Preview("PriceTicker Sizes") {
    ZStack {
        AppColors.backgroundPrimary
            .ignoresSafeArea()

        VStack(spacing: AppTypography.lg) {
            // Large bullish
            PriceTicker(
                symbol: "BTCUSDT",
                price: 42_350.75,
                changePercent: 3.45,
                size: .large
            )

            // Medium bearish
            PriceTicker(
                symbol: "ETHUSDT",
                price: 2_245.30,
                changePercent: -1.25,
                size: .medium
            )

            // Small
            HStack(spacing: AppTypography.md) {
                PriceTicker(
                    symbol: "BTC",
                    price: 42_350.75,
                    changePercent: 2.1,
                    size: .small
                )

                PriceTicker(
                    symbol: "ETH",
                    price: 2_245.30,
                    changePercent: -0.8,
                    size: .small
                )
            }
        }
        .padding(AppTypography.xl)
    }
}
