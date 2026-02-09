//
//  GlassCard.swift
//  HEAN
//
//  Premium glassmorphism card component
//

import SwiftUI

/// Premium glassmorphism card with blur, gradient border, and shadow
struct GlassCard<Content: View>: View {
    let content: Content
    let cornerRadius: CGFloat
    let borderWidth: CGFloat
    let shadowRadius: CGFloat
    let padding: CGFloat?

    init(
        cornerRadius: CGFloat = AppTypography.radiusMd,
        borderWidth: CGFloat = 1,
        shadowRadius: CGFloat = 10,
        padding: CGFloat? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.content = content()
        self.cornerRadius = cornerRadius
        self.borderWidth = borderWidth
        self.shadowRadius = shadowRadius
        self.padding = padding
    }

    var body: some View {
        Group {
            if let padding = padding {
                content.padding(padding)
            } else {
                content
            }
        }
            .background(
                ZStack {
                    // Ultra-thin material blur
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(.ultraThinMaterial)
                        .opacity(0.8)

                    // Gradient overlay
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(
                            LinearGradient(
                                colors: [
                                    Color.white.opacity(0.05),
                                    Color.white.opacity(0.02)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }
            )
            .overlay(
                // Gradient border
                RoundedRectangle(cornerRadius: cornerRadius)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(0.3),
                                Color.white.opacity(0.1),
                                Color.white.opacity(0.05)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: borderWidth
                    )
            )
            .shadow(color: Color.black.opacity(0.3), radius: shadowRadius, x: 0, y: 4)
    }
}

/// Accent-bordered glass card variant
struct GlassCardAccent<Content: View>: View {
    let content: Content
    let cornerRadius: CGFloat

    init(
        cornerRadius: CGFloat = AppTypography.radiusMd,
        @ViewBuilder content: () -> Content
    ) {
        self.content = content()
        self.cornerRadius = cornerRadius
    }

    var body: some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(.ultraThinMaterial)
                    .opacity(0.9)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                AppColors.accentPrimary.opacity(0.5),
                                AppColors.accentPrimary.opacity(0.2),
                                Color.white.opacity(0.1)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1.5
                    )
            )
            .shadow(color: AppColors.accentPrimary.opacity(0.15), radius: 15, x: 0, y: 5)
    }
}

// MARK: - Preview
#Preview("GlassCard Variants") {
    ZStack {
        AppColors.backgroundPrimary
            .ignoresSafeArea()

        VStack(spacing: AppTypography.lg) {
            // Basic card
            GlassCard {
                VStack(alignment: .leading, spacing: AppTypography.sm) {
                    Text("Basic Card")
                        .font(AppTypography.headline())
                        .foregroundColor(AppColors.textPrimary)

                    Text("Premium glassmorphism effect")
                        .font(AppTypography.caption())
                        .foregroundColor(AppColors.textSecondary)
                }
                .padding(AppTypography.md)
            }

            // Large radius card
            GlassCard(cornerRadius: AppTypography.radiusLg) {
                HStack {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 32))
                        .foregroundColor(AppColors.accentPrimary)

                    VStack(alignment: .leading) {
                        Text("Trading Stats")
                            .font(AppTypography.headline())
                            .foregroundColor(AppColors.textPrimary)
                        Text("Real-time data")
                            .font(AppTypography.caption())
                            .foregroundColor(AppColors.textSecondary)
                    }
                    Spacer()
                }
                .padding(AppTypography.lg)
            }

            // Compact card
            GlassCard(cornerRadius: AppTypography.radiusSm, shadowRadius: 5) {
                Text("Compact Card")
                    .font(AppTypography.body())
                    .foregroundColor(AppColors.textPrimary)
                    .padding(AppTypography.sm)
            }
        }
        .padding(AppTypography.xl)
    }
}
