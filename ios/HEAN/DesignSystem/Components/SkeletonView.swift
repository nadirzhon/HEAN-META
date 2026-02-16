//
//  SkeletonView.swift
//  HEAN
//
//  Loading placeholder with shimmer animation
//

import SwiftUI

/// Loading placeholder with shimmer animation
struct SkeletonView<Content: View>: View {
    let content: Content
    let isLoading: Bool

    @State private var phase: CGFloat = 0

    init(
        isLoading: Bool,
        @ViewBuilder content: () -> Content
    ) {
        self.isLoading = isLoading
        self.content = content()
    }

    var body: some View {
        content
            .opacity(isLoading ? 0.3 : 1.0)
            .overlay(
                Group {
                    if isLoading {
                        GeometryReader { geometry in
                            shimmerOverlay(size: geometry.size)
                        }
                    }
                }
            )
            .onAppear {
                if isLoading {
                    startShimmerAnimation()
                }
            }
    }

    private func shimmerOverlay(size: CGSize) -> some View {
        LinearGradient(
            colors: [
                Color.clear,
                Color.white.opacity(0.3),
                Color.clear
            ],
            startPoint: .leading,
            endPoint: .trailing
        )
        .frame(width: size.width * 2)
        .offset(x: -size.width + (phase * size.width * 2))
    }

    private func startShimmerAnimation() {
        withAnimation(Animation.linear(duration: 1.5).repeatForever(autoreverses: false)) {
            phase = 1.0
        }
    }
}

// MARK: - Convenience Modifiers
extension View {
    func skeleton(isLoading: Bool) -> some View {
        SkeletonView(isLoading: isLoading) {
            self
        }
    }
}

// MARK: - Preview
#Preview("SkeletonView") {
    ZStack {
        Theme.Colors.background
            .ignoresSafeArea()

        VStack(spacing: 20) {
            // Card skeleton
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Theme.Colors.card)
                    .frame(width: 120, height: 20)

                RoundedRectangle(cornerRadius: 8)
                    .fill(Theme.Colors.card)
                    .frame(width: 200, height: 40)

                RoundedRectangle(cornerRadius: 8)
                    .fill(Theme.Colors.card)
                    .frame(width: 150, height: 16)
            }
            .padding(Theme.Spacing.lg)
            .background(Theme.Colors.card)
            .cornerRadius(Theme.CornerRadius.md)
            .skeleton(isLoading: true)

            // Text skeleton
            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Theme.Colors.textTertiary.opacity(0.3))
                    .frame(height: 16)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Theme.Colors.textTertiary.opacity(0.3))
                    .frame(width: 250, height: 16)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Theme.Colors.textTertiary.opacity(0.3))
                    .frame(width: 180, height: 16)
            }
            .skeleton(isLoading: true)

            // Circle avatar skeleton
            Circle()
                .fill(Theme.Colors.card)
                .frame(width: 60, height: 60)
                .skeleton(isLoading: true)
        }
        .padding(Theme.Spacing.xl)
    }
}
