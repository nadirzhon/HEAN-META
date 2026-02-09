//
//  View+Modifiers.swift
//  HEAN
//

import SwiftUI

// MARK: - Glass Background Modifier

struct GlassBackgroundModifier: ViewModifier {
    var cornerRadius: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(.ultraThinMaterial)
                    .opacity(0.8)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(
                        LinearGradient(
                            colors: [.white.opacity(0.2), .white.opacity(0.05)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.5
                    )
            )
    }
}

extension View {
    func glassBackground(cornerRadius: CGFloat = 16) -> some View {
        modifier(GlassBackgroundModifier(cornerRadius: cornerRadius))
    }
}

// MARK: - Shimmer Effect

struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(
                    colors: [
                        .clear,
                        .white.opacity(0.1),
                        .clear
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .offset(x: phase)
                .mask(content)
            )
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    phase = 300
                }
            }
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

// MARK: - Pulse Animation

struct PulseModifier: ViewModifier {
    @State private var isPulsing = false
    let color: Color

    func body(content: Content) -> some View {
        content
            .overlay(
                Circle()
                    .fill(color.opacity(isPulsing ? 0 : 0.3))
                    .scaleEffect(isPulsing ? 2 : 1)
                    .animation(.easeInOut(duration: 1).repeatForever(autoreverses: false), value: isPulsing)
            )
            .onAppear { isPulsing = true }
    }
}

extension View {
    func pulse(color: Color = .green) -> some View {
        modifier(PulseModifier(color: color))
    }
}

// MARK: - Conditional Modifier

extension View {
    @ViewBuilder
    func `if`<Transform: View>(_ condition: Bool, transform: (Self) -> Transform) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}
