//
//  GenesisComponents.swift
//  HEAN
//
//  Shared components for Genesis cinematic scenes
//

import SwiftUI

// MARK: - Navigation Bar

struct GenesisNavBar: View {
    @Binding var currentScene: Int
    let totalScenes: Int
    @Binding var isAutoPlaying: Bool
    var namespace: Namespace.ID
    let onPrevious: () -> Void
    let onNext: () -> Void
    let onToggleAutoPlay: () -> Void

    var body: some View {
        HStack(spacing: Theme.Spacing.xl) {
            // Previous
            Button(action: onPrevious) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundColor(currentScene > 0 ? Theme.Colors.textPrimary : Theme.Colors.textTertiary)
            }
            .disabled(currentScene == 0)

            Spacer()

            // Scene dots with matchedGeometryEffect active indicator
            HStack(spacing: Theme.Spacing.sm) {
                ForEach(0..<totalScenes, id: \.self) { index in
                    ZStack {
                        // Inactive dot
                        Circle()
                            .fill(Theme.Colors.textTertiary.opacity(0.4))
                            .frame(width: 6, height: 6)

                        // Active dot (slides between positions)
                        if index == currentScene {
                            Circle()
                                .fill(Theme.Colors.accent)
                                .frame(width: 10, height: 10)
                                .matchedGeometryEffect(id: "activeDot", in: namespace)
                                .shadow(color: Theme.Colors.accent.opacity(0.5), radius: 4)
                        }
                    }
                    .frame(width: 12, height: 12)
                    .onTapGesture {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.7)) {
                            currentScene = index
                        }
                    }
                }
            }

            Spacer()

            // Auto-play / Next
            HStack(spacing: Theme.Spacing.lg) {
                Button(action: onToggleAutoPlay) {
                    Image(systemName: isAutoPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(Theme.Colors.accent)
                }

                Button(action: onNext) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundColor(currentScene < totalScenes - 1 ? Theme.Colors.textPrimary : Theme.Colors.textTertiary)
                }
                .disabled(currentScene >= totalScenes - 1)
            }
        }
        .padding(.vertical, Theme.Spacing.md)
        .padding(.horizontal, Theme.Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Theme.CornerRadius.xl)
                .fill(.ultraThinMaterial)
                .opacity(0.6)
        )
    }
}

// MARK: - Brain Text Overlay

struct BrainTextOverlay: View {
    let text: String
    let isVisible: Bool

    var body: some View {
        VStack(spacing: Theme.Spacing.md) {
            // "HEAN Brain" label
            HStack(spacing: Theme.Spacing.sm) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.Colors.accent)

                Text("HEAN")
                    .font(Theme.Typography.caption(11, weight: .bold))
                    .foregroundColor(Theme.Colors.accent)
                    .tracking(2)
            }
            .opacity(isVisible ? 1 : 0)

            // Brain thought text
            Text(text)
                .font(Theme.Typography.bodyFont(15, weight: .regular))
                .foregroundColor(Theme.Colors.textPrimary.opacity(0.9))
                .multilineTextAlignment(.center)
                .lineSpacing(6)
                .opacity(isVisible ? 1 : 0)
                .offset(y: isVisible ? 0 : 15)
        }
        .padding(.horizontal, Theme.Spacing.xxl)
        .animation(.easeOut(duration: 1.0), value: isVisible)
    }
}

// MARK: - Neural Node

struct NeuralNode: View {
    let label: String
    let icon: String
    let color: Color
    let isActive: Bool
    let size: CGFloat

    init(
        label: String,
        icon: String = "circle.fill",
        color: Color = Theme.Colors.accent,
        isActive: Bool = false,
        size: CGFloat = 56
    ) {
        self.label = label
        self.icon = icon
        self.color = color
        self.isActive = isActive
        self.size = size
    }

    var body: some View {
        VStack(spacing: Theme.Spacing.xs) {
            ZStack {
                // Outer glow
                Circle()
                    .fill(color.opacity(isActive ? 0.3 : 0.1))
                    .frame(width: size + 16, height: size + 16)
                    .blur(radius: isActive ? 8 : 4)

                // Node circle
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                color.opacity(isActive ? 0.8 : 0.4),
                                color.opacity(isActive ? 0.3 : 0.1)
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: size / 2
                        )
                    )
                    .frame(width: size, height: size)

                // Border
                Circle()
                    .strokeBorder(color.opacity(isActive ? 0.8 : 0.3), lineWidth: 1.5)
                    .frame(width: size, height: size)

                // Icon
                Image(systemName: icon)
                    .font(.system(size: size * 0.35, weight: .medium))
                    .foregroundColor(isActive ? .white : color.opacity(0.7))
            }
            .scaleEffect(isActive ? 1.05 : 1.0)
            .animation(.easeInOut(duration: 0.4), value: isActive)

            Text(label)
                .font(Theme.Typography.caption(10, weight: .semibold))
                .foregroundColor(isActive ? color : Theme.Colors.textTertiary)
                .tracking(1)
        }
    }
}

// MARK: - Animated Connection Line

struct ConnectionLine: Shape {
    var progress: CGFloat

    var animatableData: CGFloat {
        get { progress }
        set { progress = newValue }
    }

    let from: CGPoint
    let to: CGPoint

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        return path.trimmedPath(from: 0, to: progress)
    }
}

// MARK: - Pulse Particle

struct PulseParticle: View {
    let color: Color
    let size: CGFloat

    @State private var isPulsing = false

    var body: some View {
        ZStack {
            Circle()
                .fill(color)
                .frame(width: size, height: size)

            Circle()
                .fill(color.opacity(0.4))
                .frame(width: size * 2, height: size * 2)
                .blur(radius: 4)
                .scaleEffect(isPulsing ? 1.3 : 0.8)
                .opacity(isPulsing ? 0.2 : 0.6)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                isPulsing = true
            }
        }
    }
}

// MARK: - Shield Flash Effect

struct ShieldFlash: View {
    let isActive: Bool
    let color: Color

    var body: some View {
        ZStack {
            // Hexagonal shield shape approximation
            Circle()
                .strokeBorder(color.opacity(isActive ? 0.8 : 0), lineWidth: 3)
                .frame(width: 80, height: 80)
                .scaleEffect(isActive ? 1.3 : 1.0)
                .blur(radius: isActive ? 2 : 0)

            Circle()
                .strokeBorder(color.opacity(isActive ? 0.4 : 0), lineWidth: 1.5)
                .frame(width: 90, height: 90)
                .scaleEffect(isActive ? 1.5 : 1.0)
                .blur(radius: isActive ? 6 : 0)
        }
        .animation(.easeOut(duration: 0.5), value: isActive)
    }
}

// MARK: - Animated Text (Character-by-Character Stagger)

struct AnimatedText: View {
    let text: String
    let isVisible: Bool
    let font: Font
    let color: Color
    let stagger: Double

    init(
        _ text: String,
        isVisible: Bool,
        font: Font = Theme.Typography.caption(12, weight: .medium),
        color: Color = Theme.Colors.textTertiary,
        stagger: Double = 0.03
    ) {
        self.text = text
        self.isVisible = isVisible
        self.font = font
        self.color = color
        self.stagger = stagger
    }

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(text.enumerated()), id: \.offset) { i, char in
                Text(String(char))
                    .font(font)
                    .foregroundColor(color)
                    .opacity(isVisible ? 1 : 0)
                    .offset(y: isVisible ? 0 : 8)
                    .animation(
                        .easeOut(duration: 0.35).delay(Double(i) * stagger),
                        value: isVisible
                    )
            }
        }
    }
}

// MARK: - Genesis Hero Layer (Morphing Central Orb)

struct GenesisHeroLayer: View {
    let currentScene: Int

    var body: some View {
        GeometryReader { geo in
            let center = CGPoint(x: geo.size.width / 2, y: geo.size.height * 0.35)

            ZStack {
                // Outer glow
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                orbColor.opacity(outerGlowOpacity),
                                orbColor.opacity(0)
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: outerGlowRadius
                        )
                    )
                    .frame(width: outerGlowRadius * 2, height: outerGlowRadius * 2)
                    .position(center)

                // Inner orb
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                orbColor.opacity(innerOpacity),
                                orbColor.opacity(innerOpacity * 0.3),
                                orbColor.opacity(0)
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: orbSize / 2
                        )
                    )
                    .frame(width: orbSize, height: orbSize)
                    .position(center)

                // Border ring (visible in scene 4)
                if currentScene == 4 {
                    Circle()
                        .strokeBorder(
                            LinearGradient(
                                colors: [
                                    Theme.Colors.accent.opacity(0.6),
                                    Color(hex: "7B61FF").opacity(0.4),
                                    Theme.Colors.accent.opacity(0.2)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 2
                        )
                        .frame(width: 90, height: 90)
                        .position(center)
                        .transition(.opacity)
                }

                // Brain icon (scene 4 only)
                if currentScene == 4 {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 36, weight: .light))
                        .foregroundColor(.white.opacity(0.9))
                        .position(center)
                        .transition(.opacity.combined(with: .scale(scale: 0.6)))
                }
            }
            .opacity(heroOpacity)
            .animation(.easeInOut(duration: 0.8), value: currentScene)
        }
        .allowsHitTesting(false)
    }

    // Scene-dependent properties
    private var heroOpacity: Double {
        switch currentScene {
        case 0: return 0
        case 1: return 0.6
        case 2: return 0.8
        case 3: return 0.3
        case 4: return 1.0
        default: return 0
        }
    }

    private var orbSize: CGFloat {
        switch currentScene {
        case 1: return 20
        case 2: return 50
        case 3: return 24
        case 4: return 90
        default: return 0
        }
    }

    private var orbColor: Color {
        switch currentScene {
        case 1: return .white
        case 2: return Theme.Colors.accent
        case 3: return Theme.Colors.accent
        case 4: return Theme.Colors.accent
        default: return .clear
        }
    }

    private var innerOpacity: Double {
        switch currentScene {
        case 1: return 0.9
        case 2: return 0.15
        case 3: return 0.3
        case 4: return 0.5
        default: return 0
        }
    }

    private var outerGlowOpacity: Double {
        switch currentScene {
        case 1: return 0.3
        case 2: return 0.12
        case 3: return 0.1
        case 4: return 0.3
        default: return 0
        }
    }

    private var outerGlowRadius: CGFloat {
        switch currentScene {
        case 1: return 30
        case 2: return 60
        case 3: return 20
        case 4: return 80
        default: return 0
        }
    }
}

// MARK: - Growing Chart Line

struct GrowthChart: Shape {
    var progress: CGFloat

    var animatableData: CGFloat {
        get { progress }
        set { progress = newValue }
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let w = rect.width
        let h = rect.height

        // Smooth upward curve with slight dips (realistic equity curve)
        let points: [(CGFloat, CGFloat)] = [
            (0.0, 0.85), (0.08, 0.78), (0.15, 0.72),
            (0.22, 0.75), (0.28, 0.65), (0.35, 0.60),
            (0.42, 0.62), (0.48, 0.52), (0.55, 0.48),
            (0.60, 0.50), (0.65, 0.42), (0.72, 0.38),
            (0.78, 0.35), (0.82, 0.30), (0.88, 0.25),
            (0.92, 0.28), (0.95, 0.22), (1.0, 0.15),
        ]

        let trimmedCount = max(1, Int(CGFloat(points.count) * progress))
        let visiblePoints = Array(points.prefix(trimmedCount))

        guard let first = visiblePoints.first else { return path }
        path.move(to: CGPoint(x: first.0 * w, y: first.1 * h))

        for i in 1..<visiblePoints.count {
            let p = visiblePoints[i]
            let prev = visiblePoints[i - 1]
            let cp1 = CGPoint(
                x: (prev.0 + (p.0 - prev.0) * 0.5) * w,
                y: prev.1 * h
            )
            let cp2 = CGPoint(
                x: (prev.0 + (p.0 - prev.0) * 0.5) * w,
                y: p.1 * h
            )
            path.addCurve(
                to: CGPoint(x: p.0 * w, y: p.1 * h),
                control1: cp1,
                control2: cp2
            )
        }

        return path
    }
}
