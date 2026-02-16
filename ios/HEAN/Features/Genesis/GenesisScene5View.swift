//
//  GenesisScene5View.swift
//  HEAN
//
//  Scene 5: "Conscious Growth"
//  All nodes converge into a unified brain. An equity curve rises steadily.
//  Brain sphere is handled by GenesisHeroLayer; this view provides convergence + chart.
//

import SwiftUI

struct GenesisScene5View: View {
    let sceneActive: Bool

    @State private var nodesConverged = false
    @State private var chartProgress: CGFloat = 0
    @State private var showText = false
    @State private var heanTextOpacity: CGFloat = 0
    @State private var heanTextScale: CGFloat = 0.8

    private let brainText: String = L.isRussian
        ? "Моя цель — не погоня за сиюминутной прибылью. Моя цель — осознанный, управляемый рост, основанный на глубоком понимании рынка. Добро пожаловать в HEAN."
        : "My goal is not the pursuit of short-term profit. My goal is conscious, managed growth built on a deep understanding of the market. Welcome to HEAN."

    // Scattered starting positions for converging nodes (relative 0..1)
    private let scatterPositions: [(CGFloat, CGFloat)] = [
        (0.15, 0.2), (0.85, 0.15), (0.1, 0.55),
        (0.9, 0.5), (0.25, 0.7), (0.75, 0.65),
    ]

    private let nodeColors: [Color] = [
        Color(hex: "7B61FF"), Theme.Colors.accent, Color(hex: "F97316"),
        Theme.Colors.warning, Theme.Colors.success, Color(hex: "3B82F6"),
    ]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let brainCenter = CGPoint(x: w * 0.5, y: h * 0.3)

            ZStack {
                // Converging node particles
                ForEach(0..<6, id: \.self) { i in
                    let startPos = CGPoint(
                        x: scatterPositions[i].0 * w,
                        y: scatterPositions[i].1 * h
                    )
                    let pos = nodesConverged ? brainCenter : startPos

                    Circle()
                        .fill(nodeColors[i].opacity(nodesConverged ? 0.6 : 0.8))
                        .frame(width: nodesConverged ? 8 : 16, height: nodesConverged ? 8 : 16)
                        .blur(radius: nodesConverged ? 4 : 1)
                        .position(pos)
                        .animation(
                            .easeInOut(duration: 1.8).delay(Double(i) * 0.1),
                            value: nodesConverged
                        )
                }

                // Connection lines between scattered nodes (fade out on converge)
                if !nodesConverged {
                    ForEach(0..<5, id: \.self) { i in
                        let from = CGPoint(
                            x: scatterPositions[i].0 * w,
                            y: scatterPositions[i].1 * h
                        )
                        let to = CGPoint(
                            x: scatterPositions[i + 1].0 * w,
                            y: scatterPositions[i + 1].1 * h
                        )
                        Path { path in
                            path.move(to: from)
                            path.addLine(to: to)
                        }
                        .stroke(Theme.Colors.accent.opacity(0.15), lineWidth: 1)
                    }
                }

                // Note: Brain sphere is handled by GenesisHeroLayer

                // HEAN title reveal
                Text("H E A N")
                    .font(.system(size: 28, weight: .ultraLight, design: .monospaced))
                    .foregroundColor(Theme.Colors.accent)
                    .tracking(12)
                    .opacity(heanTextOpacity)
                    .scaleEffect(heanTextScale)
                    .position(x: w / 2, y: h * 0.3 + 70)

                // Equity growth chart
                VStack {
                    Spacer()

                    ZStack(alignment: .bottomLeading) {
                        // Grid lines
                        VStack(spacing: 0) {
                            ForEach(0..<4, id: \.self) { _ in
                                Spacer()
                                Rectangle()
                                    .fill(Theme.Colors.border)
                                    .frame(height: 0.5)
                            }
                        }
                        .frame(height: 120)

                        // The growth line
                        GrowthChart(progress: chartProgress)
                            .stroke(
                                LinearGradient(
                                    colors: [
                                        Theme.Colors.accent.opacity(0.6),
                                        Theme.Colors.success
                                    ],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                                style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round)
                            )
                            .frame(height: 120)

                        // Glow under the chart
                        GrowthChart(progress: chartProgress)
                            .stroke(
                                Theme.Colors.success.opacity(0.2),
                                style: StrokeStyle(lineWidth: 8, lineCap: .round)
                            )
                            .blur(radius: 6)
                            .frame(height: 120)
                    }
                    .padding(.horizontal, Theme.Spacing.xl)
                    .padding(.bottom, 10)
                    .opacity(chartProgress > 0 ? 1 : 0)

                    // Brain text
                    BrainTextOverlay(text: brainText, isVisible: showText)
                        .padding(.bottom, 100)
                }
                .frame(width: w, height: h)
            }
        }
        .onChange(of: sceneActive) { _, isActive in
            if isActive {
                animateSequence()
            } else {
                resetAnimations()
            }
        }
        .onAppear {
            if sceneActive { animateSequence() }
        }
    }

    private func animateSequence() {
        // Step 1: Converge nodes
        withAnimation(.easeInOut(duration: 2.0).delay(0.5)) {
            nodesConverged = true
        }

        // Step 2: HEAN text
        withAnimation(.easeOut(duration: 0.8).delay(2.5)) {
            heanTextOpacity = 1.0
            heanTextScale = 1.0
        }

        // Step 3: Chart grows
        withAnimation(.easeInOut(duration: 2.5).delay(3.0)) {
            chartProgress = 1.0
        }

        // Step 4: Brain text
        withAnimation(.easeOut(duration: 1.0).delay(4.5)) {
            showText = true
        }
    }

    private func resetAnimations() {
        nodesConverged = false
        chartProgress = 0
        showText = false
        heanTextOpacity = 0
        heanTextScale = 0.8
    }
}

#Preview {
    GenesisScene5View(sceneActive: true)
        .preferredColorScheme(.dark)
}
