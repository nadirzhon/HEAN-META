//
//  GenesisScene4View.swift
//  HEAN
//
//  Scene 4: "Signal Path"
//  A pulse travels through the decision pipeline:
//  Analysis → Strategy → Risk (shield flash) → Execution
//

import SwiftUI

struct GenesisScene4View: View {
    let sceneActive: Bool

    @State private var nodesVisible: [Bool] = Array(repeating: false, count: 4)
    @State private var linesDrawn: [CGFloat] = Array(repeating: 0, count: 3)
    @State private var pulseStage = -1 // -1 = not started, 0-3 = at node
    @State private var pulseX: CGFloat = 0
    @State private var shieldActive = false
    @State private var showText = false
    @State private var nodeGlow: [Bool] = Array(repeating: false, count: 4)

    private let brainText: String = L.isRussian
        ? "Каждое решение проходит через тысячи проверок. Каждая мысль анализируется, прежде чем стать действием. Ничего случайного."
        : "Every decision passes through thousands of checks. Every thought is analyzed before becoming an action. Nothing is random."

    private struct PipelineNode {
        let name: String
        let icon: String
        let color: Color
    }

    private let nodes: [PipelineNode] = [
        PipelineNode(
            name: L.isRussian ? "Анализ" : "Analysis",
            icon: "magnifyingglass",
            color: Color(hex: "7B61FF")
        ),
        PipelineNode(
            name: L.isRussian ? "Стратегия" : "Strategy",
            icon: "target",
            color: Theme.Colors.accent
        ),
        PipelineNode(
            name: L.isRussian ? "Риск" : "Risk",
            icon: "shield.fill",
            color: Theme.Colors.warning
        ),
        PipelineNode(
            name: L.isRussian ? "Исполнение" : "Execution",
            icon: "bolt.fill",
            color: Theme.Colors.success
        ),
    ]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let centerY = h * 0.4
            let startX = w * 0.12
            let endX = w * 0.88
            let spacing = (endX - startX) / CGFloat(nodes.count - 1)

            ZStack {
                // Connection lines between nodes
                ForEach(0..<3, id: \.self) { i in
                    let fromX = startX + CGFloat(i) * spacing
                    let toX = startX + CGFloat(i + 1) * spacing
                    let from = CGPoint(x: fromX, y: centerY)
                    let to = CGPoint(x: toX, y: centerY)

                    ConnectionLine(progress: linesDrawn[i], from: from, to: to)
                        .stroke(
                            LinearGradient(
                                colors: [
                                    nodes[i].color.opacity(0.5),
                                    nodes[i + 1].color.opacity(0.5)
                                ],
                                startPoint: .leading,
                                endPoint: .trailing
                            ),
                            lineWidth: 2
                        )
                }

                // Traveling pulse
                if pulseStage >= 0 {
                    let currentX = startX + pulseX * (endX - startX)
                    ZStack {
                        Circle()
                            .fill(Theme.Colors.accent)
                            .frame(width: 14, height: 14)

                        Circle()
                            .fill(Theme.Colors.accent.opacity(0.4))
                            .frame(width: 30, height: 30)
                            .blur(radius: 6)
                    }
                    .position(x: currentX, y: centerY)
                }

                // Shield flash around Risk node (index 2)
                ShieldFlash(
                    isActive: shieldActive,
                    color: Theme.Colors.warning
                )
                .position(x: startX + 2 * spacing, y: centerY)

                // Pipeline nodes
                ForEach(0..<nodes.count, id: \.self) { i in
                    let x = startX + CGFloat(i) * spacing

                    NeuralNode(
                        label: nodes[i].name,
                        icon: nodes[i].icon,
                        color: nodes[i].color,
                        isActive: nodeGlow[i],
                        size: 52
                    )
                    .position(x: x, y: centerY)
                    .scaleEffect(nodesVisible[i] ? 1.0 : 0.0)
                    .opacity(nodesVisible[i] ? 1.0 : 0.0)
                }

                // Arrow indicators between nodes
                ForEach(0..<3, id: \.self) { i in
                    let midX = startX + (CGFloat(i) + 0.5) * spacing
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(Theme.Colors.textTertiary.opacity(linesDrawn[i]))
                        .position(x: midX, y: centerY - 35)
                }

                // Brain text
                VStack {
                    Spacer()
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
        // Nodes appear sequentially
        for i in 0..<nodes.count {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.6).delay(Double(i) * 0.3)) {
                nodesVisible[i] = true
            }
        }

        // Lines draw after nodes
        let lineStart = Double(nodes.count) * 0.3 + 0.2
        for i in 0..<3 {
            withAnimation(.easeInOut(duration: 0.5).delay(lineStart + Double(i) * 0.2)) {
                linesDrawn[i] = 1.0
            }
        }

        // Pulse travels through pipeline
        let pulseStart = lineStart + 3 * 0.2 + 0.5
        DispatchQueue.main.asyncAfter(deadline: .now() + pulseStart) {
            startPulse()
        }

        // Text
        withAnimation(.easeOut(duration: 1.0).delay(pulseStart + 3.5)) {
            showText = true
        }
    }

    private func startPulse() {
        pulseStage = 0

        // Pulse travels from left to right
        func activateNode(_ index: Int, at time: Double) {
            DispatchQueue.main.asyncAfter(deadline: .now() + time) {
                withAnimation(.easeOut(duration: 0.3)) {
                    nodeGlow[index] = true
                }

                // Shield flash at Risk node
                if index == 2 {
                    withAnimation(.easeOut(duration: 0.4)) {
                        shieldActive = true
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                        withAnimation(.easeOut(duration: 0.3)) {
                            shieldActive = false
                        }
                    }
                }

                // Dim after pulse passes
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                    if index < nodes.count - 1 {
                        withAnimation(.easeOut(duration: 0.5)) {
                            nodeGlow[index] = false
                        }
                    }
                }
            }
        }

        // Animate pulse position
        let totalDuration: Double = 3.0
        let stepDuration = totalDuration / Double(nodes.count - 1)

        for i in 0..<nodes.count {
            activateNode(i, at: stepDuration * Double(i))
        }

        // Animate the pulse dot smoothly
        withAnimation(.easeInOut(duration: totalDuration)) {
            pulseX = 1.0
        }
    }

    private func resetAnimations() {
        nodesVisible = Array(repeating: false, count: 4)
        linesDrawn = Array(repeating: 0, count: 3)
        pulseStage = -1
        pulseX = 0
        shieldActive = false
        showText = false
        nodeGlow = Array(repeating: false, count: 4)
    }
}

#Preview {
    GenesisScene4View(sceneActive: true)
        .preferredColorScheme(.dark)
}
