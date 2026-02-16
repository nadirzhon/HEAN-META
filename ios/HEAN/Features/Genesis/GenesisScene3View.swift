//
//  GenesisScene3View.swift
//  HEAN
//
//  Scene 3: "Architecture of Consciousness"
//  Agent nodes appear connected by neural pathways. Pulses flow between them.
//  Central hub glow is handled by GenesisHeroLayer.
//

import SwiftUI

struct GenesisScene3View: View {
    let sceneActive: Bool

    @State private var nodeAppeared: [Bool] = Array(repeating: false, count: 6)
    @State private var lineProgress: [CGFloat] = Array(repeating: 0, count: 8)
    @State private var pulsePositions: [CGFloat] = Array(repeating: 0, count: 8)
    @State private var showText = false
    @State private var isPulsing = false

    private let brainText: String = L.isRussian
        ? "Мой разум — это не монолит. Это симбиоз специализированных агентов. Каждый из них — эксперт в своей области: от анализа рисков до поиска новых идей."
        : "My mind is not a monolith. It is a symbiosis of specialized agents. Each one is an expert in its field: from risk analysis to finding new ideas."

    private struct AgentInfo {
        let name: String
        let icon: String
        let color: Color
    }

    private let agents: [AgentInfo] = [
        AgentInfo(name: L.isRussian ? "Оракул" : "Oracle", icon: "eye.fill", color: Color(hex: "7B61FF")),
        AgentInfo(name: L.isRussian ? "Стратегия" : "Strategy", icon: "target", color: Theme.Colors.accent),
        AgentInfo(name: L.isRussian ? "Физика" : "Physics", icon: "waveform.path", color: Color(hex: "F97316")),
        AgentInfo(name: L.isRussian ? "Риск" : "Risk", icon: "shield.fill", color: Theme.Colors.warning),
        AgentInfo(name: L.isRussian ? "Мозг" : "Brain", icon: "brain.head.profile", color: Theme.Colors.success),
        AgentInfo(name: L.isRussian ? "Исполнение" : "Execution", icon: "bolt.fill", color: Color(hex: "3B82F6")),
    ]

    // Connections: pairs of agent indices
    private let connections: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), // Ring
        (0, 3), (1, 4), // Cross connections
    ]

    var body: some View {
        GeometryReader { geo in
            let center = CGPoint(x: geo.size.width / 2, y: geo.size.height * 0.42)
            let radius: CGFloat = min(geo.size.width, geo.size.height) * 0.28

            ZStack {
                // Connection lines
                ForEach(0..<connections.count, id: \.self) { i in
                    let conn = connections[i]
                    let from = nodePosition(index: conn.0, center: center, radius: radius)
                    let to = nodePosition(index: conn.1, center: center, radius: radius)

                    ConnectionLine(progress: lineProgress[i], from: from, to: to)
                        .stroke(
                            Theme.Colors.accent.opacity(0.3),
                            lineWidth: i < 6 ? 1.5 : 1.0
                        )

                    // Traveling pulse dot
                    if isPulsing && lineProgress[i] >= 1.0 {
                        let pos = interpolate(from: from, to: to, t: pulsePositions[i])
                        Circle()
                            .fill(Theme.Colors.accent)
                            .frame(width: 6, height: 6)
                            .blur(radius: 2)
                            .position(pos)
                    }
                }

                // Agent nodes
                ForEach(0..<agents.count, id: \.self) { i in
                    let pos = nodePosition(index: i, center: center, radius: radius)

                    NeuralNode(
                        label: agents[i].name,
                        icon: agents[i].icon,
                        color: agents[i].color,
                        isActive: nodeAppeared[i],
                        size: 48
                    )
                    .position(pos)
                    .scaleEffect(nodeAppeared[i] ? 1.0 : 0.3)
                    .opacity(nodeAppeared[i] ? 1.0 : 0)
                }

                // Note: Central hub glow is now rendered by GenesisHeroLayer

                // Brain text
                VStack {
                    Spacer()
                    BrainTextOverlay(text: brainText, isVisible: showText)
                        .padding(.bottom, 100)
                }
                .frame(width: geo.size.width, height: geo.size.height)
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

    private func nodePosition(index: Int, center: CGPoint, radius: CGFloat) -> CGPoint {
        let angle = (Double(index) / Double(agents.count)) * .pi * 2 - .pi / 2
        return CGPoint(
            x: center.x + radius * cos(angle),
            y: center.y + radius * sin(angle)
        )
    }

    private func interpolate(from: CGPoint, to: CGPoint, t: CGFloat) -> CGPoint {
        CGPoint(
            x: from.x + (to.x - from.x) * t,
            y: from.y + (to.y - from.y) * t
        )
    }

    private func animateSequence() {
        // Nodes appear one by one
        for i in 0..<agents.count {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.6).delay(Double(i) * 0.3)) {
                nodeAppeared[i] = true
            }
        }

        // Lines draw after all nodes
        let lineDelay = Double(agents.count) * 0.3 + 0.2
        for i in 0..<connections.count {
            withAnimation(.easeInOut(duration: 0.6).delay(lineDelay + Double(i) * 0.15)) {
                lineProgress[i] = 1.0
            }
        }

        // Start pulsing after lines are drawn
        let pulseDelay = lineDelay + Double(connections.count) * 0.15 + 0.3
        DispatchQueue.main.asyncAfter(deadline: .now() + pulseDelay) {
            isPulsing = true
            startPulseAnimation()
        }

        // Text appears
        withAnimation(.easeOut(duration: 1.0).delay(pulseDelay + 0.5)) {
            showText = true
        }
    }

    private func startPulseAnimation() {
        // Continuous pulse animation along connections
        func animatePulse() {
            for i in 0..<connections.count {
                let delay = Double(i) * 0.2
                withAnimation(.linear(duration: 1.2).delay(delay)) {
                    pulsePositions[i] = 1.0
                }
                // Reset after completion
                DispatchQueue.main.asyncAfter(deadline: .now() + delay + 1.2) {
                    pulsePositions[i] = 0.0
                }
            }
        }

        // Repeat the pulse
        func repeatPulse() {
            animatePulse()
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                if isPulsing { repeatPulse() }
            }
        }

        repeatPulse()
    }

    private func resetAnimations() {
        isPulsing = false
        showText = false
        nodeAppeared = Array(repeating: false, count: 6)
        lineProgress = Array(repeating: 0, count: 8)
        pulsePositions = Array(repeating: 0, count: 8)
    }
}

#Preview {
    GenesisScene3View(sceneActive: true)
        .preferredColorScheme(.dark)
}
