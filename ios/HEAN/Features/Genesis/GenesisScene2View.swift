//
//  GenesisScene2View.swift
//  HEAN
//
//  Scene 2: "Spark of Reason"
//  A bright point appears in the chaos. Order begins to emerge.
//

import SwiftUI

struct GenesisScene2View: View {
    let sceneActive: Bool

    @State private var sparkScale: CGFloat = 0
    @State private var sparkOpacity: CGFloat = 0
    @State private var ringScale: CGFloat = 0.5
    @State private var ringOpacity: CGFloat = 0
    @State private var orderProgress: CGFloat = 0
    @State private var showText = false
    @State private var chaosOpacity: CGFloat = 1.0

    private var brainText: String {
        L.isRussian
            ? "Мы задали вопрос: что, если убрать эмоции? Заменить их чистой, холодной логикой. Так родилась идея HEAN."
            : "We asked a question: what if we remove emotions? Replace them with pure, cold logic. This is how the idea of HEAN was born."
    }

    var body: some View {
        ZStack {
            // Remaining chaos (fading out)
            TimelineView(.animation) { context in
                let t = context.date.timeIntervalSinceReferenceDate
                SparkCanvas(time: t, orderProgress: orderProgress)
            }
            .opacity(chaosOpacity)

            // Central spark
            ZStack {
                // Shockwave rings
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .strokeBorder(
                            Theme.Colors.accent.opacity(ringOpacity * (1 - CGFloat(i) * 0.3)),
                            lineWidth: 2 - CGFloat(i) * 0.5
                        )
                        .frame(width: 60 + CGFloat(i) * 40, height: 60 + CGFloat(i) * 40)
                        .scaleEffect(ringScale + CGFloat(i) * 0.2)
                }

                // Core glow
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                Theme.Colors.accent,
                                Theme.Colors.accent.opacity(0.5),
                                Theme.Colors.accent.opacity(0)
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: 40
                        )
                    )
                    .frame(width: 80, height: 80)
                    .scaleEffect(sparkScale)
                    .opacity(sparkOpacity)

                // Central bright dot
                Circle()
                    .fill(.white)
                    .frame(width: 12, height: 12)
                    .scaleEffect(sparkScale)
                    .opacity(sparkOpacity)
            }

            // Brain text
            VStack {
                Spacer()
                BrainTextOverlay(text: brainText, isVisible: showText)
                    .padding(.bottom, 100)
            }
        }
        .onChange(of: sceneActive) { _, isActive in
            if isActive {
                animateIn()
            } else {
                resetAnimations()
            }
        }
        .onAppear {
            if sceneActive { animateIn() }
        }
    }

    private func animateIn() {
        // Spark appears
        withAnimation(.spring(response: 0.6, dampingFraction: 0.5).delay(0.5)) {
            sparkScale = 1.0
            sparkOpacity = 1.0
        }

        // Shockwave expands
        withAnimation(.easeOut(duration: 1.5).delay(0.8)) {
            ringScale = 1.5
            ringOpacity = 0.8
        }

        // Rings fade
        withAnimation(.easeOut(duration: 2.0).delay(2.0)) {
            ringOpacity = 0
        }

        // Order emerges, chaos fades
        withAnimation(.easeInOut(duration: 3.0).delay(1.5)) {
            orderProgress = 1.0
        }

        withAnimation(.easeOut(duration: 2.5).delay(3.0)) {
            chaosOpacity = 0.6
        }

        // Text appears
        withAnimation(.easeOut(duration: 1.0).delay(3.0)) {
            showText = true
        }
    }

    private func resetAnimations() {
        sparkScale = 0
        sparkOpacity = 0
        ringScale = 0.5
        ringOpacity = 0
        orderProgress = 0
        showText = false
        chaosOpacity = 1.0
    }
}

// Extracted Canvas to help Swift type-checker
private struct SparkCanvas: View {
    let time: Double
    let orderProgress: CGFloat

    var body: some View {
        Canvas { ctx, size in
            let w = size.width
            let h = size.height
            drawTransitionWaves(ctx: ctx, w: w, h: h)
        }
    }

    private func drawTransitionWaves(ctx: GraphicsContext, w: CGFloat, h: CGFloat) {
        let chaos: Double = max(0, 1.0 - orderProgress)
        let cyanMix: Double = orderProgress

        for i in 0..<12 {
            let fi = Double(i)
            let freq: Double = 1.5 + fi * 0.3 * (0.3 + chaos * 0.7)
            let speed: Double = 0.3 + fi * 0.1 * chaos
            let ampBase: CGFloat = h * 0.04 * (0.3 + chaos * 0.7)
            let ampMod: CGFloat = ampBase * (1.0 + 0.3 * sin(time * 0.2 + fi))
            let spread: CGFloat = 0.2 + chaos * 0.8
            let yBase: CGFloat = h * 0.5 + (h * 0.35 * CGFloat(fi - 6) / 6.0) * spread

            var path = Path()
            var first = true
            var x: CGFloat = 0

            while x <= w {
                let nx = Double(x / w)
                let y = yBase + ampMod * sin(nx * freq * .pi * 2 + time * speed)
                if first {
                    path.move(to: CGPoint(x: x, y: y))
                    first = false
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
                x += 3
            }

            let baseAlpha: Double = 0.2 + 0.15 * sin(time * 0.4 + fi)
            let color: Color
            if i % 2 == 0 {
                color = Color(
                    red: 0.94 * (1 - cyanMix),
                    green: 0.27 + 0.56 * cyanMix,
                    blue: 0.27 + 0.73 * cyanMix
                ).opacity(baseAlpha)
            } else {
                color = Color(
                    red: 0.13 * (1 - cyanMix),
                    green: 0.77 * (1 - cyanMix) + 0.83 * cyanMix,
                    blue: 0.37 + 0.63 * cyanMix
                ).opacity(baseAlpha)
            }

            ctx.stroke(path, with: .color(color), lineWidth: 1.5)
        }
    }
}

#Preview {
    GenesisScene2View(sceneActive: true)
        .preferredColorScheme(.dark)
}
