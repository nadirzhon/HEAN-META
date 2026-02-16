//
//  GenesisScene1View.swift
//  HEAN
//
//  Scene 1: "Chaos"
//  Chaotic market lines in red/green — raw financial noise.
//

import SwiftUI

struct GenesisScene1View: View {
    let sceneActive: Bool

    @State private var showText = false
    @State private var lineOpacity: CGFloat = 0

    private var brainText: String {
        L.isRussian
            ? "Мир финансов. Миллиарды решений, продиктованных страхом и жадностью. Хаос эмоций."
            : "The world of finance. Billions of decisions driven by fear and greed. A chaos of emotions."
    }

    var body: some View {
        ZStack {
            // Chaotic market lines
            TimelineView(.animation) { context in
                let t = context.date.timeIntervalSinceReferenceDate
                ChaosCanvas(time: t)
            }
            .opacity(lineOpacity)

            // Vignette overlay
            RadialGradient(
                colors: [.clear, Theme.Colors.background.opacity(0.6)],
                center: .center,
                startRadius: 100,
                endRadius: 400
            )
            .ignoresSafeArea()

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
        withAnimation(.easeIn(duration: 1.5)) {
            lineOpacity = 1
        }
        withAnimation(.easeOut(duration: 1.0).delay(2.0)) {
            showText = true
        }
    }

    private func resetAnimations() {
        showText = false
        lineOpacity = 0
    }
}

// Extracted Canvas view for type-checker performance
private struct ChaosCanvas: View {
    let time: Double

    var body: some View {
        Canvas { ctx, size in
            let w = size.width
            let h = size.height

            drawWaves(ctx: ctx, w: w, h: h)
            drawNoiseDots(ctx: ctx, w: w, h: h)
        }
    }

    private func drawWaves(ctx: GraphicsContext, w: CGFloat, h: CGFloat) {
        for i in 0..<18 {
            let fi = Double(i)
            let freq: Double = 1.5 + fi * 0.4
            let speed: Double = 0.4 + fi * 0.15
            let ampBase: CGFloat = h * 0.06
            let ampMod: CGFloat = ampBase * (1.0 + 0.4 * sin(time * 0.2 + fi))
            let yBase: CGFloat = h * CGFloat(i + 1) / 20.0

            var path = Path()
            var first = true
            var x: CGFloat = 0
            let step: CGFloat = 3

            while x <= w {
                let nx = Double(x / w)
                let wave1 = sin(nx * freq * .pi * 2 + time * speed)
                let wave2 = sin(nx * freq * 2.5 * .pi + time * speed * 1.7 + fi)
                let y = yBase + ampMod * wave1 + ampMod * 0.3 * wave2

                if first {
                    path.move(to: CGPoint(x: x, y: y))
                    first = false
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
                x += step
            }

            let alpha: Double = 0.3 + 0.2 * sin(time * 0.5 + fi * 0.7)
            let color: Color
            if i % 2 == 0 {
                color = Color(red: 0.94, green: 0.27, blue: 0.27).opacity(alpha)
            } else {
                color = Color(red: 0.13, green: 0.77, blue: 0.37).opacity(alpha)
            }

            ctx.stroke(path, with: .color(color), lineWidth: 1.5)
        }
    }

    private func drawNoiseDots(ctx: GraphicsContext, w: CGFloat, h: CGFloat) {
        for i in 0..<40 {
            let fi = Double(i)
            let x = w * CGFloat((sin(time * 0.3 + fi * 2.1) + 1) / 2)
            let y = h * CGFloat((cos(time * 0.25 + fi * 1.7) + 1) / 2)
            let dotSize: CGFloat = 2 + 2 * CGFloat(sin(time + fi))

            let dotColor: Color
            if i % 3 == 0 {
                dotColor = Color.red.opacity(0.3)
            } else if i % 3 == 1 {
                dotColor = Color.green.opacity(0.3)
            } else {
                dotColor = Color.white.opacity(0.1)
            }

            let rect = CGRect(x: x - dotSize / 2, y: y - dotSize / 2, width: dotSize, height: dotSize)
            ctx.fill(Path(ellipseIn: rect), with: .color(dotColor))
        }
    }
}

#Preview {
    GenesisScene1View(sceneActive: true)
        .preferredColorScheme(.dark)
}
