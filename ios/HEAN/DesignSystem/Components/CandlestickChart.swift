//
//  CandlestickChart.swift
//  HEAN
//
//  Full candlestick chart with gestures
//

import SwiftUI

// Note: Candle struct is defined in Models/Candle.swift

/// Extension to add computed properties for chart rendering
extension Candle {
    var isBullish: Bool {
        close >= open
    }

    var chartColor: Color {
        isBullish ? Theme.Colors.success : Theme.Colors.error
    }
}

/// Full candlestick chart with pinch/drag gestures
struct CandlestickChart: View {
    let candles: [Candle]
    let currentPrice: Double?
    let showGrid: Bool
    let showVolume: Bool

    @State private var scale: CGFloat = 1.0
    @State private var offset: CGFloat = 0
    @State private var lastDragValue: CGFloat = 0

    init(
        candles: [Candle],
        currentPrice: Double? = nil,
        showGrid: Bool = true,
        showVolume: Bool = true
    ) {
        self.candles = candles
        self.currentPrice = currentPrice
        self.showGrid = showGrid
        self.showVolume = showVolume
    }

    private var visibleCandles: [Candle] {
        guard !candles.isEmpty else { return [] }

        let candleWidth: CGFloat = 8
        let spacing: CGFloat = 4
        let totalWidth = (candleWidth + spacing) * scale

        let visibleCount = max(Int(UIScreen.main.bounds.width / totalWidth), 1)
        let offsetCount = Int(-offset / totalWidth)

        let start = max(0, candles.count - visibleCount - offsetCount)
        let end = min(candles.count, start + visibleCount)

        return Array(candles[start..<end])
    }

    private var priceRange: (min: Double, max: Double) {
        guard !visibleCandles.isEmpty else { return (0, 1) }

        let prices = visibleCandles.flatMap { [$0.high, $0.low] }
        let minPrice = prices.min() ?? 0
        let maxPrice = prices.max() ?? 1

        let padding = (maxPrice - minPrice) * 0.1
        return (minPrice - padding, maxPrice + padding)
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topTrailing) {
                // Background
                Theme.Colors.card

                // Grid
                if showGrid {
                    gridLines(in: geometry.size)
                }

                // Candles
                candlesView(in: geometry.size)

                // Current price line
                if let price = currentPrice {
                    currentPriceLine(price: price, in: geometry.size)
                }

                // Price scale
                priceScale(in: geometry.size)
            }
            .cornerRadius(Theme.CornerRadius.md)
            .gesture(
                DragGesture()
                    .onChanged { value in
                        let delta = value.translation.width - lastDragValue
                        offset += delta
                        lastDragValue = value.translation.width

                        // Clamp offset
                        let maxOffset: CGFloat = 0
                        let minOffset = -CGFloat(candles.count) * 12 * scale
                        offset = min(maxOffset, max(minOffset, offset))
                    }
                    .onEnded { _ in
                        lastDragValue = 0
                    }
            )
            .gesture(
                MagnificationGesture()
                    .onChanged { value in
                        scale = max(0.5, min(3.0, value))
                    }
            )
        }
        .frame(height: 300)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Candlestick chart showing \(candles.count) candles")
    }

    private func gridLines(in size: CGSize) -> some View {
        Canvas { context, size in
            let horizontalLines = 5
            let step = size.height / CGFloat(horizontalLines)

            for i in 0...horizontalLines {
                let y = CGFloat(i) * step
                var path = Path()
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                context.stroke(
                    path,
                    with: .color(Theme.Colors.textTertiary.opacity(0.2)),
                    lineWidth: 1
                )
            }
        }
    }

    private func candlesView(in size: CGSize) -> some View {
        Canvas { context, size in
            let candleWidth: CGFloat = 8 * scale
            let spacing: CGFloat = 4 * scale
            let totalWidth = candleWidth + spacing

            let (minPrice, maxPrice) = priceRange
            let priceHeight = maxPrice - minPrice

            for (index, candle) in visibleCandles.enumerated() {
                let x = CGFloat(index) * totalWidth + offset + totalWidth / 2

                // Normalize prices to canvas height
                let openY = size.height - ((candle.open - minPrice) / priceHeight) * size.height
                let closeY = size.height - ((candle.close - minPrice) / priceHeight) * size.height
                let highY = size.height - ((candle.high - minPrice) / priceHeight) * size.height
                let lowY = size.height - ((candle.low - minPrice) / priceHeight) * size.height

                // Wick (high-low line)
                var wickPath = Path()
                wickPath.move(to: CGPoint(x: x, y: highY))
                wickPath.addLine(to: CGPoint(x: x, y: lowY))
                context.stroke(
                    wickPath,
                    with: .color(candle.chartColor),
                    lineWidth: 1
                )

                // Body (open-close rectangle)
                let bodyHeight = abs(closeY - openY)
                let bodyY = min(openY, closeY)

                let bodyRect = CGRect(
                    x: x - candleWidth / 2,
                    y: bodyHeight < 2 ? bodyY - 1 : bodyY,
                    width: candleWidth,
                    height: max(bodyHeight, 2)
                )

                context.fill(
                    Path(roundedRect: bodyRect, cornerRadius: 2),
                    with: .color(candle.chartColor)
                )
            }
        }
    }

    private func currentPriceLine(price: Double, in size: CGSize) -> some View {
        let (minPrice, maxPrice) = priceRange
        let priceHeight = maxPrice - minPrice
        let y = size.height - ((price - minPrice) / priceHeight) * size.height

        return Path { path in
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width - 60, y: y))
        }
        .stroke(
            Theme.Colors.accent,
            style: StrokeStyle(lineWidth: 1, dash: [5, 5])
        )
    }

    private func priceScale(in size: CGSize) -> some View {
        let (minPrice, maxPrice) = priceRange
        let steps = 5

        return VStack(alignment: .trailing, spacing: 0) {
            ForEach(0...steps, id: \.self) { i in
                let price = maxPrice - (maxPrice - minPrice) * Double(i) / Double(steps)
                Text(String(format: "$%.2f", price))
                    .font(Theme.Typography.caption(10, weight: .medium))
                    .foregroundColor(Theme.Colors.textTertiary)
                    .frame(width: 55, alignment: .trailing)
                    .padding(.trailing, 4)

                if i < steps {
                    Spacer()
                }
            }
        }
        .frame(width: 60, height: size.height)
        .background(Theme.Colors.background.opacity(0.8))
    }
}

// MARK: - Preview
#Preview("CandlestickChart") {
    ZStack {
        Theme.Colors.background
            .ignoresSafeArea()

        VStack {
            Text("BTC/USDT - 1H")
                .font(Theme.Typography.headlineFont())
                .foregroundColor(Theme.Colors.textPrimary)

            // Generate sample candles
            CandlestickChart(
                candles: generateSampleCandles(),
                currentPrice: 42_500,
                showGrid: true,
                showVolume: true
            )

            Text("Pinch to zoom • Drag to scroll")
                .font(Theme.Typography.caption())
                .foregroundColor(Theme.Colors.textSecondary)
        }
        .padding(Theme.Spacing.lg)
    }
}

// MARK: - Sample Data Generator
private func generateSampleCandles() -> [Candle] {
    var candles: [Candle] = []
    var price = 42_000.0
    let calendar = Calendar.current

    for i in 0..<50 {
        let timestamp = calendar.date(byAdding: .hour, value: -50 + i, to: Date()) ?? Date()

        let open = price
        let change = Double.random(in: -200...200)
        let close = open + change
        let high = max(open, close) + Double.random(in: 0...100)
        let low = min(open, close) - Double.random(in: 0...100)
        let volume = Double.random(in: 1000...5000)

        candles.append(Candle(
            timestamp: timestamp,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        ))

        price = close
    }

    return candles
}
