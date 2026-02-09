//
//  Sparkline.swift
//  HEAN
//
//  Mini trend chart component
//

import SwiftUI

/// Mini trend chart with auto-coloring and optional gradient fill
struct Sparkline: View {
    let dataPoints: [Double]
    let showGradient: Bool
    let smoothCurves: Bool
    let lineWidth: CGFloat
    let overrideColor: Color?

    init(
        dataPoints: [Double],
        showGradient: Bool = true,
        smoothCurves: Bool = true,
        lineWidth: CGFloat = 2
    ) {
        self.dataPoints = dataPoints
        self.showGradient = showGradient
        self.smoothCurves = smoothCurves
        self.lineWidth = lineWidth
        self.overrideColor = nil
    }

    /// Convenience initializer with explicit color and fill options
    init(
        data: [Double],
        color: Color? = nil,
        lineWidth: CGFloat = 2,
        showFill: Bool = true
    ) {
        self.dataPoints = data
        self.showGradient = showFill
        self.smoothCurves = true
        self.lineWidth = lineWidth
        self.overrideColor = color
    }

    private var trend: Trend {
        guard let first = dataPoints.first, let last = dataPoints.last else {
            return .neutral
        }
        return last >= first ? .up : .down
    }

    private var lineColor: Color {
        if let color = overrideColor {
            return color
        }
        switch trend {
        case .up: return AppColors.success
        case .down: return AppColors.error
        case .neutral: return AppColors.textSecondary
        }
    }

    private enum Trend {
        case up, down, neutral
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Gradient fill
                if showGradient {
                    sparklinePath(in: geometry.size, closed: true)
                        .fill(
                            LinearGradient(
                                colors: [
                                    lineColor.opacity(0.3),
                                    lineColor.opacity(0.05),
                                    lineColor.opacity(0.0)
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                }

                // Line
                sparklinePath(in: geometry.size, closed: false)
                    .stroke(lineColor, lineWidth: lineWidth)
            }
        }
        .frame(height: 60)
        .accessibilityLabel("Sparkline chart showing \(trend == .up ? "upward" : trend == .down ? "downward" : "neutral") trend")
    }

    private func sparklinePath(in size: CGSize, closed: Bool) -> Path {
        guard dataPoints.count > 1 else { return Path() }

        let minValue = dataPoints.min() ?? 0
        let maxValue = dataPoints.max() ?? 1
        let range = maxValue - minValue

        // Avoid division by zero
        let normalizedRange = range > 0 ? range : 1

        var path = Path()

        let stepX = size.width / CGFloat(dataPoints.count - 1)

        for (index, value) in dataPoints.enumerated() {
            let x = CGFloat(index) * stepX
            let normalizedValue = (value - minValue) / normalizedRange
            let y = size.height - (normalizedValue * size.height)

            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                if smoothCurves {
                    // Bezier curve for smoothness
                    let previousX = CGFloat(index - 1) * stepX
                    let previousValue = (dataPoints[index - 1] - minValue) / normalizedRange
                    let previousY = size.height - (previousValue * size.height)

                    let controlX = (previousX + x) / 2
                    path.addQuadCurve(
                        to: CGPoint(x: x, y: y),
                        control: CGPoint(x: controlX, y: previousY)
                    )
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
        }

        // Close path for gradient fill
        if closed {
            let lastX = CGFloat(dataPoints.count - 1) * stepX
            path.addLine(to: CGPoint(x: lastX, y: size.height))
            path.addLine(to: CGPoint(x: 0, y: size.height))
            path.closeSubpath()
        }

        return path
    }
}

// MARK: - Preview
#Preview("Sparkline Variants") {
    ZStack {
        AppColors.backgroundPrimary
            .ignoresSafeArea()

        VStack(spacing: AppTypography.lg) {
            // Upward trend
            VStack(alignment: .leading, spacing: AppTypography.xs) {
                Text("Upward Trend (Smooth)")
                    .font(AppTypography.caption())
                    .foregroundColor(AppColors.textSecondary)

                Sparkline(
                    dataPoints: [100, 105, 103, 110, 115, 112, 120, 125, 130],
                    smoothCurves: true
                )
            }
            .padding(AppTypography.md)
            .background(AppColors.backgroundSecondary)
            .cornerRadius(AppTypography.radiusMd)

            // Downward trend
            VStack(alignment: .leading, spacing: AppTypography.xs) {
                Text("Downward Trend (Angular)")
                    .font(AppTypography.caption())
                    .foregroundColor(AppColors.textSecondary)

                Sparkline(
                    dataPoints: [130, 125, 128, 120, 115, 118, 110, 105, 100],
                    smoothCurves: false
                )
            }
            .padding(AppTypography.md)
            .background(AppColors.backgroundSecondary)
            .cornerRadius(AppTypography.radiusMd)

            // Volatile
            VStack(alignment: .leading, spacing: AppTypography.xs) {
                Text("Volatile (No Gradient)")
                    .font(AppTypography.caption())
                    .foregroundColor(AppColors.textSecondary)

                Sparkline(
                    dataPoints: [100, 110, 95, 115, 90, 120, 100, 125, 105],
                    showGradient: false
                )
            }
            .padding(AppTypography.md)
            .background(AppColors.backgroundSecondary)
            .cornerRadius(AppTypography.radiusMd)
        }
        .padding(AppTypography.xl)
    }
}
