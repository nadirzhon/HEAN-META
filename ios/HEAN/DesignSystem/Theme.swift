//
//  Theme.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import SwiftUI

// MARK: - Color Extension for Hex
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r, g, b, a: UInt64
        switch hex.count {
        case 6:
            (r, g, b, a) = (int >> 16, int >> 8 & 0xFF, int & 0xFF, 255)
        case 8:
            (r, g, b, a) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (r, g, b, a) = (0, 0, 0, 255)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

enum Theme {
    enum Colors {
        static let background = Color(hex: "0A0A0F")
        static let card = Color(hex: "12121A")
        static let cardElevated = Color(hex: "1A1A24")
        static let accent = Color(hex: "00D4FF")
        static let success = Color(hex: "22C55E")
        static let error = Color(hex: "EF4444")
        static let warning = Color(hex: "F59E0B")

        static let textPrimary = Color.white
        static let textSecondary = Color.white.opacity(0.6)
        static let textTertiary = Color.white.opacity(0.4)

        static let border = Color.white.opacity(0.1)
        static let divider = Color.white.opacity(0.05)
    }

    enum Typography {
        static let largeTitle = Font.system(size: 34, weight: .bold, design: .rounded)
        static let title1 = Font.system(size: 28, weight: .bold)
        static let title2 = Font.system(size: 22, weight: .semibold)
        static let title3 = Font.system(size: 20, weight: .semibold)
        static let headline = Font.system(size: 17, weight: .semibold)
        static let body = Font.system(size: 17, weight: .regular)
        static let callout = Font.system(size: 16, weight: .regular)
        static let subheadline = Font.system(size: 15, weight: .regular)
        static let footnote = Font.system(size: 13, weight: .regular)
        static let caption1 = Font.system(size: 12, weight: .regular)
        static let caption2 = Font.system(size: 11, weight: .regular)

        static let monoLarge = Font.system(size: 28, weight: .bold, design: .monospaced)
        static let mono = Font.system(size: 17, weight: .medium, design: .monospaced)
        static let monoSmall = Font.system(size: 13, weight: .regular, design: .monospaced)
    }

    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 24
        static let xxl: CGFloat = 32
    }

    enum CornerRadius {
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
    }

    enum Animation {
        static let spring = SwiftUI.Animation.spring(response: 0.3, dampingFraction: 0.7)
        static let easeOut = SwiftUI.Animation.easeOut(duration: 0.2)
        static let easeIn = SwiftUI.Animation.easeIn(duration: 0.15)
    }
}

