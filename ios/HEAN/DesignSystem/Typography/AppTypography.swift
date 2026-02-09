//
//  AppTypography.swift
//  HEAN
//
//  Typography system for HEAN trading app
//

import SwiftUI

/// Design token typography
struct AppTypography {
    // MARK: - Spacing (8pt grid)
    static let xs: CGFloat = 8
    static let sm: CGFloat = 12
    static let md: CGFloat = 16
    static let lg: CGFloat = 20
    static let xl: CGFloat = 24

    // MARK: - Corner Radius
    static let radiusSm: CGFloat = 8
    static let radiusMd: CGFloat = 12
    static let radiusLg: CGFloat = 16

    // MARK: - Font Weights
    static let regular = Font.Weight.regular
    static let medium = Font.Weight.medium
    static let semibold = Font.Weight.semibold
    static let bold = Font.Weight.bold

    // MARK: - Font Styles
    static func display(_ size: CGFloat = 32, weight: Font.Weight = .bold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }

    static func title(_ size: CGFloat = 24, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }

    static func headline(_ size: CGFloat = 18, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }

    static func body(_ size: CGFloat = 16, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }

    static func caption(_ size: CGFloat = 14, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }

    static func mono(_ size: CGFloat = 16, weight: Font.Weight = .medium) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

/// Animation durations
struct AppAnimation {
    static let fast: Double = 0.15
    static let normal: Double = 0.25
    static let slow: Double = 0.4

    static let spring = Animation.spring(duration: 0.25, bounce: 0.15)
    static let easeOut = Animation.easeOut(duration: normal)
    static let easeIn = Animation.easeIn(duration: normal)
}

// MARK: - Double Extensions for Currency Formatting

extension Double {
    /// Format as currency: $1,234.56
    var asCurrency: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
        formatter.locale = Locale(identifier: "en_US")
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: self)) ?? "$0.00"
    }

    /// Format as compact currency: $1.2K, $3.4M
    var asCompactCurrency: String {
        let absValue = abs(self)
        let sign = self < 0 ? "-" : ""
        if absValue >= 1_000_000 {
            return "\(sign)$\(String(format: "%.1fM", absValue / 1_000_000))"
        } else if absValue >= 1_000 {
            return "\(sign)$\(String(format: "%.1fK", absValue / 1_000))"
        }
        return "\(sign)$\(String(format: "%.2f", absValue))"
    }

    /// Format as percentage: +3.45% or -1.23%
    var asPercent: String {
        String(format: "%+.2f%%", self)
    }

    /// Format as PnL: +$234.56 or -$45.67
    var asPnL: String {
        let sign = self >= 0 ? "+" : ""
        return "\(sign)\(self.asCurrency)"
    }

    /// Format as crypto quantity: 0.00123456
    var asCryptoQty: String {
        if abs(self) < 0.01 {
            return String(format: "%.8f", self)
        } else if abs(self) < 1 {
            return String(format: "%.6f", self)
        } else if abs(self) < 100 {
            return String(format: "%.4f", self)
        }
        return String(format: "%.2f", self)
    }

    /// Format as crypto price: $42,350.75 or $0.00001234
    var asCryptoPrice: String {
        if abs(self) < 0.01 {
            return "$\(String(format: "%.8f", self))"
        } else if abs(self) < 1 {
            return "$\(String(format: "%.6f", self))"
        }
        return asCurrency
    }
}
