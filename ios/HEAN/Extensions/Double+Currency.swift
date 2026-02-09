//
//  Double+Currency.swift
//  HEAN
//

import Foundation

extension Double {
    /// Format as currency: $1,234.56
    var asCurrency: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
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
