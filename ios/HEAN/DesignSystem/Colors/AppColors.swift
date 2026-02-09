//
//  AppColors.swift
//  HEAN
//
//  Premium color system for HEAN trading app
//

import SwiftUI

/// Design token colors for HEAN app
struct AppColors {
    // MARK: - Backgrounds
    static let backgroundPrimary = Color(red: 10/255, green: 10/255, blue: 15/255)       // 0A0A0F
    static let backgroundSecondary = Color(red: 18/255, green: 18/255, blue: 26/255)     // 12121A
    static let backgroundTertiary = Color(red: 26/255, green: 26/255, blue: 36/255)      // 1A1A24

    // MARK: - Accents
    static let accentPrimary = Color(red: 0/255, green: 212/255, blue: 255/255)          // 00D4FF
    static let success = Color(red: 34/255, green: 197/255, blue: 94/255)                // 22C55E
    static let error = Color(red: 239/255, green: 68/255, blue: 68/255)                  // EF4444
    static let warning = Color(red: 245/255, green: 158/255, blue: 11/255)               // F59E0B

    // MARK: - Text
    static let textPrimary = Color.white                                                  // FFFFFF
    static let textSecondary = Color(red: 161/255, green: 161/255, blue: 170/255)        // A1A1AA
    static let textTertiary = Color(red: 113/255, green: 113/255, blue: 122/255)         // 71717A

    // MARK: - Semantic
    static let bullish = success
    static let bearish = error

    // MARK: - Glass Effects
    static let glassBorder = LinearGradient(
        colors: [
            Color.white.opacity(0.2),
            Color.white.opacity(0.05)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}
