//
//  Animations.swift
//  HEAN
//
//  Animation presets for HEAN app
//

import SwiftUI

/// Animation presets for consistent motion throughout the app
enum Animations {
    /// Fast animation (0.15s) - for quick feedback
    static let fast = Animation.easeOut(duration: 0.15)

    /// Normal animation (0.25s) - standard transitions
    static let normal = Animation.easeOut(duration: 0.25)

    /// Slow animation (0.4s) - dramatic transitions
    static let slow = Animation.easeOut(duration: 0.4)

    /// Spring animation with slight bounce
    static let spring = Animation.spring(response: 0.3, dampingFraction: 0.7)

    /// Bouncy spring for playful interactions
    static let bouncy = Animation.spring(response: 0.35, dampingFraction: 0.5)

    /// Gentle spring for subtle movements
    static let gentle = Animation.spring(response: 0.4, dampingFraction: 0.8)

    /// Pulse animation for attention
    static let pulse = Animation.easeInOut(duration: 0.6).repeatForever(autoreverses: true)

    /// Linear animation for continuous motion
    static let linear = Animation.linear(duration: 1.0)
}

/// Animation durations for convenience
enum AnimationDuration {
    static let fast: Double = 0.15
    static let normal: Double = 0.25
    static let slow: Double = 0.4
}
