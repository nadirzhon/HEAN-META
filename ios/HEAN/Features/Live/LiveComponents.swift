//
//  LiveComponents.swift
//  HEAN
//
//  Sub-views for the Live tab: AI explanation, physics, balance of forces, anomalies
//

import SwiftUI

// MARK: - AI Explanation Card

struct AIExplanationCard: View {
    let summary: String
    let regime: String

    var body: some View {
        GlassCardAccent {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(Theme.Colors.accent)
                    Text(L.aiExplanation)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text(regime.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(regimeColor.opacity(0.2))
                        .foregroundColor(regimeColor)
                        .cornerRadius(6)
                }

                Text(summary)
                    .font(.system(size: 13))
                    .foregroundColor(Theme.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineLimit(4)
            }
            .padding(14)
        }
    }

    private var regimeColor: Color {
        switch regime.lowercased() {
        case "trending", "bullish": return Theme.Colors.success
        case "ranging", "neutral": return Theme.Colors.warning
        case "volatile", "bearish": return Theme.Colors.error
        default: return .gray
        }
    }
}

// MARK: - Physics Compact Card (replaces PhysicsInlineSection)

struct PhysicsCompactCard: View {
    let physics: PhysicsState

    var body: some View {
        GlassCard(padding: 14) {
            VStack(spacing: 10) {
                // Header
                HStack {
                    Image(systemName: "atom")
                        .font(.system(size: 12))
                        .foregroundColor(Theme.Colors.accent)
                    Text("Market Physics")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(Theme.Colors.textSecondary)
                    Spacer()
                }

                // Three metrics in a row
                HStack(spacing: 0) {
                    // Temperature
                    VStack(spacing: 4) {
                        Image(systemName: "thermometer.medium")
                            .font(.system(size: 16))
                            .foregroundColor(temperatureColor)
                        Text(String(format: "%.0f°", physics.temperature))
                            .font(.system(size: 20, weight: .bold, design: .monospaced))
                            .foregroundColor(temperatureColor)
                            .lineLimit(1)
                        Text(temperatureLabel)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(Theme.Colors.textTertiary)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)

                    Rectangle()
                        .fill(Color.white.opacity(0.08))
                        .frame(width: 1, height: 44)

                    // Entropy
                    VStack(spacing: 4) {
                        Image(systemName: "waveform.path.ecg")
                            .font(.system(size: 16))
                            .foregroundColor(entropyColor)
                        Text(String(format: "%.2f", physics.entropy))
                            .font(.system(size: 20, weight: .bold, design: .monospaced))
                            .foregroundColor(entropyColor)
                            .lineLimit(1)
                        Text(entropyLabel)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(Theme.Colors.textTertiary)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)

                    Rectangle()
                        .fill(Color.white.opacity(0.08))
                        .frame(width: 1, height: 44)

                    // Phase
                    VStack(spacing: 4) {
                        Text(phaseEmoji)
                            .font(.system(size: 18))
                        Text(physics.phase.capitalized)
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(phaseColor)
                            .lineLimit(1)
                        Text(physics.phaseDisplayName)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(Theme.Colors.textTertiary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private var temperatureLabel: String {
        if physics.temperature < 30 { return L.cold }
        if physics.temperature < 70 { return L.warm }
        return L.hot
    }

    private var temperatureColor: Color {
        if physics.temperature < 30 { return .blue }
        if physics.temperature < 70 { return .orange }
        return .red
    }

    private var entropyLabel: String {
        if physics.entropy < 0.3 { return L.ordered }
        if physics.entropy < 0.7 { return L.mixed }
        return L.chaotic
    }

    private var entropyColor: Color {
        if physics.entropy < 0.3 { return Theme.Colors.success }
        if physics.entropy < 0.7 { return Theme.Colors.warning }
        return Theme.Colors.error
    }

    private var phaseEmoji: String {
        switch physics.phase.lowercased() {
        case "ice": return "🧊"
        case "water": return "💧"
        case "vapor": return "♨️"
        default: return "🔮"
        }
    }

    private var phaseColor: Color {
        switch physics.phase.lowercased() {
        case "ice": return .blue
        case "water": return .cyan
        case "vapor": return .red
        default: return .purple
        }
    }
}

// MARK: - Balance of Forces Card

struct BalanceOfForcesCard: View {
    let participants: ParticipantBreakdown

    var body: some View {
        GlassCard(padding: 14) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "scale.3d")
                        .foregroundColor(Theme.Colors.accent)
                    Text(L.balanceOfForces)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text(participants.dominantPlayer.replacingOccurrences(of: "_", with: " ").uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(Theme.Colors.warning)
                }

                // Buy vs Sell pressure
                let buyPressure = (participants.retailSentimentPercent + participants.whaleActivityPercent) / 2
                let sellPressure = 1.0 - buyPressure

                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(L.buy)
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(Theme.Colors.success)
                        Text("\(Int(buyPressure * 100))%")
                            .font(.system(size: 18, weight: .bold, design: .monospaced))
                            .foregroundColor(Theme.Colors.success)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Theme.Colors.error.opacity(0.3))
                                .frame(height: 6)
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Theme.Colors.success)
                                .frame(width: geo.size.width * buyPressure, height: 6)
                        }
                    }
                    .frame(height: 6)
                    .padding(.horizontal, 12)

                    VStack(alignment: .trailing, spacing: 2) {
                        Text(L.sell)
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(Theme.Colors.error)
                        Text("\(Int(sellPressure * 100))%")
                            .font(.system(size: 18, weight: .bold, design: .monospaced))
                            .foregroundColor(Theme.Colors.error)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }

                // Participant mini-bars
                HStack(spacing: 6) {
                    participantMini("MM", value: participants.mmActivityPercent, color: .blue)
                    participantMini("Inst", value: participants.institutionalFlowPercent, color: .purple)
                    participantMini("Ret", value: participants.retailSentimentPercent, color: .green)
                    participantMini("Whl", value: participants.whaleActivityPercent, color: .orange)
                    participantMini("Arb", value: participants.arbPressurePercent, color: .cyan)
                }
            }
        }
        .explainable(.buyPressure, value: (participants.retailSentimentPercent + participants.whaleActivityPercent) / 2)
    }

    private func participantMini(_ label: String, value: Double, color: Color) -> some View {
        VStack(spacing: 3) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(height: max(4, CGFloat(value) * 30))
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(Theme.Colors.textSecondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Whale Trades Section

struct WhaleTradesSection: View {
    let anomalies: [Anomaly]

    var whaleAnomalies: [Anomaly] {
        anomalies.filter { $0.type.lowercased() == "whale" }
    }

    var body: some View {
        if !whaleAnomalies.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "fish.fill")
                        .foregroundColor(.orange)
                    Text(L.whaleTrades)
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text("\(whaleAnomalies.count)")
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                }

                ForEach(whaleAnomalies.prefix(3)) { anomaly in
                    anomalyMiniCard(anomaly, color: .orange)
                }
            }
            .explainable(.whaleTrades, value: Double(whaleAnomalies.count) / 10.0)
        }
    }
}

// MARK: - Liquidations Section

struct LiquidationsSection: View {
    let anomalies: [Anomaly]

    var liquidationAnomalies: [Anomaly] {
        anomalies.filter { $0.type.lowercased() == "liquidation" }
    }

    var body: some View {
        if !liquidationAnomalies.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "bolt.fill")
                        .foregroundColor(.red)
                    Text(L.liquidations)
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text("\(liquidationAnomalies.count)")
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                }

                ForEach(liquidationAnomalies.prefix(3)) { anomaly in
                    anomalyMiniCard(anomaly, color: .red)
                }
            }
            .explainable(.liquidations, value: Double(liquidationAnomalies.count) / 10.0)
        }
    }
}

// MARK: - Shared Anomaly Mini Card

private func anomalyMiniCard(_ anomaly: Anomaly, color: Color) -> some View {
    HStack(spacing: 10) {
        Circle()
            .fill(color)
            .frame(width: 6, height: 6)

        Text(anomaly.description)
            .font(.caption)
            .foregroundColor(Theme.Colors.textPrimary)
            .lineLimit(1)

        Spacer()

        Text(anomaly.timestamp.suffix(9).prefix(8))
            .font(.caption2)
            .foregroundColor(Theme.Colors.textSecondary)
            .monospacedDigit()
    }
    .padding(.vertical, 4)
}

// MARK: - Szilard Profit Card

struct SzilardProfitCard: View {
    let profit: Double

    var body: some View {
        GlassCard(padding: 14) {
            HStack {
                Image(systemName: "atom")
                    .foregroundColor(profit >= 0 ? Theme.Colors.success : Theme.Colors.error)
                VStack(alignment: .leading, spacing: 2) {
                    Text(L.szilardProfit)
                        .font(.system(size: 11))
                        .foregroundColor(Theme.Colors.textSecondary)
                    Text(String(format: "%.4f", profit))
                        .font(.system(size: 15, weight: .bold, design: .monospaced))
                        .foregroundColor(profit >= 0 ? Theme.Colors.success : Theme.Colors.error)
                }
                Spacer()
                Text(profit >= 0 ? L.edgePresent : L.noEdge)
                    .font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background((profit >= 0 ? Theme.Colors.success : Theme.Colors.error).opacity(0.15))
                    .foregroundColor(profit >= 0 ? Theme.Colors.success : Theme.Colors.error)
                    .cornerRadius(6)
            }
        }
        .explainable(.szilardProfit, value: profit)
    }
}
