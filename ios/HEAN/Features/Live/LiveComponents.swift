//
//  LiveComponents.swift
//  HEAN
//
//  Sub-views for the Live tab: AI explanation, physics gauges, balance of forces, anomalies
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
                    Text("AI Explanation")
                        .font(.headline)
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text(regime.uppercased())
                        .font(.caption2)
                        .fontWeight(.bold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(regimeColor.opacity(0.2))
                        .foregroundColor(regimeColor)
                        .cornerRadius(6)
                }

                Text(summary)
                    .font(.subheadline)
                    .foregroundColor(Theme.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding()
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

// MARK: - Physics Inline Section

struct PhysicsInlineSection: View {
    let physics: PhysicsState

    var body: some View {
        HStack(spacing: 12) {
            InfoTile(
                icon: "thermometer.medium",
                title: "Temperature",
                value: String(format: "%.0f°", physics.temperature),
                subtitle: temperatureLabel,
                color: temperatureColor,
                term: .temperature,
                numericValue: physics.temperaturePercent
            )

            InfoTile(
                icon: "waveform.path.ecg",
                title: "Entropy",
                value: String(format: "%.2f", physics.entropy),
                subtitle: entropyLabel,
                color: entropyColor,
                term: .entropy,
                numericValue: physics.entropyPercent
            )

            InfoTile(
                icon: "drop.fill",
                title: "Phase",
                value: phaseEmoji,
                subtitle: physics.phaseDisplayName,
                color: phaseColor,
                term: .phase,
                numericValue: nil
            )
        }
    }

    private var temperatureLabel: String {
        if physics.temperature < 30 { return "Cold" }
        if physics.temperature < 70 { return "Warm" }
        return "Hot"
    }

    private var temperatureColor: Color {
        if physics.temperature < 30 { return .blue }
        if physics.temperature < 70 { return .orange }
        return .red
    }

    private var entropyLabel: String {
        if physics.entropy < 0.3 { return "Ordered" }
        if physics.entropy < 0.7 { return "Mixed" }
        return "Chaotic"
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
        GlassCard(padding: 16) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "scale.3d")
                        .foregroundColor(Theme.Colors.accent)
                    Text("Balance of Forces")
                        .font(.headline)
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text(participants.dominantPlayer.replacingOccurrences(of: "_", with: " ").uppercased())
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundColor(Theme.Colors.warning)
                }

                // Buy vs Sell pressure visualization
                let buyPressure = (participants.retailSentimentPercent + participants.whaleActivityPercent) / 2
                let sellPressure = 1.0 - buyPressure

                HStack(spacing: 0) {
                    // Buy side
                    VStack(alignment: .leading, spacing: 2) {
                        Text("BUY")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(Theme.Colors.success)
                        Text("\(Int(buyPressure * 100))%")
                            .font(.system(.title3, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(Theme.Colors.success)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    // Pressure bar
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Theme.Colors.error.opacity(0.3))
                                .frame(height: 8)
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Theme.Colors.success)
                                .frame(width: geo.size.width * buyPressure, height: 8)
                        }
                    }
                    .frame(height: 8)
                    .padding(.horizontal, 12)

                    // Sell side
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("SELL")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(Theme.Colors.error)
                        Text("\(Int(sellPressure * 100))%")
                            .font(.system(.title3, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(Theme.Colors.error)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }

                // Participant mini-bars
                HStack(spacing: 8) {
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
        VStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(height: max(4, CGFloat(value) * 30))
            Text(label)
                .font(.system(size: 8))
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
                    Text("Whale Trades")
                        .font(.subheadline)
                        .fontWeight(.bold)
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
                    Text("Liquidations")
                        .font(.subheadline)
                        .fontWeight(.bold)
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
                    Text("Szilard Profit")
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                    Text(String(format: "%.4f", profit))
                        .font(.system(.subheadline, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(profit >= 0 ? Theme.Colors.success : Theme.Colors.error)
                }
                Spacer()
                Text(profit >= 0 ? "Edge Present" : "No Edge")
                    .font(.caption2)
                    .fontWeight(.bold)
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
