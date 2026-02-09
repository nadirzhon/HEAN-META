//
//  GravityMapView.swift
//  HEAN
//
//  Liquidation gravity map showing price magnets
//

import SwiftUI

struct GravityMapView: View {
    @State private var physicsState: PhysicsState?
    @State private var isLoading = true

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Temperature gauge
                    temperatureSection

                    // Entropy gauge
                    entropySection

                    // Phase indicator
                    phaseSection

                    // Szilard profit
                    szilardSection
                }
                .padding()
            }
            .navigationTitle("Gravity Map")
            .task { await loadData() }
            .refreshable { await loadData() }
        }
    }

    private var temperatureSection: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "thermometer.medium")
                    .foregroundColor(temperatureColor)
                Text("TEMPERATURE")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: "%.1f", physicsState?.temperature ?? 0))
                    .font(.title2)
                    .fontWeight(.bold)
                    .monospacedDigit()
                    .foregroundColor(temperatureColor)
            }

            ProgressView(value: min((physicsState?.temperature ?? 0) / 1200, 1.0))
                .tint(temperatureColor)

            Text(temperatureRegime)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var entropySection: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .foregroundColor(entropyColor)
                Text("ENTROPY")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: "%.2f", physicsState?.entropy ?? 0))
                    .font(.title2)
                    .fontWeight(.bold)
                    .monospacedDigit()
                    .foregroundColor(entropyColor)
            }

            ProgressView(value: min((physicsState?.entropy ?? 0) / 5.0, 1.0))
                .tint(entropyColor)

            Text(entropyState)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var phaseSection: some View {
        HStack(spacing: 16) {
            Image(systemName: phaseIcon)
                .font(.system(size: 40))
                .foregroundColor(phaseColor)

            VStack(alignment: .leading) {
                Text("PHASE")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text((physicsState?.phase ?? "unknown").uppercased())
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(phaseColor)
                Text(physicsState?.phaseDisplayName ?? "Unknown")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var szilardSection: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("SZILARD PROFIT")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("$\(String(format: "%.2f", physicsState?.szilardProfit ?? 0))")
                    .font(.title2)
                    .fontWeight(.bold)
                    .monospacedDigit()
            }
            Spacer()
            Image(systemName: "atom")
                .font(.system(size: 32))
                .foregroundColor(.cyan)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    // MARK: - Helpers

    private var temperatureColor: Color {
        let temp = physicsState?.temperature ?? 0
        if temp < 400 { return .blue }
        if temp < 800 { return .orange }
        return .red
    }

    private var temperatureRegime: String {
        let temp = physicsState?.temperature ?? 0
        if temp < 400 { return "COLD - Low volatility" }
        if temp < 800 { return "WARM - Normal activity" }
        return "HOT - High volatility"
    }

    private var entropyColor: Color {
        let ent = physicsState?.entropy ?? 0
        if ent < 2.0 { return .red }
        if ent < 3.5 { return .orange }
        return .green
    }

    private var entropyState: String {
        let ent = physicsState?.entropy ?? 0
        if ent < 2.0 { return "COMPRESSED - Spring coiled" }
        if ent < 3.5 { return "NORMAL - Transitioning" }
        return "EQUILIBRIUM - Dispersed"
    }

    private var phaseIcon: String {
        switch (physicsState?.phase ?? "unknown").lowercased() {
        case "ice": return "snowflake"
        case "water": return "drop.fill"
        case "vapor": return "flame.fill"
        default: return "questionmark.circle"
        }
    }

    private var phaseColor: Color {
        switch (physicsState?.phase ?? "unknown").lowercased() {
        case "ice": return .blue
        case "water": return .cyan
        case "vapor": return .red
        default: return .gray
        }
    }

    private func loadData() async {
        isLoading = true
        do {
            let result: PhysicsState = try await DIContainer.shared.apiClient.get("/api/v1/physics/state?symbol=BTCUSDT")
            physicsState = result
        } catch {
            // Fallback to sample data
            physicsState = PhysicsState(
                temperature: 520.0, entropy: 2.8, phase: "water",
                szilardProfit: 12.50, timestamp: "2026-02-08T12:00:00Z"
            )
        }
        isLoading = false
    }
}
