//
//  SettingsView.swift
//  HEAN
//
//  App settings with API configuration and preferences
//

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = SettingsViewModel()
    @AppStorage("apiBaseURL") private var apiURL: String = "http://localhost:8000"
    @AppStorage("wsBaseURL") private var wsURL: String = "ws://localhost:8000/ws"
    @AppStorage("apiAuthKey") private var authKey: String = ""
    @AppStorage("enableHaptics") private var enableHaptics: Bool = true
    @AppStorage("enableNotifications") private var enableNotifications: Bool = true
    @AppStorage("showConfidenceScores") private var showConfidence: Bool = true
    @AppStorage("maxRiskPercent") private var maxRiskPercent: Double = 2.0
    @AppStorage("maxLeverage") private var maxLeverage: Double = 5.0
    @AppStorage("minRiskReward") private var minRiskReward: Double = 1.5
    @AppStorage("analysisInterval") private var analysisInterval: Double = 30.0
    @AppStorage("confidenceThreshold") private var confidenceThreshold: Double = 0.6
    @State private var showSaved = false
    @State private var engineMessage: String?
    @State private var isEngineLoading = false
    @State private var showStartConfirm = false
    @State private var showStopConfirm = false
    @State private var showPauseConfirm = false
    @State private var showResumeConfirm = false
    @State private var showKillConfirm = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("API Base URL").font(.caption).foregroundColor(.gray)
                        TextField("http://localhost:8000", text: $apiURL)
                            .font(.system(.body, design: .monospaced))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("WebSocket URL").font(.caption).foregroundColor(.gray)
                        TextField("ws://localhost:8000/ws", text: $wsURL)
                            .font(.system(.body, design: .monospaced))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Auth Key (optional)").font(.caption).foregroundColor(.gray)
                        SecureField("API authentication key", text: $authKey)
                            .font(.system(.body, design: .monospaced))
                    }
                } header: {
                    Label("Connection", systemImage: "network")
                }

                Section {
                    Toggle(isOn: $enableHaptics) { Label("Haptic Feedback", systemImage: "hand.tap") }
                    Toggle(isOn: $enableNotifications) { Label("Push Notifications", systemImage: "bell.badge") }
                    Toggle(isOn: $showConfidence) { Label("Show Confidence Scores", systemImage: "percent") }
                } header: {
                    Label("Preferences", systemImage: "slider.horizontal.3")
                }

                Section {
                    Button {
                        showStartConfirm = true
                    } label: {
                        HStack {
                            Label("Start Engine", systemImage: "play.fill").foregroundColor(Color(hex: "22C55E"))
                            if isEngineLoading { Spacer(); ProgressView().tint(.white).scaleEffect(0.7) }
                        }
                    }
                    .disabled(isEngineLoading)

                    Button {
                        showPauseConfirm = true
                    } label: {
                        Label("Pause Engine", systemImage: "pause.fill").foregroundColor(Color(hex: "F59E0B"))
                    }
                    .disabled(isEngineLoading)

                    Button {
                        showResumeConfirm = true
                    } label: {
                        Label("Resume Engine", systemImage: "play.circle.fill").foregroundColor(Color(hex: "3B82F6"))
                    }
                    .disabled(isEngineLoading)

                    Button {
                        showStopConfirm = true
                    } label: {
                        Label("Stop Engine", systemImage: "stop.fill").foregroundColor(Color(hex: "EF4444"))
                    }
                    .disabled(isEngineLoading)

                    Button { showKillConfirm = true } label: {
                        Label("Emergency Kill Switch", systemImage: "hand.raised.fill")
                            .foregroundColor(Color(hex: "EF4444")).fontWeight(.bold)
                    }
                    .disabled(isEngineLoading)

                    if let msg = engineMessage {
                        Text(msg).font(.caption).foregroundColor(.gray)
                    }
                } header: {
                    Label("Engine Control", systemImage: "engine.combustion")
                }
                .confirmationDialog("Start Engine", isPresented: $showStartConfirm, titleVisibility: .visible) {
                    Button("Start", role: .none) {
                        Task { await startEngine() }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This will start the trading engine on Bybit Testnet.")
                }
                .confirmationDialog("Pause Engine", isPresented: $showPauseConfirm, titleVisibility: .visible) {
                    Button("Pause", role: .none) {
                        Task { await pauseEngine() }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This will pause order execution while keeping positions open.")
                }
                .confirmationDialog("Resume Engine", isPresented: $showResumeConfirm, titleVisibility: .visible) {
                    Button("Resume", role: .none) {
                        Task { await resumeEngine() }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This will resume order execution.")
                }
                .confirmationDialog("Stop Engine", isPresented: $showStopConfirm, titleVisibility: .visible) {
                    Button("Stop", role: .destructive) {
                        Task { await stopEngine() }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This will stop the trading engine gracefully.")
                }
                .confirmationDialog("Emergency Kill Switch", isPresented: $showKillConfirm, titleVisibility: .visible) {
                    Button("Kill Engine", role: .destructive) {
                        Task { await killEngine() }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This will close all positions, cancel all orders, and stop the engine.")
                }

                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Max Risk per Trade: \(String(format: "%.1f", maxRiskPercent))%")
                            .font(.subheadline)
                        Slider(value: $maxRiskPercent, in: 0.5...10, step: 0.5)
                            .tint(Color(hex: "00D4FF"))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Max Leverage: \(String(format: "%.0f", maxLeverage))x")
                            .font(.subheadline)
                        Slider(value: $maxLeverage, in: 1...20, step: 1)
                            .tint(Color(hex: "00D4FF"))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Min Risk/Reward: 1:\(String(format: "%.1f", minRiskReward))")
                            .font(.subheadline)
                        Slider(value: $minRiskReward, in: 1...5, step: 0.5)
                            .tint(Color(hex: "00D4FF"))
                    }
                } header: {
                    Label("Trading Parameters", systemImage: "chart.bar.xaxis")
                }

                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Analysis Interval: \(Int(analysisInterval))s")
                            .font(.subheadline)
                        Slider(value: $analysisInterval, in: 10...120, step: 5)
                            .tint(Color(hex: "00D4FF"))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Confidence Threshold: \(Int(confidenceThreshold * 100))%")
                            .font(.subheadline)
                        Slider(value: $confidenceThreshold, in: 0.3...0.95, step: 0.05)
                            .tint(Color(hex: "00D4FF"))
                    }
                } header: {
                    Label("Brain Settings", systemImage: "brain")
                }

                Section {
                    HStack { Text("Version"); Spacer(); Text("2.0.0").foregroundColor(.gray) }
                    HStack { Text("Build"); Spacer(); Text("2026.02").foregroundColor(.gray) }
                    HStack { Text("Target"); Spacer(); Text("Bybit Testnet").foregroundColor(Color(hex: "00D4FF")) }
                    HStack {
                        Text("Engine")
                        Spacer()
                        Text(viewModel.engineState)
                            .foregroundColor(viewModel.engineState == "Running" ? Color(hex: "22C55E") : .gray)
                    }
                    HStack {
                        Text("Uptime")
                        Spacer()
                        Text(viewModel.formattedUptime)
                            .foregroundColor(.gray)
                    }
                } header: {
                    Label("About", systemImage: "info.circle")
                }
            }
            .scrollContentBackground(.hidden)
            .background(Color(hex: "0A0A0F"))
            .navigationTitle("Settings")
            .onAppear {
                viewModel.configure(apiClient: container.apiClient)
                Task { await viewModel.refresh() }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") { saveSettings() }
                        .foregroundColor(Color(hex: "00D4FF"))
                }
            }
            .overlay(alignment: .top) {
                if showSaved {
                    Text("Settings Saved").font(.headline).foregroundColor(.white)
                        .padding().background(Color(hex: "22C55E").cornerRadius(12))
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .animation(.spring(response: 0.3), value: showSaved)
        }
    }

    private func saveSettings() {
        // Reconfigure DIContainer with new URLs
        container.reconfigure(apiBaseURL: apiURL, wsBaseURL: wsURL)
        Haptics.success()
        showSaved = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { showSaved = false }
    }

    private func startEngine() async {
        isEngineLoading = true
        defer { isEngineLoading = false }
        do {
            struct EngineResponse: Codable { let status: String?; let message: String? }
            let result: EngineResponse = try await container.apiClient.post(
                "/api/v1/engine/start",
                body: ["confirm_phrase": "I_UNDERSTAND_LIVE_TRADING"]
            )
            engineMessage = result.message ?? "Engine started"
            Haptics.success()
        } catch {
            engineMessage = "Failed: \(error.localizedDescription)"
        }
    }

    private func stopEngine() async {
        isEngineLoading = true
        defer { isEngineLoading = false }
        do {
            struct EngineResponse: Codable { let status: String?; let message: String? }
            let result: EngineResponse = try await container.apiClient.post(
                "/api/v1/engine/stop", body: [:]
            )
            engineMessage = result.message ?? "Engine stopped"
            Haptics.success()
        } catch {
            engineMessage = "Failed: \(error.localizedDescription)"
        }
    }

    private func pauseEngine() async {
        isEngineLoading = true
        defer { isEngineLoading = false }
        do {
            struct EngineResponse: Codable { let status: String?; let message: String? }
            let result: EngineResponse = try await container.apiClient.post(
                "/api/v1/engine/pause", body: [:]
            )
            engineMessage = result.message ?? "Engine paused"
            Haptics.success()
        } catch {
            engineMessage = "Failed: \(error.localizedDescription)"
        }
    }

    private func resumeEngine() async {
        isEngineLoading = true
        defer { isEngineLoading = false }
        do {
            struct EngineResponse: Codable { let status: String?; let message: String? }
            let result: EngineResponse = try await container.apiClient.post(
                "/api/v1/engine/resume", body: [:]
            )
            engineMessage = result.message ?? "Engine resumed"
            Haptics.success()
        } catch {
            engineMessage = "Failed: \(error.localizedDescription)"
        }
    }

    private func killEngine() async {
        isEngineLoading = true
        defer { isEngineLoading = false }
        do {
            struct EngineResponse: Codable { let status: String?; let message: String? }
            let result: EngineResponse = try await container.apiClient.post(
                "/api/v1/engine/kill", body: ["reason": "ios_emergency_kill"]
            )
            engineMessage = result.message ?? "Engine killed"
            Haptics.success()
        } catch {
            engineMessage = "Failed: \(error.localizedDescription)"
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}
