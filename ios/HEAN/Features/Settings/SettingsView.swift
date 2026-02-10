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
    @AppStorage("appLanguage") private var appLanguage: String = "en"
    @AppStorage("apiBaseURL") private var apiURL: String = "http://localhost:8000"
    @AppStorage("wsBaseURL") private var wsURL: String = "ws://localhost:8000/ws"
    @State private var authKey: String = KeychainStore.shared.get("api_auth_key") ?? ""
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
                    Picker(L.language, selection: $appLanguage) {
                        Text("English").tag("en")
                        Text("Русский").tag("ru")
                    }
                } header: {
                    Label(L.language, systemImage: "globe")
                }

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
                    Label(L.connection, systemImage: "network")
                }

                Section {
                    Toggle(isOn: $enableHaptics) { Label(L.hapticFeedback, systemImage: "hand.tap") }
                    Toggle(isOn: $enableNotifications) { Label(L.pushNotifications, systemImage: "bell.badge") }
                    Toggle(isOn: $showConfidence) { Label(L.showConfidenceScores, systemImage: "percent") }
                } header: {
                    Label(L.preferences, systemImage: "slider.horizontal.3")
                }

                Section {
                    Button {
                        showStartConfirm = true
                    } label: {
                        HStack {
                            Label(L.startEngine, systemImage: "play.fill").foregroundColor(Color(hex: "22C55E"))
                            if isEngineLoading { Spacer(); ProgressView().tint(.white).scaleEffect(0.7) }
                        }
                    }
                    .disabled(isEngineLoading)

                    Button {
                        showPauseConfirm = true
                    } label: {
                        Label(L.pauseEngine, systemImage: "pause.fill").foregroundColor(Color(hex: "F59E0B"))
                    }
                    .disabled(isEngineLoading)

                    Button {
                        showResumeConfirm = true
                    } label: {
                        Label(L.resumeEngine, systemImage: "play.circle.fill").foregroundColor(Color(hex: "3B82F6"))
                    }
                    .disabled(isEngineLoading)

                    Button {
                        showStopConfirm = true
                    } label: {
                        Label(L.stopEngine, systemImage: "stop.fill").foregroundColor(Color(hex: "EF4444"))
                    }
                    .disabled(isEngineLoading)

                    Button { showKillConfirm = true } label: {
                        Label(L.emergencyKill, systemImage: "hand.raised.fill")
                            .foregroundColor(Color(hex: "EF4444")).fontWeight(.bold)
                    }
                    .disabled(isEngineLoading)

                    if let msg = engineMessage {
                        Text(msg).font(.caption).foregroundColor(.gray)
                    }
                } header: {
                    Label(L.engineControl, systemImage: "engine.combustion")
                }
                .confirmationDialog(L.startEngine, isPresented: $showStartConfirm, titleVisibility: .visible) {
                    Button(L.startEngine, role: .none) {
                        Task { await startEngine() }
                    }
                    Button(L.cancel, role: .cancel) {}
                } message: {
                    Text(L.startEngineMsg)
                }
                .confirmationDialog(L.pauseEngine, isPresented: $showPauseConfirm, titleVisibility: .visible) {
                    Button(L.pauseEngine, role: .none) {
                        Task { await pauseEngine() }
                    }
                    Button(L.cancel, role: .cancel) {}
                } message: {
                    Text(L.pauseEngineMsg)
                }
                .confirmationDialog(L.resumeEngine, isPresented: $showResumeConfirm, titleVisibility: .visible) {
                    Button(L.resumeEngine, role: .none) {
                        Task { await resumeEngine() }
                    }
                    Button(L.cancel, role: .cancel) {}
                } message: {
                    Text(L.resumeEngineMsg)
                }
                .confirmationDialog(L.stopEngine, isPresented: $showStopConfirm, titleVisibility: .visible) {
                    Button(L.stopEngine, role: .destructive) {
                        Task { await stopEngine() }
                    }
                    Button(L.cancel, role: .cancel) {}
                } message: {
                    Text(L.stopEngineMsg)
                }
                .confirmationDialog(L.emergencyKill, isPresented: $showKillConfirm, titleVisibility: .visible) {
                    Button(L.emergencyKill, role: .destructive) {
                        Task { await killEngine() }
                    }
                    Button(L.cancel, role: .cancel) {}
                } message: {
                    Text(L.killEngineMsg)
                }

                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(L.maxRiskPerTrade): \(String(format: "%.1f", maxRiskPercent))%")
                            .font(.subheadline)
                        Slider(value: $maxRiskPercent, in: 0.5...10, step: 0.5)
                            .tint(Color(hex: "00D4FF"))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(L.maxLeverage): \(String(format: "%.0f", maxLeverage))x")
                            .font(.subheadline)
                        Slider(value: $maxLeverage, in: 1...20, step: 1)
                            .tint(Color(hex: "00D4FF"))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(L.minRiskReward): 1:\(String(format: "%.1f", minRiskReward))")
                            .font(.subheadline)
                        Slider(value: $minRiskReward, in: 1...5, step: 0.5)
                            .tint(Color(hex: "00D4FF"))
                    }
                } header: {
                    Label(L.tradingParameters, systemImage: "chart.bar.xaxis")
                }

                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(L.analysisInterval): \(Int(analysisInterval))s")
                            .font(.subheadline)
                        Slider(value: $analysisInterval, in: 10...120, step: 5)
                            .tint(Color(hex: "00D4FF"))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(L.confidenceThreshold): \(Int(confidenceThreshold * 100))%")
                            .font(.subheadline)
                        Slider(value: $confidenceThreshold, in: 0.3...0.95, step: 0.05)
                            .tint(Color(hex: "00D4FF"))
                    }
                } header: {
                    Label(L.brainSettings, systemImage: "brain")
                }

                Section {
                    HStack { Text(L.version); Spacer(); Text("2.0.0").foregroundColor(.gray) }
                    HStack { Text(L.build); Spacer(); Text("2026.02").foregroundColor(.gray) }
                    HStack { Text(L.target); Spacer(); Text("Bybit Testnet").foregroundColor(Color(hex: "00D4FF")) }
                    HStack {
                        Text(L.engine)
                        Spacer()
                        Text(viewModel.engineState)
                            .foregroundColor(viewModel.engineState == "Running" ? Color(hex: "22C55E") : .gray)
                    }
                    HStack {
                        Text(L.uptime)
                        Spacer()
                        Text(viewModel.formattedUptime)
                            .foregroundColor(.gray)
                    }
                } header: {
                    Label(L.about, systemImage: "info.circle")
                }
            }
            .scrollContentBackground(.hidden)
            .background(Color(hex: "0A0A0F"))
            .navigationTitle(L.settings)
            .onAppear {
                // One-time migration from UserDefaults to Keychain
                if let legacyKey = UserDefaults.standard.string(forKey: "apiAuthKey"), !legacyKey.isEmpty {
                    KeychainStore.shared.save(legacyKey, for: "api_auth_key")
                    UserDefaults.standard.removeObject(forKey: "apiAuthKey")
                    authKey = legacyKey
                }

                viewModel.configure(apiClient: container.apiClient)
                Task { await viewModel.refresh() }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(L.save) { saveSettings() }
                        .foregroundColor(Color(hex: "00D4FF"))
                }
            }
            .overlay(alignment: .top) {
                if showSaved {
                    Text(L.settingsSaved).font(.headline).foregroundColor(.white)
                        .padding().background(Color(hex: "22C55E").cornerRadius(12))
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .animation(.spring(response: 0.3), value: showSaved)
        }
    }

    private func saveSettings() {
        // Save auth key to Keychain
        if authKey.isEmpty {
            KeychainStore.shared.delete("api_auth_key")
        } else {
            KeychainStore.shared.save(authKey, for: "api_auth_key")
        }

        // Reconfigure DIContainer with new URLs (picks up auth token from Keychain)
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
