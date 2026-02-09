//
//  MindViewModel.swift
//  HEAN
//
//  ViewModel for Mind (AI thought feed)
//

import SwiftUI
import Combine

@MainActor
final class MindViewModel: ObservableObject {
    @Published var analysis: BrainAnalysis?
    @Published var thoughtHistory: [BrainThought] = []
    @Published var isLoading = true
    @Published var error: String?

    private var apiClient: APIClient?
    private var isConfigured = false

    func configure(apiClient: APIClient) {
        guard !isConfigured else { return }
        isConfigured = true
        self.apiClient = apiClient
    }

    func refresh() async {
        guard let apiClient = apiClient else { return }

        do {
            let result: BrainAnalysis = try await apiClient.get("/api/v1/brain/analysis")
            self.analysis = result
        } catch {
            if self.analysis == nil {
                self.analysis = BrainAnalysis(
                    timestamp: ISO8601DateFormatter().string(from: Date()),
                    thoughts: [],
                    forces: [
                        BrainForce(id: "1", name: "Temperature", value: 0.5, label: "Warming"),
                        BrainForce(id: "2", name: "Entropy", value: -0.3, label: "Compressing"),
                    ],
                    signal: nil,
                    summary: "Analyzing market conditions...",
                    marketRegime: "unknown"
                )
            }
        }

        do {
            let thoughts: [BrainThought] = try await apiClient.get("/api/v1/brain/thoughts?limit=50")
            self.thoughtHistory = thoughts
        } catch {
            if self.thoughtHistory.isEmpty {
                self.thoughtHistory = [
                    BrainThought(id: "1", timestamp: ISO8601DateFormatter().string(from: Date()),
                                stage: "physics", content: "Waiting for market data...", confidence: nil),
                ]
            }
        }

        isLoading = false
    }
}
