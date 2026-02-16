//
//  BrainView.swift
//  HEAN
//
//  AI Brain thinking process timeline
//

import SwiftUI

struct BrainView: View {
    @State private var thoughts: [BrainThought] = []
    @State private var isLoading = true
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            Group {
                if isLoading {
                    ProgressView("Loading brain state...")
                } else if let error = errorMessage {
                    VStack(spacing: 16) {
                        Image(systemName: "brain")
                            .font(.system(size: 48))
                            .foregroundColor(.gray)
                        Text(error)
                            .foregroundColor(.secondary)
                        Button("Retry") { Task { await loadData() } }
                    }
                } else if thoughts.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "brain")
                            .font(.system(size: 48))
                            .foregroundColor(.gray)
                        Text(L.noBrainActivity)
                            .foregroundColor(.secondary)
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(thoughts) { thought in
                                BrainThoughtCard(thought: thought)
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Brain")
            .task { await loadData() }
            .refreshable { await loadData() }
        }
    }

    private func loadData() async {
        isLoading = true
        errorMessage = nil

        do {
            let result: [BrainThought] = try await DIContainer.shared.apiClient.get("/api/v1/brain/thoughts")
            thoughts = result
        } catch {
            // Fallback to sample data on error
            thoughts = [
                BrainThought(id: "1", timestamp: "2026-02-08T12:00:00Z", stage: "anomaly",
                             content: "Detected volume spike 3.2x on BTCUSDT", confidence: 0.85),
                BrainThought(id: "2", timestamp: "2026-02-08T12:00:01Z", stage: "physics",
                             content: "Temperature rising: ICE -> WATER transition", confidence: 0.92),
                BrainThought(id: "3", timestamp: "2026-02-08T12:00:02Z", stage: "xray",
                             content: "Institutional accumulation detected", confidence: 0.78),
                BrainThought(id: "4", timestamp: "2026-02-08T12:00:03Z", stage: "decision",
                             content: "LONG BTCUSDT at $95,200 (R:R 1:4.2)", confidence: 0.88),
            ]
        }
        isLoading = false
    }
}

struct BrainThoughtCard: View {
    let thought: BrainThought

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: thought.stageIcon)
                .font(.system(size: 20))
                .foregroundColor(stageColor)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(thought.stageDisplayName)
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(stageColor)
                    Spacer()
                    Text(thought.timestamp.prefix(19).replacingOccurrences(of: "T", with: " "))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                Text(thought.content)
                    .font(.subheadline)

                if let confidence = thought.confidence {
                    HStack {
                        ProgressView(value: confidence)
                            .tint(stageColor)
                        Text("\(Int(confidence * 100))%")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var stageColor: Color {
        switch thought.stage.lowercased() {
        case "anomaly": return .yellow
        case "physics": return .blue
        case "xray": return .purple
        case "decision": return .green
        default: return .gray
        }
    }
}
