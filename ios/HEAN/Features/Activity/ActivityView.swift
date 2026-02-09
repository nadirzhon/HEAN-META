//
//  ActivityView.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import SwiftUI

struct ActivityView: View {
    @EnvironmentObject var container: DIContainer
    @State private var filterType: EventFilterType = .all
    @State private var events: [TradingEvent] = []

    enum EventFilterType: String, CaseIterable {
        case all = "All"
        case trades = "Trades"
        case signals = "Signals"
        case errors = "Errors"
    }

    var filteredEvents: [TradingEvent] {
        switch filterType {
        case .all:
            return events
        case .trades:
            return events.filter { 
                $0.type == .orderPlaced || 
                $0.type == .orderFilled || 
                $0.type == .orderCancelled ||
                $0.type == .positionOpened || 
                $0.type == .positionClosed
            }
        case .signals:
            return events.filter { $0.type == .signal }
        case .errors:
            return events.filter { $0.type == .error }
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Stats header
                EventStatsHeader(events: events)

                // Filter picker
                Picker("Filter", selection: $filterType) {
                    ForEach(EventFilterType.allCases, id: \.self) { type in
                        Text(type.rawValue).tag(type)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                // Events list
                ScrollView {
                    LazyVStack(spacing: Theme.Spacing.sm) {
                        ForEach(filteredEvents) { event in
                            ActivityEventRow(event: event)
                        }
                    }
                    .padding()
                }
            }
            .background(Theme.Colors.background)
            .navigationTitle("Activity")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        events.removeAll()
                    } label: {
                        Image(systemName: "trash")
                    }
                }
            }
            .task {
                await loadRecentEvents()
                subscribeToEventStream()
            }
        }
    }
    
    private func loadRecentEvents() async {
        do {
            events = try await container.eventService.fetchRecentEvents(limit: 100)
        } catch {
            print("Failed to load recent events: \(error)")
        }
    }
    
    private func subscribeToEventStream() {
        Task {
            for await event in container.eventService.eventStream.values {
                events.insert(event, at: 0)
                
                // Keep only the most recent 100 events
                if events.count > 100 {
                    events.removeLast()
                }
            }
        }
    }
}

struct EventStatsHeader: View {
    @EnvironmentObject var container: DIContainer
    let events: [TradingEvent]

    var eventsPerSecond: Double {
        // Calculate based on events in the last 60 seconds
        let cutoff = Date().addingTimeInterval(-60)
        let recentEvents = events.filter { $0.timestamp > cutoff }
        return Double(recentEvents.count) / 60.0
    }

    var lastEventAge: String {
        guard let lastEvent = events.first else {
            return "N/A"
        }

        let interval = Date().timeIntervalSince(lastEvent.timestamp)
        if interval < 1 {
            return "Just now"
        } else if interval < 60 {
            return "\(Int(interval))s ago"
        } else {
            return "\(Int(interval / 60))m ago"
        }
    }

    var body: some View {
        GlassCard {
            HStack(spacing: Theme.Spacing.xl) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Events/sec")
                        .font(Theme.Typography.caption2)
                        .foregroundColor(Theme.Colors.textTertiary)
                    Text(String(format: "%.1f", eventsPerSecond))
                        .font(Theme.Typography.monoLarge)
                        .foregroundColor(Theme.Colors.accent)
                }

                Divider()
                    .frame(height: 40)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Last Event")
                        .font(Theme.Typography.caption2)
                        .foregroundColor(Theme.Colors.textTertiary)
                    Text(lastEventAge)
                        .font(Theme.Typography.body)
                        .foregroundColor(Theme.Colors.textSecondary)
                }

                Spacer()
            }
        }
        .padding()
    }
}

struct ActivityEventRow: View {
    let event: TradingEvent

    var eventColor: Color {
        switch event.type {
        case .error:
            return Theme.Colors.error
        case .signal:
            return Theme.Colors.accent
        case .orderPlaced, .orderFilled, .orderCancelled, .positionOpened, .positionClosed:
            return Theme.Colors.success
        case .riskAlert:
            return Theme.Colors.warning
        case .systemInfo:
            return Theme.Colors.textSecondary
        }
    }

    var eventIcon: String {
        switch event.type {
        case .error:
            return "exclamationmark.triangle.fill"
        case .signal:
            return "waveform.path.ecg"
        case .orderPlaced, .orderFilled, .orderCancelled:
            return "doc.text.fill"
        case .positionOpened, .positionClosed:
            return "chart.line.uptrend.xyaxis"
        case .riskAlert:
            return "exclamationmark.shield.fill"
        case .systemInfo:
            return "info.circle.fill"
        }
    }

    var body: some View {
        HStack(spacing: Theme.Spacing.md) {
            // Icon
            Image(systemName: eventIcon)
                .font(.system(size: 16))
                .foregroundColor(eventColor)
                .frame(width: 24, height: 24)
                .background(
                    Circle()
                        .fill(eventColor.opacity(0.15))
                )

            // Content
            VStack(alignment: .leading, spacing: 4) {
                Text(event.type.displayName)
                    .font(Theme.Typography.caption1)
                    .fontWeight(.semibold)
                    .foregroundColor(Theme.Colors.textPrimary)

                Text(event.message)
                    .font(Theme.Typography.caption2)
                    .foregroundColor(Theme.Colors.textSecondary)
                    .lineLimit(2)
            }

            Spacer()

            // Timestamp
            Text(formatTime(event.timestamp))
                .font(Theme.Typography.caption2)
                .foregroundColor(Theme.Colors.textTertiary)
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, Theme.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: Theme.CornerRadius.sm)
                .fill(Theme.Colors.card)
        )
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

#Preview {
    ActivityView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}
