//
//  HEANWidget.swift
//  HEAN
//
//  Home screen widgets for quick portfolio and PnL glance
//

import SwiftUI
import WidgetKit

// MARK: - Widget Timeline Entry

struct HEANWidgetEntry: TimelineEntry {
    let date: Date
    let equity: Double
    let pnl: Double
    let pnlPercent: Double
    let riskState: String
    let activePositions: Int
    let isPlaceholder: Bool

    static let placeholder = HEANWidgetEntry(
        date: Date(),
        equity: 12456.78,
        pnl: 234.56,
        pnlPercent: 2.4,
        riskState: "NORMAL",
        activePositions: 3,
        isPlaceholder: true
    )
}

// MARK: - Timeline Provider

struct HEANWidgetProvider: TimelineProvider {
    func placeholder(in context: Context) -> HEANWidgetEntry {
        .placeholder
    }

    func getSnapshot(in context: Context, completion: @escaping (HEANWidgetEntry) -> Void) {
        completion(.placeholder)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<HEANWidgetEntry>) -> Void) {
        // Fetch real data from HEAN API
        let baseURL = UserDefaultsStore.shared.apiBaseURL
        guard let url = URL(string: "\(baseURL)/api/v1/engine/status") else {
            let timeline = Timeline(entries: [HEANWidgetEntry.placeholder], policy: .after(Date().addingTimeInterval(300)))
            completion(timeline)
            return
        }

        URLSession.shared.dataTask(with: url) { data, _, error in
            guard let data = data, error == nil else {
                let timeline = Timeline(entries: [HEANWidgetEntry.placeholder], policy: .after(Date().addingTimeInterval(60)))
                completion(timeline)
                return
            }

            let entry: HEANWidgetEntry
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                entry = HEANWidgetEntry(
                    date: Date(),
                    equity: json["equity"] as? Double ?? 0,
                    pnl: json["total_pnl"] as? Double ?? 0,
                    pnlPercent: json["return_pct"] as? Double ?? 0,
                    riskState: json["risk_state"] as? String ?? "UNKNOWN",
                    activePositions: json["active_positions"] as? Int ?? 0,
                    isPlaceholder: false
                )
            } else {
                entry = .placeholder
            }

            let timeline = Timeline(entries: [entry], policy: .after(Date().addingTimeInterval(300)))
            completion(timeline)
        }.resume()
    }
}

// MARK: - Small Widget View

struct HEANWidgetSmallView: View {
    let entry: HEANWidgetEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("HEAN")
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundColor(Color(hex: "00D4FF"))
                Spacer()
                Circle()
                    .fill(entry.riskState == "NORMAL" ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                    .frame(width: 6, height: 6)
            }

            Text(formatCurrency(entry.equity))
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .minimumScaleFactor(0.7)

            HStack(spacing: 4) {
                Image(systemName: entry.pnl >= 0 ? "arrow.up.right" : "arrow.down.right")
                    .font(.system(size: 10))
                Text(formatPnL(entry.pnl))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
            }
            .foregroundColor(entry.pnl >= 0 ? Color(hex: "22C55E") : Color(hex: "EF4444"))

            Spacer()

            HStack {
                Label("\(entry.activePositions)", systemImage: "chart.bar.doc.horizontal")
                    .font(.system(size: 10))
                    .foregroundColor(.gray)
                Spacer()
                Text(entry.date.formatted(.dateTime.hour().minute()))
                    .font(.system(size: 9))
                    .foregroundColor(.gray.opacity(0.7))
            }
        }
        .padding(12)
        .background(Color(hex: "0A0A0F"))
        .if(entry.isPlaceholder) { view in
            view.redacted(reason: .placeholder)
        }
    }

    private func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }

    private func formatPnL(_ value: Double) -> String {
        let sign = value >= 0 ? "+" : ""
        return "\(sign)$\(String(format: "%.2f", abs(value)))"
    }
}

// MARK: - Medium Widget View

struct HEANWidgetMediumView: View {
    let entry: HEANWidgetEntry

    var body: some View {
        HStack(spacing: 16) {
            // Left: Portfolio info
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("HEAN")
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundColor(Color(hex: "00D4FF"))
                    Text(entry.riskState)
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(entry.riskState == "NORMAL" ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background((entry.riskState == "NORMAL" ? Color(hex: "22C55E") : Color(hex: "EF4444")).opacity(0.15))
                        .cornerRadius(4)
                }

                Text(formatCurrency(entry.equity))
                    .font(.system(size: 26, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                HStack(spacing: 4) {
                    Image(systemName: entry.pnl >= 0 ? "arrow.up.right" : "arrow.down.right")
                    Text("\(entry.pnl >= 0 ? "+" : "")$\(String(format: "%.2f", abs(entry.pnl))) (\(String(format: "%.2f", entry.pnlPercent))%)")
                }
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(entry.pnl >= 0 ? Color(hex: "22C55E") : Color(hex: "EF4444"))
            }

            Spacer()

            // Right: Quick stats
            VStack(alignment: .trailing, spacing: 12) {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Positions")
                        .font(.system(size: 10))
                        .foregroundColor(.gray)
                    Text("\(entry.activePositions)")
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                }

                VStack(alignment: .trailing, spacing: 2) {
                    Text("Updated")
                        .font(.system(size: 10))
                        .foregroundColor(.gray)
                    Text(entry.date.formatted(.dateTime.hour().minute()))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.gray)
                }
            }
        }
        .padding(16)
        .background(Color(hex: "0A0A0F"))
        .if(entry.isPlaceholder) { view in
            view.redacted(reason: .placeholder)
        }
    }

    private func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }
}

// MARK: - Widget Configuration

struct HEANPortfolioWidget: Widget {
    let kind = "HEANPortfolioWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: HEANWidgetProvider()) { entry in
            if #available(iOS 17.0, *) {
                HEANWidgetSmallView(entry: entry)
                    .containerBackground(Color(hex: "0A0A0F"), for: .widget)
            } else {
                HEANWidgetSmallView(entry: entry)
            }
        }
        .configurationDisplayName("HEAN Portfolio")
        .description("Track your portfolio equity and P&L")
        .supportedFamilies([.systemSmall, .systemMedium])
    }
}
