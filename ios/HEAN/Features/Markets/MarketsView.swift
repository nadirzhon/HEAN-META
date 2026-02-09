//
//  MarketsView.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import SwiftUI

struct MarketsView: View {
    @EnvironmentObject var container: DIContainer
    @State private var searchText = ""
    @State private var sortBy: SortOption = .name
    @State private var markets: [Market] = []

    enum SortOption: String, CaseIterable {
        case name = "Name"
        case price = "Price"
        case change = "Change"
        case volume = "Volume"
    }

    var filteredMarkets: [Market] {
        var result = markets

        if !searchText.isEmpty {
            result = result.filter { $0.symbol.localizedCaseInsensitiveContains(searchText) }
        }

        switch sortBy {
        case .name:
            result.sort { $0.symbol < $1.symbol }
        case .price:
            result.sort { $0.lastPrice > $1.lastPrice }
        case .change:
            result.sort { $0.priceChange24h > $1.priceChange24h }
        case .volume:
            result.sort { $0.volume24h > $1.volume24h }
        }

        return result
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Sort picker
                Picker("Sort By", selection: $sortBy) {
                    ForEach(SortOption.allCases, id: \.self) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                // Markets list
                ScrollView {
                    LazyVStack(spacing: Theme.Spacing.md) {
                        ForEach(filteredMarkets) { market in
                            MarketRow(market: market)
                                .onTapGesture {
                                    Haptics.light()
                                    // Navigate to trade view
                                }
                        }
                    }
                    .padding()
                }
            }
            .background(Theme.Colors.background)
            .navigationTitle("Markets")
            .searchable(text: $searchText, prompt: "Search markets")
            .task {
                // Load initial markets
                markets = container.marketService.markets
            }
            .onReceive(container.marketService.marketUpdates) { updatedMarkets in
                markets = updatedMarkets
            }
        }
    }
}

struct MarketRow: View {
    let market: Market

    var isPositiveChange: Bool {
        market.priceChange24h >= 0
    }

    var body: some View {
        GlassCard {
            HStack(spacing: Theme.Spacing.md) {
                // Symbol and name
                VStack(alignment: .leading, spacing: 4) {
                    Text(market.symbol)
                        .font(Theme.Typography.headline)
                        .foregroundColor(Theme.Colors.textPrimary)

                    if let name = market.name {
                        Text(name)
                            .font(Theme.Typography.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                    }
                }

                Spacer()

                // Mini sparkline
                if let sparklineData = market.sparkline, !sparklineData.isEmpty {
                    Sparkline(
                        data: sparklineData,
                        color: isPositiveChange ? Theme.Colors.success : Theme.Colors.error,
                        lineWidth: 1.5,
                        showFill: true
                    )
                    .frame(width: 60, height: 30)
                }

                // Price and change
                VStack(alignment: .trailing, spacing: 4) {
                    Text("$\(String(format: "%.2f", market.lastPrice))")
                        .font(Theme.Typography.mono)
                        .foregroundColor(Theme.Colors.textPrimary)

                    HStack(spacing: 4) {
                        Image(systemName: isPositiveChange ? "arrow.up.right" : "arrow.down.right")
                            .font(.system(size: 10, weight: .bold))

                        Text(String(format: "%.2f%%", abs(market.priceChange24h)))
                            .font(Theme.Typography.caption2)
                    }
                    .foregroundColor(isPositiveChange ? Theme.Colors.success : Theme.Colors.error)
                }
            }
        }
    }
}

#Preview {
    MarketsView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}
