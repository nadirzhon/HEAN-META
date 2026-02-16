import SwiftUI

struct TradesListView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @State private var selectedFilter: TradeFilter = .all
    @State private var searchText = ""
    @State private var selectedTrade: Trade?
    
    enum TradeFilter: String, CaseIterable, Identifiable {
        var id: String { self.rawValue }
        case all = "All"
        case open = "Open"
        case closed = "Closed"
        case profitable = "Profitable"
        case losses = "Losses"
    }
    
    var filteredTrades: [Trade] {
        var trades = dataManager.trades
        if selectedFilter != .all {
            trades = trades.filter {
                switch selectedFilter {
                case .open: return $0.status == .open
                case .closed: return $0.status == .closed
                case .profitable: return ($0.profitLoss ?? 0) > 0
                case .losses: return ($0.profitLoss ?? 0) < 0
                default: return true
                }
            }
        }
        if !searchText.isEmpty {
            trades = trades.filter { $0.symbol.localizedCaseInsensitiveContains(searchText) }
        }
        return trades
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Search and Filter
            VStack(spacing: 16) {
                TextField("Search by symbol...", text: $searchText)
                    .padding(10)
                    .background(Theme.cardBackground)
                    .cornerRadius(10)
                
                Picker("Filter", selection: $selectedFilter) {
                    ForEach(TradeFilter.allCases) { filter in
                        Text(filter.rawValue).tag(filter)
                    }
                }
                .pickerStyle(.segmented)
                
                TradesSummaryCard(trades: filteredTrades)
            }
            .padding()
            .background(Theme.background)
            
            // Trades List
            List(filteredTrades) { trade in
                TradeRow(trade: trade)
                    .onTapGesture { selectedTrade = trade }
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowBackground(Theme.background)
            }
            .listStyle(.plain)
            .background(Theme.background)
        }
        .sheet(item: $selectedTrade) { trade in
            TradeDetailSheet(trade: trade)
        }
        .navigationTitle("Trades")
    }
}

// MARK: - Trades Summary Card
struct TradesSummaryCard: View {
    let trades: [Trade]
    
    private var openCount: Int { trades.filter { $0.status == .open }.count }
    private var profitableCount: Int { trades.filter { ($0.profitLoss ?? 0) > 0 }.count }
    
    var body: some View {
        HStack {
            SummaryItem(icon: "chart.bar.fill", label: "Total", value: "\(trades.count)", color: Theme.accent)
            Divider().frame(height: 30)
            SummaryItem(icon: "clock.fill", label: "Open", value: "\(openCount)", color: .orange)
            Divider().frame(height: 30)
            SummaryItem(icon: "checkmark.circle.fill", label: "Profitable", value: "\(profitableCount)", color: Theme.positive)
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(16)
    }
}

struct SummaryItem: View {
    let icon: String, label: String, value: String, color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon).font(.subheadline).foregroundStyle(color)
            Text(value).font(.headline.weight(.bold)).foregroundStyle(Theme.text)
            Text(label).font(.caption2).foregroundStyle(Theme.secondaryText)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Trade Row (replaces TradeCard)
struct TradeRow: View {
    let trade: Trade
    
    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(trade.symbol).font(.headline.weight(.bold)).foregroundStyle(Theme.text)
                Text(trade.entryDate, style: .date).font(.caption).foregroundStyle(Theme.secondaryText)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(trade.type.rawValue)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(trade.type == .long ? Theme.positive : Theme.negative)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background((trade.type == .long ? Theme.positive : Theme.negative).opacity(0.1))
                    .cornerRadius(8)
                
                if let profitLoss = trade.profitLoss {
                    Text(String(format: "%@$%.2f", profitLoss >= 0 ? "+" : "-", abs(profitLoss)))
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(profitLoss >= 0 ? Theme.positive : Theme.negative)
                } else {
                    Text("OPEN").font(.caption.weight(.bold)).foregroundStyle(.orange)
                }
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }
}

// MARK: - Trade Detail Sheet
struct TradeDetailSheet: View {
    let trade: Trade
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    TradeHeaderCard(trade: trade)
                    TradePriceCard(trade: trade)
                    TradeQuantityCard(trade: trade)
                    TradeDatesCard(trade: trade)
                    if trade.status == .closed {
                        TradePerformanceCard(trade: trade)
                    }
                }
                .padding()
            }
            .background(Theme.background)
            .navigationTitle("Trade Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() }.tint(Theme.accent) }
            }
        }
    }
}

struct TradeHeaderCard: View {
    let trade: Trade
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(trade.symbol).font(.largeTitle.bold()).foregroundStyle(Theme.text)
                Text(trade.type.rawValue).font(.headline).foregroundStyle(trade.type == .long ? Theme.positive : Theme.negative)
            }
            Spacer()
            StatusBadge(status: trade.status)
        }
        .card()
    }
}

struct StatusBadge: View {
    let status: Trade.TradeStatus
    var statusColor: Color { status == .open ? .orange : Theme.accent }
    
    var body: some View {
        Text(status.rawValue.uppercased())
            .font(.caption.weight(.bold))
            .foregroundStyle(statusColor)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(statusColor.opacity(0.1))
            .cornerRadius(8)
    }
}

struct TradePriceCard: View {
    let trade: Trade
    var body: some View {
        HStack {
            PriceDetail(label: "Entry Price", price: trade.entryPrice, color: Theme.text)
            if let exitPrice = trade.exitPrice {
                Spacer()
                Image(systemName: "arrow.right").foregroundStyle(Theme.secondaryText)
                Spacer()
                PriceDetail(label: "Exit Price", price: exitPrice, color: Theme.text)
            }
        }
        .card()
    }
}

struct PriceDetail: View {
    let label: String, price: Double, color: Color
    var body: some View {
        VStack(alignment: .center, spacing: 4) {
            Text(label).font(.caption).foregroundStyle(Theme.secondaryText)
            Text("$\(String(format: "%.2f", price))").font(.title2.weight(.semibold)).foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
    }
}

struct TradeQuantityCard: View {
    let trade: Trade
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Quantity").font(.caption).foregroundStyle(Theme.secondaryText)
                Text("\(trade.quantity) shares").font(.headline.weight(.semibold)).foregroundStyle(Theme.text)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text("Position Value").font(.caption).foregroundStyle(Theme.secondaryText)
                Text("$\(String(format: "%.2f", Double(trade.quantity) * trade.entryPrice))").font(.headline.weight(.semibold)).foregroundStyle(Theme.text)
            }
        }
        .card()
    }
}

struct TradeDatesCard: View {
    let trade: Trade
    var holdingPeriod: String? {
        if let exitDate = trade.exitDate {
            let components = Calendar.current.dateComponents([.day, .hour], from: trade.entryDate, to: exitDate)
            return "\(components.day ?? 0)d \(components.hour ?? 0)h"
        }
        return nil
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            DateRow(label: "Entry Date", date: trade.entryDate)
            if let exitDate = trade.exitDate {
                Divider()
                DateRow(label: "Exit Date", date: exitDate)
                Divider()
                HStack {
                    Text("Holding Period").font(.subheadline).foregroundStyle(Theme.secondaryText)
                    Spacer()
                    Text(holdingPeriod ?? "").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.text)
                }
            }
        }
        .card()
    }
}

struct DateRow: View {
    let label: String, date: Date
    var body: some View {
        HStack {
            Text(label).font(.subheadline).foregroundStyle(Theme.secondaryText)
            Spacer()
            Text(date, style: .datetime).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.text)
        }
    }
}

struct TradePerformanceCard: View {
    let trade: Trade
    var percentReturn: Double {
        guard let exitPrice = trade.exitPrice else { return 0 }
        return ((exitPrice - trade.entryPrice) / trade.entryPrice) * 100
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Performance").font(.headline).foregroundStyle(Theme.text)
            HStack(spacing: 20) {
                VStack(spacing: 4) {
                    Text("Profit/Loss").font(.caption).foregroundStyle(Theme.secondaryText)
                    if let profitLoss = trade.profitLoss {
                        Text(String(format: "%@$%.2f", profitLoss >= 0 ? "+" : "-", abs(profitLoss)))
                            .font(.title2.weight(.bold))
                            .foregroundStyle(profitLoss >= 0 ? Theme.positive : Theme.negative)
                    }
                }.frame(maxWidth: .infinity)
                VStack(spacing: 4) {
                    Text("Return %").font(.caption).foregroundStyle(Theme.secondaryText)
                    Text(String(format: "%@%.2f%%", percentReturn >= 0 ? "+" : "", percentReturn))
                        .font(.title2.weight(.bold))
                        .foregroundStyle(percentReturn >= 0 ? Theme.positive : Theme.negative)
                }.frame(maxWidth: .infinity)
            }
        }
        .card()
    }
}

#Preview {
    NavigationView {
        TradesListView()
            .environmentObject(TradingDataManager.preview)
    }
}
