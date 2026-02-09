import SwiftUI

struct TradesListView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @State private var selectedFilter: TradeFilter = .all
    @State private var searchText = ""
    @State private var selectedTrade: Trade?
    
    enum TradeFilter: String, CaseIterable {
        case all = "All"
        case open = "Open"
        case closed = "Closed"
        case profitable = "Profitable"
        case losses = "Losses"
    }
    
    var filteredTrades: [Trade] {
        var trades = dataManager.trades
        
        switch selectedFilter {
        case .all:
            break
        case .open:
            trades = trades.filter { $0.status == .open }
        case .closed:
            trades = trades.filter { $0.status == .closed }
        case .profitable:
            trades = trades.filter { ($0.profitLoss) > 0 }
        case .losses:
            trades = trades.filter { ($0.profitLoss) < 0 }
        }
        
        if !searchText.isEmpty {
            trades = trades.filter { $0.symbol.localizedCaseInsensitiveContains(searchText) }
        }
        
        return trades
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Search Bar
            SearchBar(text: $searchText)
                .padding(.horizontal)
            
            // Filter Selector
            FilterSelector(selected: $selectedFilter)
                .padding(.horizontal)
            
            // Trades Summary
            TradesSummaryCard(
                total: filteredTrades.count,
                open: filteredTrades.filter { $0.status == .open }.count,
                profitable: filteredTrades.filter { $0.profitLoss > 0 }.count
            )
            .padding(.horizontal)
            
            // Trades List
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(filteredTrades) { trade in
                        TradeCard(trade: trade)
                            .onTapGesture {
                                withAnimation(.spring(response: 0.3)) {
                                    selectedTrade = trade
                                }
                            }
                    }
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .sheet(item: $selectedTrade) { trade in
            TradeDetailSheet(trade: trade)
        }
    }
}

// MARK: - Search Bar

struct SearchBar: View {
    @Binding var text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.white.opacity(0.6))
            
            TextField("Search by symbol...", text: $text)
                .foregroundStyle(.white)
                .textFieldStyle(.plain)
            
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(12)
        .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 12))
    }
}

// MARK: - Filter Selector

struct FilterSelector: View {
    @Binding var selected: TradesListView.TradeFilter
    @Namespace private var namespace
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(TradesListView.TradeFilter.allCases, id: \.self) { filter in
                    FilterChip(
                        title: filter.rawValue,
                        isSelected: selected == filter,
                        namespace: namespace
                    ) {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            selected = filter
                        }
                    }
                }
            }
            .padding(.horizontal, 4)
        }
    }
}

struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let namespace: Namespace.ID
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(isSelected ? .white : .white.opacity(0.6))
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background {
                    if isSelected {
                        Capsule()
                            .fill(Color.blue)
                            .matchedGeometryEffect(id: "selectedFilter", in: namespace)
                    } else {
                        Capsule()
                            .strokeBorder(Color.white.opacity(0.3), lineWidth: 1)
                    }
                }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Trades Summary Card

struct TradesSummaryCard: View {
    let total: Int
    let open: Int
    let profitable: Int
    
    var body: some View {
        HStack(spacing: 20) {
            SummaryItem(
                icon: "chart.bar.fill",
                label: "Total",
                value: "\(total)",
                color: .blue
            )
            
            Divider()
                .frame(height: 40)
                .background(Color.white.opacity(0.2))
            
            SummaryItem(
                icon: "clock.fill",
                label: "Open",
                value: "\(open)",
                color: .orange
            )
            
            Divider()
                .frame(height: 40)
                .background(Color.white.opacity(0.2))
            
            SummaryItem(
                icon: "checkmark.circle.fill",
                label: "Profitable",
                value: "\(profitable)",
                color: .green
            )
        }
        .padding(16)
        .glassEffect(.regular.tint(.blue.opacity(0.05)).interactive(), in: .rect(cornerRadius: 16))
    }
}

struct SummaryItem: View {
    let icon: String
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(color)
            
            Text(value)
                .font(.title2.weight(.bold))
                .foregroundStyle(.white)
            
            Text(label)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.6))
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Trade Card

struct TradeCard: View {
    let trade: Trade
    
    var body: some View {
        HStack(spacing: 16) {
            // Symbol and Type
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(trade.symbol)
                        .font(.headline.weight(.bold))
                        .foregroundStyle(.white)
                    
                    Text(trade.type.rawValue)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(trade.type == .long ? .green : .red)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill((trade.type == .long ? Color.green : Color.red).opacity(0.2))
                        )
                }
                
                Text(trade.entryDate.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.6))
            }
            
            Spacer()
            
            // Status and P&L
            VStack(alignment: .trailing, spacing: 4) {
                if trade.status == .open {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(.green)
                            .frame(width: 8, height: 8)
                        
                        Text("OPEN")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.green)
                    }
                } else if let profitLoss = trade.profit {
                    Text(profitLoss >= 0 ? "+$\(String(format: "%.2f", profitLoss))" : "-$\(String(format: "%.2f", abs(profitLoss)))")
                        .font(.headline.weight(.bold))
                        .foregroundStyle(profitLoss >= 0 ? .green : .red)
                    
                    let percentChange = ((trade.exitPrice ?? trade.entryPrice) - trade.entryPrice) / trade.entryPrice * 100
                    Text("\(percentChange >= 0 ? "+" : "")\(String(format: "%.2f%%", percentChange))")
                        .font(.caption)
                        .foregroundStyle((profitLoss >= 0 ? Color.green : Color.red).opacity(0.8))
                }
            }
            
            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.4))
        }
        .padding(16)
        .glassEffect(.regular.tint((trade.profitLoss >= 0 ? Color.green : Color.red).opacity(0.03)).interactive(), in: .rect(cornerRadius: 16))
    }
}

// MARK: - Trade Detail Sheet

struct TradeDetailSheet: View {
    let trade: Trade
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            ZStack {
                AnimatedGradientBackground()
                    .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 20) {
                        // Header
                        TradeHeaderCard(trade: trade)
                        
                        // Price Details
                        TradePriceCard(trade: trade)
                        
                        // Quantity and Value
                        TradeQuantityCard(trade: trade)
                        
                        // Dates
                        TradeDatesCard(trade: trade)
                        
                        // Performance Metrics
                        if trade.status == .closed {
                            TradePerformanceCard(trade: trade)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Trade Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundStyle(.white)
                }
            }
        }
    }
}

struct TradeHeaderCard: View {
    let trade: Trade
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(trade.symbol)
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    
                    Text(trade.type.rawValue)
                        .font(.headline)
                        .foregroundStyle(trade.type == .long ? .green : .red)
                }
                
                Spacer()
                
                StatusBadge(status: trade.status)
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.blue.opacity(0.1)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct StatusBadge: View {
    let status: Trade.TradeStatus
    
    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(status == .open ? Color.green : Color.blue)
                .frame(width: 10, height: 10)
            
            Text(status.rawValue.uppercased())
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            Capsule()
                .fill((status == .open ? Color.green : Color.blue).opacity(0.2))
        )
    }
}

struct TradePriceCard: View {
    let trade: Trade
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                PriceDetail(label: "Entry Price", price: trade.entryPrice, color: .blue)
                
                if let exitPrice = trade.exitPrice {
                    Spacer()
                    Image(systemName: "arrow.right")
                        .foregroundStyle(.white.opacity(0.4))
                    Spacer()
                    PriceDetail(label: "Exit Price", price: exitPrice, color: .purple)
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.purple.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct PriceDetail: View {
    let label: String
    let price: Double
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.6))
            
            Text("$\(String(format: "%.2f", price))")
                .font(.title.weight(.bold))
                .foregroundStyle(color)
        }
    }
}

struct TradeQuantityCard: View {
    let trade: Trade
    
    var totalValue: Double {
        Double(trade.quantity) * trade.entryPrice
    }
    
    var body: some View {
        HStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Quantity")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.6))
                
                Text("\(trade.quantity) shares")
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.white)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 6) {
                Text("Position Value")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.6))
                
                Text("$\(String(format: "%.2f", totalValue))")
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.white)
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.orange.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct TradeDatesCard: View {
    let trade: Trade
    
    var body: some View {
        VStack(spacing: 16) {
            DateRow(label: "Entry Date", date: trade.entryDate, icon: "calendar.badge.plus")
            
            if let exitDate = trade.exitDate {
                Divider()
                    .background(Color.white.opacity(0.2))
                
                DateRow(label: "Exit Date", date: exitDate, icon: "calendar.badge.checkmark")
                
                Divider()
                    .background(Color.white.opacity(0.2))
                
                HoldingPeriod(entryDate: trade.entryDate, exitDate: exitDate)
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.green.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct DateRow: View {
    let label: String
    let date: Date
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundStyle(.blue)
            
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
            
            Spacer()
            
            Text(date.formatted(date: .abbreviated, time: .shortened))
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
        }
    }
}

struct HoldingPeriod: View {
    let entryDate: Date
    let exitDate: Date
    
    var holdingDays: Int {
        Calendar.current.dateComponents([.day], from: entryDate, to: exitDate).day ?? 0
    }
    
    var body: some View {
        HStack {
            Image(systemName: "clock")
                .foregroundStyle(.purple)
            
            Text("Holding Period")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
            
            Spacer()
            
            Text("\(holdingDays) days")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
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
        VStack(spacing: 16) {
            Text("Performance")
                .font(.headline)
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            HStack(spacing: 20) {
                VStack(spacing: 6) {
                    Text("Profit/Loss")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                    
                    if let profitLoss = trade.profit {
                        Text(profitLoss >= 0 ? "+$\(String(format: "%.2f", profitLoss))" : "-$\(String(format: "%.2f", abs(profitLoss)))")
                            .font(.title2.weight(.bold))
                            .foregroundStyle(profitLoss >= 0 ? .green : .red)
                    }
                }
                
                Spacer()
                
                VStack(spacing: 6) {
                    Text("Return %")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                    
                    Text("\(percentReturn >= 0 ? "+" : "")\(String(format: "%.2f%%", percentReturn))")
                        .font(.title2.weight(.bold))
                        .foregroundStyle(percentReturn >= 0 ? .green : .red)
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint((trade.profitLoss ?? 0 >= 0 ? Color.green : Color.red).opacity(0.1)).interactive(), in: .rect(cornerRadius: 20))
    }
}

#Preview {
    TradesListView()
        .environmentObject(TradingDataManager())
        .background(Color.black)
}
