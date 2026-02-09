//
//  ExampleUsage.swift
//  HEAN
//
//  Example file showing how to use the mock data system
//  Created on 2026-01-31.
//

import SwiftUI
import Combine

// MARK: - Example Dashboard View

struct TradingDashboardExample: View {
    @StateObject private var viewModel = ExampleDashboardViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    // Portfolio Summary
                    PortfolioSummaryCard(portfolio: viewModel.portfolio)

                    // Positions
                    PositionsCard(positions: viewModel.positions)

                    // Active Orders
                    ActiveOrdersCard(
                        orders: viewModel.activeOrders,
                        onCancel: { orderId in
                            Task {
                                await viewModel.cancelOrder(orderId)
                            }
                        }
                    )

                    // Recent Events
                    EventFeedCard(events: viewModel.recentEvents)
                }
                .padding()
            }
            .navigationTitle("HEAN Trading")
            .refreshable {
                await viewModel.refresh()
            }
        }
    }
}

// MARK: - Example Dashboard ViewModel

@MainActor
final class ExampleDashboardViewModel: ObservableObject {
    @Published var portfolio: Portfolio?
    @Published var positions: [Position] = []
    @Published var orders: [Order] = []
    @Published var recentEvents: [TradingEvent] = []

    var activeOrders: [Order] {
        orders.filter { $0.isActive }
    }

    private let container = MockServiceContainer.shared
    private var cancellables = Set<AnyCancellable>()

    init() {
        setupSubscriptions()
        Task {
            await loadInitialData()
        }
    }

    // MARK: - Data Loading

    private func loadInitialData() async {
        do {
            self.portfolio = try await container.portfolioService.fetchPortfolio()
            self.positions = try await container.tradingService.fetchPositions()
            self.orders = try await container.tradingService.fetchOrders(status: nil)
            self.recentEvents = try await container.eventService.fetchRecentEvents(limit: 20)
        } catch {
            print("Error loading initial data: \(error)")
        }
    }

    func refresh() async {
        await loadInitialData()
    }

    // MARK: - Real-time Updates

    private func setupSubscriptions() {
        // Position updates
        container.tradingService.positionsPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] positions in
                self?.positions = positions
            }
            .store(in: &cancellables)

        // Order updates
        container.tradingService.ordersPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] orders in
                self?.orders = orders
            }
            .store(in: &cancellables)

        // Event stream
        container.eventService.eventStream
            .receive(on: DispatchQueue.main)
            .sink { [weak self] event in
                self?.recentEvents.insert(event, at: 0)
                if let count = self?.recentEvents.count, count > 20 {
                    self?.recentEvents.removeLast()
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Actions

    func cancelOrder(_ orderId: String) async {
        do {
            try await container.tradingService.cancelOrder(id: orderId)
        } catch {
            print("Error cancelling order: \(error)")
        }
    }

    func closePosition(_ symbol: String) async {
        do {
            try await container.tradingService.closePosition(symbol: symbol)
        } catch {
            print("Error closing position: \(error)")
        }
    }

    func placeMarketOrder(symbol: String, side: OrderSide, quantity: Double) async {
        do {
            _ = try await container.tradingService.placeOrder(
                symbol: symbol,
                side: side.rawValue,
                type: "market",
                qty: quantity,
                price: nil
            )
        } catch {
            print("Error placing order: \(error)")
        }
    }
}

// MARK: - UI Components

struct PortfolioSummaryCard: View {
    let portfolio: Portfolio?

    var body: some View {
        if let portfolio = portfolio {
            VStack(spacing: 12) {
                Text("Portfolio")
                    .font(.headline)
                    .frame(maxWidth: .infinity, alignment: .leading)

                HStack {
                    VStack(alignment: .leading) {
                        Text("Equity")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(portfolio.formattedEquity)
                            .font(.title2)
                            .bold()
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("Total PnL")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(portfolio.formattedTotalPnL)
                            .font(.title3)
                            .foregroundColor(portfolio.isProfit ? .green : .red)
                            .bold()
                    }
                }

                Divider()

                HStack {
                    Label("Available", systemImage: "dollarsign.circle")
                        .font(.caption)
                    Spacer()
                    Text(portfolio.formattedAvailableBalance)
                        .font(.caption)
                }

                HStack {
                    Label("Margin Used", systemImage: "chart.bar.fill")
                        .font(.caption)
                    Spacer()
                    Text(String(format: "%.1f%%", portfolio.marginUsagePercent))
                        .font(.caption)
                }
            }
            .padding()
            .background(Color(uiColor: .systemBackground))
            .cornerRadius(12)
            .shadow(radius: 2)
        }
    }
}

struct PositionsCard: View {
    let positions: [Position]

    var body: some View {
        VStack(spacing: 8) {
            Text("Positions (\(positions.count))")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            if positions.isEmpty {
                Text("No open positions")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(positions) { position in
                    PositionRow(position: position)
                    if position.id != positions.last?.id {
                        Divider()
                    }
                }
            }
        }
        .padding()
        .background(Color(uiColor: .systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct PositionRow: View {
    let position: Position

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text(position.symbol)
                    .font(.subheadline)
                    .bold()
                Spacer()
                Text(position.side.displayName)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(position.side == .long ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                    .foregroundColor(position.side == .long ? .green : .red)
                    .cornerRadius(4)
            }

            HStack {
                Text("Size: \(position.formattedSize)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("\(position.formattedPnL) (\(position.formattedPnLPercent))")
                    .font(.caption)
                    .foregroundColor(position.isProfit ? .green : .red)
            }
        }
    }
}

struct ActiveOrdersCard: View {
    let orders: [Order]
    let onCancel: (String) -> Void

    var body: some View {
        VStack(spacing: 8) {
            Text("Active Orders (\(orders.count))")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            if orders.isEmpty {
                Text("No active orders")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(orders) { order in
                    OrderRow(order: order, onCancel: {
                        onCancel(order.id)
                    })
                    if order.id != orders.last?.id {
                        Divider()
                    }
                }
            }
        }
        .padding()
        .background(Color(uiColor: .systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct OrderRow: View {
    let order: Order
    let onCancel: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(order.symbol)
                    .font(.subheadline)
                    .bold()
                Text("\(order.side.displayName) \(order.type?.displayName ?? "")")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(order.formattedPrice)
                    .font(.caption)
            }

            Spacer()

            Button(action: onCancel) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
            }
            .buttonStyle(.plain)
        }
    }
}

struct EventFeedCard: View {
    let events: [TradingEvent]

    var body: some View {
        VStack(spacing: 8) {
            Text("Recent Events")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            ForEach(events.prefix(10)) { event in
                EventRow(event: event)
                if event.id != events.prefix(10).last?.id {
                    Divider()
                }
            }
        }
        .padding()
        .background(Color(uiColor: .systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct EventRow: View {
    let event: TradingEvent

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Circle()
                .fill(eventColor)
                .frame(width: 8, height: 8)
                .padding(.top, 4)

            VStack(alignment: .leading, spacing: 2) {
                Text(event.message)
                    .font(.caption)
                Text(event.formattedTime)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
    }

    private var eventColor: Color {
        switch event.type {
        case .signal: return .blue
        case .orderPlaced, .orderFilled: return .green
        case .orderCancelled: return .orange
        case .positionOpened, .positionClosed: return .purple
        case .riskAlert, .error: return .red
        case .systemInfo: return .gray
        }
    }
}

// MARK: - Preview

struct TradingDashboardExample_Previews: PreviewProvider {
    static var previews: some View {
        TradingDashboardExample()
    }
}
