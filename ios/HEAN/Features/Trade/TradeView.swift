//
//  TradeView.swift
//  HEAN
//ч
//  Created by HEAN Team on 2026-01-31.
//

import SwiftUI

struct TradeView: View {
    @EnvironmentObject var container: DIContainer
    @State private var selectedSymbol = "BTCUSDT"
    @State private var selectedInterval = "15m"
    @State private var showOrderSheet = false
    @State private var orderSide: OrderSide = .buy

    enum OrderSide: String {
        case buy = "Buy"
        case sell = "Sell"
    }

    let intervals = ["15m", "1h", "4h", "1d"]

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: Theme.Spacing.lg) {
                    // Price Header
                    PriceHeaderCard(symbol: selectedSymbol)

                    // Chart
                    ChartCard(symbol: selectedSymbol, interval: selectedInterval)

                    // Interval Selector
                    IntervalSelector(selectedInterval: $selectedInterval, intervals: intervals)

                    // Order Buttons
                    OrderButtonsRow(showOrderSheet: $showOrderSheet, orderSide: $orderSide)

                    // Order Book Preview
                    OrderBookPreviewCard()
                }
                .padding()
            }
            .background(Theme.Colors.background)
            .navigationTitle("Trade")
            .sheet(isPresented: $showOrderSheet) {
                OrderSheet(
                    symbol: selectedSymbol,
                    side: orderSide,
                    isPresented: $showOrderSheet,
                    tradingService: container.tradingService
                )
            }
        }
    }
}

struct PriceHeaderCard: View {
    let symbol: String
    @State private var currentPrice: Double = 42567.89
    @State private var priceChange: Double = 2.34
    @State private var priceTimer: Timer?

    var body: some View {
        GlassCardAccent {
            VStack(spacing: Theme.Spacing.md) {
                Text(symbol)
                    .font(Theme.Typography.headline)
                    .foregroundColor(Theme.Colors.textSecondary)

                PriceTicker(price: currentPrice, size: .large)

                HStack(spacing: Theme.Spacing.xl) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("24h Change")
                            .font(Theme.Typography.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                        PnLBadge(value: priceChange, percentage: priceChange)
                    }

                    Divider()
                        .frame(height: 30)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("24h Volume")
                            .font(Theme.Typography.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                        Text("$2.4B")
                            .font(Theme.Typography.mono)
                            .foregroundColor(Theme.Colors.textSecondary)
                    }
                }
            }
        }
        .onAppear {
            // Simulate price updates
            priceTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
                Task { @MainActor in
                    currentPrice += Double.random(in: -50...50)
                }
            }
        }
        .onDisappear {
            priceTimer?.invalidate()
            priceTimer = nil
        }
    }
}

struct ChartCard: View {
    let symbol: String
    let interval: String

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                Text("Price Chart")
                    .font(Theme.Typography.headline)
                    .foregroundColor(Theme.Colors.textPrimary)

                // Generate mock candles
                let candles = (0..<30).map { i in
                    let basePrice = 42000.0
                    let variation = Double.random(in: -500...500)
                    let open = basePrice + variation
                    let close = open + Double.random(in: -200...200)
                    let high = max(open, close) + Double.random(in: 0...100)
                    let low = min(open, close) - Double.random(in: 0...100)

                    return Candle(
                        timestamp: Date().addingTimeInterval(TimeInterval(-i * 900)),
                        open: open,
                        high: high,
                        low: low,
                        close: close,
                        volume: Double.random(in: 1000...5000)
                    )
                }

                CandlestickChart(candles: candles.reversed())
                    .frame(height: 250)
            }
        }
    }
}

struct IntervalSelector: View {
    @Binding var selectedInterval: String
    let intervals: [String]

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            ForEach(intervals, id: \.self) { interval in
                Button {
                    Haptics.selection()
                    selectedInterval = interval
                } label: {
                    Text(interval)
                        .font(Theme.Typography.caption1)
                        .fontWeight(.semibold)
                        .foregroundColor(selectedInterval == interval ? Theme.Colors.background : Theme.Colors.textSecondary)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(
                            Capsule()
                                .fill(selectedInterval == interval ? Theme.Colors.accent : Theme.Colors.card)
                        )
                }
            }

            Spacer()
        }
    }
}

struct OrderButtonsRow: View {
    @Binding var showOrderSheet: Bool
    @Binding var orderSide: TradeView.OrderSide

    var body: some View {
        HStack(spacing: Theme.Spacing.md) {
            Button {
                Haptics.medium()
                orderSide = .buy
                showOrderSheet = true
            } label: {
                Text("Buy")
                    .font(Theme.Typography.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: Theme.CornerRadius.md)
                            .fill(Theme.Colors.success)
                    )
            }

            Button {
                Haptics.medium()
                orderSide = .sell
                showOrderSheet = true
            } label: {
                Text("Sell")
                    .font(Theme.Typography.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: Theme.CornerRadius.md)
                            .fill(Theme.Colors.error)
                    )
            }
        }
    }
}

struct OrderBookPreviewCard: View {
    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                Text("Order Book")
                    .font(Theme.Typography.headline)
                    .foregroundColor(Theme.Colors.textPrimary)

                // Mock order book data
                VStack(spacing: 4) {
                    ForEach(0..<5) { i in
                        HStack {
                            Text("$\(42500 + i * 10)")
                                .font(Theme.Typography.monoSmall)
                                .foregroundColor(Theme.Colors.success)

                            Spacer()

                            Text("\(Double.random(in: 0.1...5.0), specifier: "%.2f")")
                                .font(Theme.Typography.monoSmall)
                                .foregroundColor(Theme.Colors.textSecondary)
                        }
                    }
                }

                Divider()

                VStack(spacing: 4) {
                    ForEach(0..<5) { i in
                        HStack {
                            Text("$\(42450 - i * 10)")
                                .font(Theme.Typography.monoSmall)
                                .foregroundColor(Theme.Colors.error)

                            Spacer()

                            Text("\(Double.random(in: 0.1...5.0), specifier: "%.2f")")
                                .font(Theme.Typography.monoSmall)
                                .foregroundColor(Theme.Colors.textSecondary)
                        }
                    }
                }
            }
        }
    }
}

struct OrderSheet: View {
    let symbol: String
    let side: TradeView.OrderSide
    @Binding var isPresented: Bool
    let tradingService: TradingServiceProtocol?

    @State private var orderType: String = "Market"
    @State private var quantity: String = ""
    @State private var price: String = ""
    @State private var isSubmitting = false
    @State private var errorMessage: String?

    let orderTypes = ["Market", "Limit"]

    init(symbol: String, side: TradeView.OrderSide, isPresented: Binding<Bool>, tradingService: TradingServiceProtocol? = nil) {
        self.symbol = symbol
        self.side = side
        self._isPresented = isPresented
        self.tradingService = tradingService
    }

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Order Details")) {
                    Picker("Type", selection: $orderType) {
                        ForEach(orderTypes, id: \.self) { type in
                            Text(type).tag(type)
                        }
                    }
                    .pickerStyle(.segmented)

                    TextField("Quantity", text: $quantity)
                        .keyboardType(.decimalPad)

                    if orderType == "Limit" {
                        TextField("Price", text: $price)
                            .keyboardType(.decimalPad)
                    }
                }

                if let err = errorMessage {
                    Section {
                        Text(err).foregroundColor(.red).font(.caption)
                    }
                }

                Section {
                    Button {
                        Task { await placeOrder() }
                    } label: {
                        HStack {
                            if isSubmitting {
                                ProgressView().tint(.white)
                            }
                            Text("Place \(side.rawValue) Order")
                                .font(Theme.Typography.headline)
                                .foregroundColor(.white)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: Theme.CornerRadius.md)
                                .fill(side == .buy ? Theme.Colors.success : Theme.Colors.error)
                        )
                    }
                    .disabled(isSubmitting || quantity.isEmpty)
                    .listRowBackground(Color.clear)
                }
            }
            .navigationTitle("\(side.rawValue) \(symbol)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
            }
        }
    }

    private func placeOrder() async {
        guard let qty = Double(quantity), qty > 0 else {
            errorMessage = "Please enter a valid quantity"
            return
        }

        guard let service = tradingService else {
            errorMessage = "Trading service not available"
            return
        }

        isSubmitting = true
        errorMessage = nil

        do {
            let orderPrice = orderType == "Limit" ? Double(price) : nil
            _ = try await service.placeOrder(
                symbol: symbol,
                side: side.rawValue.lowercased(),
                type: orderType.lowercased(),
                qty: qty,
                price: orderPrice
            )
            Haptics.success()
            isPresented = false
        } catch {
            errorMessage = error.localizedDescription
            isSubmitting = false
        }
    }
}

#Preview {
    TradeView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}
