//
//  ComponentShowcase.swift
//  HEAN
//
//  Showcase of all premium UI components
//

import SwiftUI

/// Demo view showcasing all HEAN components
struct ComponentShowcase: View {
    @State private var isLoading = false
    @State private var connectionStatus: StatusIndicator.ConnectionStatus = .connected
    @State private var riskState: RiskBadge.RiskState = .normal
    @State private var btcPrice: Double = 42_350.75
    @State private var ethPrice: Double = 2_245.30

    var body: some View {
        ScrollView {
            VStack(spacing: Theme.Spacing.xl) {
                // Header
                header

                // Status Section
                statusSection

                // Price Tickers
                priceTickersSection

                // Charts
                chartsSection

                // PnL Badges
                pnlSection

                // Risk Management
                riskSection

                // Loading States
                loadingSection
            }
            .padding(Theme.Spacing.lg)
        }
        .background(Theme.Colors.background.ignoresSafeArea())
    }

    private var header: some View {
        GlassCard {
            HStack {
                VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                    Text("HEAN")
                        .font(Theme.Typography.title(28, weight: .bold))
                        .foregroundColor(Theme.Colors.textPrimary)

                    Text("Premium Trading Dashboard")
                        .font(Theme.Typography.caption())
                        .foregroundColor(Theme.Colors.textSecondary)
                }

                Spacer()

                StatusIndicator(
                    status: connectionStatus,
                    latency: 45,
                    showLabel: true
                )
            }
            .padding(20)
        }
    }

    private var statusSection: some View {
        VStack(spacing: Theme.Spacing.lg) {
            sectionTitle("Connection Status")

            GlassCard {
                VStack(spacing: Theme.Spacing.lg) {
                    HStack {
                        Text("WebSocket")
                            .font(Theme.Typography.bodyFont())
                            .foregroundColor(Theme.Colors.textPrimary)
                        Spacer()
                        StatusIndicator(status: .connected, latency: 32, showLabel: false)
                    }

                    Divider()
                        .background(Theme.Colors.textTertiary.opacity(0.3))

                    HStack {
                        Text("Market Data")
                            .font(Theme.Typography.bodyFont())
                            .foregroundColor(Theme.Colors.textPrimary)
                        Spacer()
                        StatusIndicator(status: .reconnecting, showLabel: false)
                    }

                    Divider()
                        .background(Theme.Colors.textTertiary.opacity(0.3))

                    HStack {
                        Text("API")
                            .font(Theme.Typography.bodyFont())
                            .foregroundColor(Theme.Colors.textPrimary)
                        Spacer()
                        StatusIndicator(status: .disconnected, showLabel: false)
                    }
                }
                .padding(Theme.Spacing.lg)
            }
        }
    }

    private var priceTickersSection: some View {
        VStack(spacing: Theme.Spacing.lg) {
            sectionTitle("Live Prices")

            // Large tickers
            HStack(spacing: Theme.Spacing.lg) {
                PriceTicker(
                    symbol: "BTCUSDT",
                    price: btcPrice,
                    changePercent: 3.45,
                    size: .large
                )
                .frame(maxWidth: .infinity)

                PriceTicker(
                    symbol: "ETHUSDT",
                    price: ethPrice,
                    changePercent: -1.25,
                    size: .large
                )
                .frame(maxWidth: .infinity)
            }

            // Small tickers grid
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Theme.Spacing.md) {
                PriceTicker(symbol: "SOL", price: 95.40, changePercent: 5.2, size: .small)
                PriceTicker(symbol: "ADA", price: 0.58, changePercent: -2.1, size: .small)
                PriceTicker(symbol: "DOT", price: 7.23, changePercent: 1.8, size: .small)
                PriceTicker(symbol: "AVAX", price: 38.90, changePercent: -0.5, size: .small)
            }

            // Price update simulator
            Button(action: simulatePriceUpdate) {
                Text("Simulate Price Update")
                    .font(Theme.Typography.bodyFont(14, weight: .semibold))
                    .foregroundColor(Theme.Colors.textPrimary)
                    .padding(.horizontal, 20)
                    .padding(.vertical, Theme.Spacing.md)
                    .background(Theme.Colors.accent.opacity(0.2))
                    .cornerRadius(Theme.CornerRadius.sm)
            }
        }
    }

    private var chartsSection: some View {
        VStack(spacing: Theme.Spacing.lg) {
            sectionTitle("Charts")

            // Candlestick chart
            GlassCard {
                VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                    Text("BTC/USDT - 1H")
                        .font(Theme.Typography.headlineFont())
                        .foregroundColor(Theme.Colors.textPrimary)

                    CandlestickChart(
                        candles: generateSampleCandles(),
                        currentPrice: 42_500,
                        showGrid: true,
                        showVolume: true
                    )

                    Text("Pinch to zoom • Drag to scroll")
                        .font(Theme.Typography.caption(11))
                        .foregroundColor(Theme.Colors.textTertiary)
                }
                .padding(Theme.Spacing.lg)
            }

            // Sparklines
            HStack(spacing: Theme.Spacing.lg) {
                GlassCard {
                    VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                        Text("24h Trend")
                            .font(Theme.Typography.caption())
                            .foregroundColor(Theme.Colors.textSecondary)

                        Sparkline(
                            dataPoints: [100, 105, 103, 110, 115, 112, 120, 125, 130],
                            smoothCurves: true
                        )
                    }
                    .padding(Theme.Spacing.md)
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                        Text("7d Trend")
                            .font(Theme.Typography.caption())
                            .foregroundColor(Theme.Colors.textSecondary)

                        Sparkline(
                            dataPoints: [130, 125, 128, 120, 115, 118, 110, 105, 100],
                            smoothCurves: true
                        )
                    }
                    .padding(Theme.Spacing.md)
                }
            }
        }
    }

    private var pnlSection: some View {
        VStack(spacing: Theme.Spacing.lg) {
            sectionTitle("Portfolio Performance")

            GlassCard {
                VStack(spacing: Theme.Spacing.lg) {
                    // Total PnL
                    HStack {
                        Text("Total P&L")
                            .font(Theme.Typography.bodyFont())
                            .foregroundColor(Theme.Colors.textSecondary)
                        Spacer()
                        PnLBadge(value: 2_456.78, format: .dollar, size: .expanded)
                    }

                    Divider()
                        .background(Theme.Colors.textTertiary.opacity(0.3))

                    // Today's PnL
                    HStack {
                        Text("Today")
                            .font(Theme.Typography.bodyFont())
                            .foregroundColor(Theme.Colors.textSecondary)
                        Spacer()
                        PnLBadge(value: 15.67, format: .percent, size: .compact)
                    }

                    Divider()
                        .background(Theme.Colors.textTertiary.opacity(0.3))

                    // This week
                    HStack {
                        Text("This Week")
                            .font(Theme.Typography.bodyFont())
                            .foregroundColor(Theme.Colors.textSecondary)
                        Spacer()
                        PnLBadge(value: -345.20, format: .dollar, size: .compact)
                    }
                }
                .padding(Theme.Spacing.lg)
            }
        }
    }

    private var riskSection: some View {
        VStack(spacing: Theme.Spacing.lg) {
            sectionTitle("Risk Management")

            // Compact badges
            HStack(spacing: Theme.Spacing.lg) {
                RiskBadge(state: .normal, variant: .compact)
                RiskBadge(state: .softBrake, variant: .compact)
                RiskBadge(state: .quarantine, variant: .compact)
                RiskBadge(state: .hardStop, variant: .compact)
            }

            // Current state (expanded)
            RiskBadge(state: riskState, variant: .expanded)

            // State toggle buttons
            HStack(spacing: Theme.Spacing.md) {
                Button("Normal") { riskState = .normal }
                Button("Soft") { riskState = .softBrake }
                Button("Quarantine") { riskState = .quarantine }
                Button("Stop") { riskState = .hardStop }
            }
            .font(Theme.Typography.caption(12, weight: .medium))
            .buttonStyle(.bordered)
            .tint(Theme.Colors.accent)
        }
    }

    private var loadingSection: some View {
        VStack(spacing: Theme.Spacing.lg) {
            sectionTitle("Loading States")

            GlassCard {
                VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Theme.Colors.cardElevated)
                        .frame(width: 120, height: 20)

                    RoundedRectangle(cornerRadius: 8)
                        .fill(Theme.Colors.cardElevated)
                        .frame(width: 200, height: 40)

                    RoundedRectangle(cornerRadius: 8)
                        .fill(Theme.Colors.cardElevated)
                        .frame(width: 150, height: 16)
                }
                .padding(Theme.Spacing.lg)
            }
            .skeleton(isLoading: isLoading)

            Toggle("Show Loading", isOn: $isLoading)
                .tint(Theme.Colors.accent)
                .padding(.horizontal, Theme.Spacing.lg)
        }
    }

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(Theme.Typography.headlineFont(18, weight: .bold))
            .foregroundColor(Theme.Colors.textPrimary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func simulatePriceUpdate() {
        btcPrice += Double.random(in: -500...500)
        ethPrice += Double.random(in: -50...50)
    }
}

// MARK: - Preview
#Preview {
    ComponentShowcase()
}

// MARK: - Helper Functions
private func generateSampleCandles() -> [Candle] {
    var candles: [Candle] = []
    var price = 42_000.0
    let calendar = Calendar.current

    for i in 0..<50 {
        let timestamp = calendar.date(byAdding: .hour, value: -50 + i, to: Date()) ?? Date()

        let open = price
        let change = Double.random(in: -200...200)
        let close = open + change
        let high = max(open, close) + Double.random(in: 0...100)
        let low = min(open, close) - Double.random(in: 0...100)
        let volume = Double.random(in: 1000...5000)

        candles.append(Candle(
            timestamp: timestamp,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        ))

        price = close
    }

    return candles
}
