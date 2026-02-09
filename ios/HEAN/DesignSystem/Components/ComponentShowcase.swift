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
            VStack(spacing: AppTypography.xl) {
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
            .padding(AppTypography.md)
        }
        .background(AppColors.backgroundPrimary.ignoresSafeArea())
    }

    private var header: some View {
        GlassCard {
            HStack {
                VStack(alignment: .leading, spacing: AppTypography.xs) {
                    Text("HEAN")
                        .font(AppTypography.title(28, weight: .bold))
                        .foregroundColor(AppColors.textPrimary)

                    Text("Premium Trading Dashboard")
                        .font(AppTypography.caption())
                        .foregroundColor(AppColors.textSecondary)
                }

                Spacer()

                StatusIndicator(
                    status: connectionStatus,
                    latency: 45,
                    showLabel: true
                )
            }
            .padding(AppTypography.lg)
        }
    }

    private var statusSection: some View {
        VStack(spacing: AppTypography.md) {
            sectionTitle("Connection Status")

            GlassCard {
                VStack(spacing: AppTypography.md) {
                    HStack {
                        Text("WebSocket")
                            .font(AppTypography.body())
                            .foregroundColor(AppColors.textPrimary)
                        Spacer()
                        StatusIndicator(status: .connected, latency: 32, showLabel: false)
                    }

                    Divider()
                        .background(AppColors.textTertiary.opacity(0.3))

                    HStack {
                        Text("Market Data")
                            .font(AppTypography.body())
                            .foregroundColor(AppColors.textPrimary)
                        Spacer()
                        StatusIndicator(status: .reconnecting, showLabel: false)
                    }

                    Divider()
                        .background(AppColors.textTertiary.opacity(0.3))

                    HStack {
                        Text("API")
                            .font(AppTypography.body())
                            .foregroundColor(AppColors.textPrimary)
                        Spacer()
                        StatusIndicator(status: .disconnected, showLabel: false)
                    }
                }
                .padding(AppTypography.md)
            }
        }
    }

    private var priceTickersSection: some View {
        VStack(spacing: AppTypography.md) {
            sectionTitle("Live Prices")

            // Large tickers
            HStack(spacing: AppTypography.md) {
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
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: AppTypography.sm) {
                PriceTicker(symbol: "SOL", price: 95.40, changePercent: 5.2, size: .small)
                PriceTicker(symbol: "ADA", price: 0.58, changePercent: -2.1, size: .small)
                PriceTicker(symbol: "DOT", price: 7.23, changePercent: 1.8, size: .small)
                PriceTicker(symbol: "AVAX", price: 38.90, changePercent: -0.5, size: .small)
            }

            // Price update simulator
            Button(action: simulatePriceUpdate) {
                Text("Simulate Price Update")
                    .font(AppTypography.body(14, weight: .semibold))
                    .foregroundColor(AppColors.textPrimary)
                    .padding(.horizontal, AppTypography.lg)
                    .padding(.vertical, AppTypography.sm)
                    .background(AppColors.accentPrimary.opacity(0.2))
                    .cornerRadius(AppTypography.radiusSm)
            }
        }
    }

    private var chartsSection: some View {
        VStack(spacing: AppTypography.md) {
            sectionTitle("Charts")

            // Candlestick chart
            GlassCard {
                VStack(alignment: .leading, spacing: AppTypography.sm) {
                    Text("BTC/USDT - 1H")
                        .font(AppTypography.headline())
                        .foregroundColor(AppColors.textPrimary)

                    CandlestickChart(
                        candles: generateSampleCandles(),
                        currentPrice: 42_500,
                        showGrid: true,
                        showVolume: true
                    )

                    Text("Pinch to zoom • Drag to scroll")
                        .font(AppTypography.caption(11))
                        .foregroundColor(AppColors.textTertiary)
                }
                .padding(AppTypography.md)
            }

            // Sparklines
            HStack(spacing: AppTypography.md) {
                GlassCard {
                    VStack(alignment: .leading, spacing: AppTypography.xs) {
                        Text("24h Trend")
                            .font(AppTypography.caption())
                            .foregroundColor(AppColors.textSecondary)

                        Sparkline(
                            dataPoints: [100, 105, 103, 110, 115, 112, 120, 125, 130],
                            smoothCurves: true
                        )
                    }
                    .padding(AppTypography.sm)
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: AppTypography.xs) {
                        Text("7d Trend")
                            .font(AppTypography.caption())
                            .foregroundColor(AppColors.textSecondary)

                        Sparkline(
                            dataPoints: [130, 125, 128, 120, 115, 118, 110, 105, 100],
                            smoothCurves: true
                        )
                    }
                    .padding(AppTypography.sm)
                }
            }
        }
    }

    private var pnlSection: some View {
        VStack(spacing: AppTypography.md) {
            sectionTitle("Portfolio Performance")

            GlassCard {
                VStack(spacing: AppTypography.md) {
                    // Total PnL
                    HStack {
                        Text("Total P&L")
                            .font(AppTypography.body())
                            .foregroundColor(AppColors.textSecondary)
                        Spacer()
                        PnLBadge(value: 2_456.78, format: .dollar, size: .expanded)
                    }

                    Divider()
                        .background(AppColors.textTertiary.opacity(0.3))

                    // Today's PnL
                    HStack {
                        Text("Today")
                            .font(AppTypography.body())
                            .foregroundColor(AppColors.textSecondary)
                        Spacer()
                        PnLBadge(value: 15.67, format: .percent, size: .compact)
                    }

                    Divider()
                        .background(AppColors.textTertiary.opacity(0.3))

                    // This week
                    HStack {
                        Text("This Week")
                            .font(AppTypography.body())
                            .foregroundColor(AppColors.textSecondary)
                        Spacer()
                        PnLBadge(value: -345.20, format: .dollar, size: .compact)
                    }
                }
                .padding(AppTypography.md)
            }
        }
    }

    private var riskSection: some View {
        VStack(spacing: AppTypography.md) {
            sectionTitle("Risk Management")

            // Compact badges
            HStack(spacing: AppTypography.md) {
                RiskBadge(state: .normal, variant: .compact)
                RiskBadge(state: .softBrake, variant: .compact)
                RiskBadge(state: .quarantine, variant: .compact)
                RiskBadge(state: .hardStop, variant: .compact)
            }

            // Current state (expanded)
            RiskBadge(state: riskState, variant: .expanded)

            // State toggle buttons
            HStack(spacing: AppTypography.sm) {
                Button("Normal") { riskState = .normal }
                Button("Soft") { riskState = .softBrake }
                Button("Quarantine") { riskState = .quarantine }
                Button("Stop") { riskState = .hardStop }
            }
            .font(AppTypography.caption(12, weight: .medium))
            .buttonStyle(.bordered)
            .tint(AppColors.accentPrimary)
        }
    }

    private var loadingSection: some View {
        VStack(spacing: AppTypography.md) {
            sectionTitle("Loading States")

            GlassCard {
                VStack(alignment: .leading, spacing: AppTypography.sm) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(AppColors.backgroundTertiary)
                        .frame(width: 120, height: 20)

                    RoundedRectangle(cornerRadius: 8)
                        .fill(AppColors.backgroundTertiary)
                        .frame(width: 200, height: 40)

                    RoundedRectangle(cornerRadius: 8)
                        .fill(AppColors.backgroundTertiary)
                        .frame(width: 150, height: 16)
                }
                .padding(AppTypography.md)
            }
            .skeleton(isLoading: isLoading)

            Toggle("Show Loading", isOn: $isLoading)
                .tint(AppColors.accentPrimary)
                .padding(.horizontal, AppTypography.md)
        }
    }

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(AppTypography.headline(18, weight: .bold))
            .foregroundColor(AppColors.textPrimary)
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
