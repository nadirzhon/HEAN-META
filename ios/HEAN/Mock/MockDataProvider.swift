//
//  MockDataProvider.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

/// Provides static mock data generators for the HEAN trading system
enum MockDataProvider {

    // MARK: - Markets

    static func generateMarkets() -> [Market] {
        let configs: [(symbol: String, base: String, quote: String, name: String, price: Double, change: Double)] = [
            ("BTCUSDT", "BTC", "USDT", "Bitcoin", 67234.50, 2.5),
            ("ETHUSDT", "ETH", "USDT", "Ethereum", 3456.78, -1.2),
            ("SOLUSDT", "SOL", "USDT", "Solana", 142.33, 5.8),
            ("BNBUSDT", "BNB", "USDT", "BNB", 582.45, 1.1),
            ("XRPUSDT", "XRP", "USDT", "Ripple", 0.5234, -2.3),
            ("ADAUSDT", "ADA", "USDT", "Cardano", 0.6789, 3.4),
            ("DOGEUSDT", "DOGE", "USDT", "Dogecoin", 0.08912, -0.8),
            ("MATICUSDT", "MATIC", "USDT", "Polygon", 1.1234, 4.2),
            ("LINKUSDT", "LINK", "USDT", "Chainlink", 18.56, 2.9),
            ("AVAXUSDT", "AVAX", "USDT", "Avalanche", 42.78, -3.1),
            ("ATOMUSDT", "ATOM", "USDT", "Cosmos", 11.23, 1.5),
            ("DOTUSDT", "DOT", "USDT", "Polkadot", 8.45, -1.9)
        ]

        return configs.map { config in
            let change24h = config.price * (config.change / 100)
            let high24h = config.price * (1 + abs(config.change) / 100 * 1.5)
            let low24h = config.price * (1 - abs(config.change) / 100 * 1.2)
            let volume24h = Double.random(in: 50_000_000...500_000_000)

            return Market(
                id: config.symbol,
                symbol: config.symbol,
                baseCurrency: config.base,
                quoteCurrency: config.quote,
                name: config.name,
                price: config.price,
                change24h: change24h,
                changePercent24h: config.change,
                volume24h: volume24h,
                high24h: high24h,
                low24h: low24h,
                sparkline: generateSparkline(points: 20)
            )
        }
    }

    // MARK: - Candles

    static func generateCandles(count: Int, basePrice: Double) -> [Candle] {
        var candles: [Candle] = []
        var currentPrice = basePrice
        let now = Date()

        for i in (0..<count).reversed() {
            let timestamp = now.addingTimeInterval(TimeInterval(-i * 60)) // 1 minute candles

            // Smooth random walk: -0.5% to +0.5% per candle
            let priceChange = currentPrice * Double.random(in: -0.005...0.005)
            currentPrice += priceChange

            let open = currentPrice
            let volatility = currentPrice * 0.003 // 0.3% intra-candle volatility

            // Generate OHLC with realistic wicks
            let close = open + Double.random(in: -volatility...volatility)
            let high = max(open, close) + Double.random(in: 0...volatility * 0.5)
            let low = min(open, close) - Double.random(in: 0...volatility * 0.5)
            let volume = Double.random(in: 1_000...10_000)

            let candle = Candle(
                id: UUID().uuidString,
                timestamp: timestamp,
                open: open,
                high: high,
                low: low,
                close: close,
                volume: volume
            )

            candles.append(candle)
            currentPrice = close
        }

        return candles
    }

    // MARK: - Positions

    static func generatePositions() -> [Position] {
        let now = Date()

        return [
            Position(
                id: UUID().uuidString,
                symbol: "BTCUSDT",
                side: .long,
                size: 0.05,
                entryPrice: 65000.0,
                markPrice: 67234.50,
                unrealizedPnL: 111.73,
                unrealizedPnLPercent: 3.44,
                leverage: 5,
                createdAt: now.addingTimeInterval(-3600)
            ),
            Position(
                id: UUID().uuidString,
                symbol: "ETHUSDT",
                side: .short,
                size: 0.5,
                entryPrice: 3500.0,
                markPrice: 3456.78,
                unrealizedPnL: 21.61,
                unrealizedPnLPercent: 1.24,
                leverage: 3,
                createdAt: now.addingTimeInterval(-7200)
            ),
            Position(
                id: UUID().uuidString,
                symbol: "SOLUSDT",
                side: .long,
                size: 5.0,
                entryPrice: 138.50,
                markPrice: 142.33,
                unrealizedPnL: 19.15,
                unrealizedPnLPercent: 2.77,
                leverage: 2,
                createdAt: now.addingTimeInterval(-1800)
            )
        ]
    }

    // MARK: - Orders

    static func generateOrders() -> [Order] {
        let now = Date()

        return [
            // Active limit orders
            Order(
                id: UUID().uuidString,
                symbol: "BTCUSDT",
                side: .buy,
                type: .limit,
                status: .new,
                price: 66500.0,
                quantity: 0.02,
                filledQuantity: 0,
                createdAt: now.addingTimeInterval(-300),
                updatedAt: now.addingTimeInterval(-300)
            ),
            Order(
                id: UUID().uuidString,
                symbol: "ETHUSDT",
                side: .sell,
                type: .limit,
                status: .new,
                price: 3480.0,
                quantity: 0.3,
                filledQuantity: 0,
                createdAt: now.addingTimeInterval(-600),
                updatedAt: now.addingTimeInterval(-600)
            ),
            // Partially filled
            Order(
                id: UUID().uuidString,
                symbol: "SOLUSDT",
                side: .buy,
                type: .limit,
                status: .partiallyFilled,
                price: 141.0,
                quantity: 10.0,
                filledQuantity: 6.5,
                createdAt: now.addingTimeInterval(-1200),
                updatedAt: now.addingTimeInterval(-180)
            ),
            // Recently filled
            Order(
                id: UUID().uuidString,
                symbol: "BTCUSDT",
                side: .buy,
                type: .market,
                status: .filled,
                price: nil,
                quantity: 0.05,
                filledQuantity: 0.05,
                createdAt: now.addingTimeInterval(-3650),
                updatedAt: now.addingTimeInterval(-3600)
            ),
            Order(
                id: UUID().uuidString,
                symbol: "ETHUSDT",
                side: .sell,
                type: .market,
                status: .filled,
                price: nil,
                quantity: 0.5,
                filledQuantity: 0.5,
                createdAt: now.addingTimeInterval(-7250),
                updatedAt: now.addingTimeInterval(-7200)
            ),
            // Cancelled
            Order(
                id: UUID().uuidString,
                symbol: "XRPUSDT",
                side: .buy,
                type: .limit,
                status: .cancelled,
                price: 0.51,
                quantity: 100.0,
                filledQuantity: 0,
                createdAt: now.addingTimeInterval(-14400),
                updatedAt: now.addingTimeInterval(-14000)
            )
        ]
    }

    // MARK: - Portfolio

    static func generatePortfolio() -> Portfolio {
        Portfolio(
            equity: 345.67,
            availableBalance: 198.43,
            usedMargin: 147.24,
            unrealizedPnL: 152.49,
            realizedPnL: -106.82,
            initialCapital: 300.0,
            lastUpdated: Date()
        )
    }

    // MARK: - Events

    static func generateEvents(count: Int) -> [TradingEvent] {
        let eventTemplates: [(type: EventType, messages: [String])] = [
            (.signal, [
                "Long signal detected on %@",
                "Short signal detected on %@",
                "Reversal signal on %@"
            ]),
            (.orderPlaced, [
                "Limit order placed on %@",
                "Market order placed on %@"
            ]),
            (.orderFilled, [
                "Order filled on %@ at market price",
                "Partial fill on %@"
            ]),
            (.positionOpened, [
                "Opened LONG position on %@",
                "Opened SHORT position on %@"
            ]),
            (.positionClosed, [
                "Closed position on %@ with profit",
                "Stopped out on %@"
            ]),
            (.riskAlert, [
                "Position size limit reached",
                "Drawdown warning: approaching threshold"
            ]),
            (.systemInfo, [
                "Trading engine started",
                "WebSocket reconnected",
                "Market data stream active"
            ])
        ]

        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        var events: [TradingEvent] = []
        let now = Date()

        for i in (0..<count).reversed() {
            let template = eventTemplates.randomElement()!
            let message = template.messages.randomElement()!
            let symbol = symbols.randomElement()!

            let finalMessage: String
            let eventSymbol: String?

            if message.contains("%@") {
                finalMessage = String(format: message, symbol)
                eventSymbol = symbol
            } else {
                finalMessage = message
                eventSymbol = nil
            }

            let event = TradingEvent(
                id: UUID().uuidString,
                type: template.type,
                symbol: eventSymbol,
                message: finalMessage,
                timestamp: now.addingTimeInterval(TimeInterval(-i * Int.random(in: 2...5))),
                metadata: nil
            )

            events.append(event)
        }

        return events.sorted { $0.timestamp > $1.timestamp }
    }

    // MARK: - Sparkline

    static func generateSparkline(points: Int) -> [Double] {
        var data: [Double] = []
        var value = 100.0

        for _ in 0..<points {
            // Smooth random walk: -2% to +2% per point
            let change = value * Double.random(in: -0.02...0.02)
            value += change
            data.append(value)
        }

        return data
    }

    // MARK: - Price Update Helpers

    /// Updates market price with smooth random walk
    static func updateMarketPrice(_ market: Market, volatility: Double = 0.001) -> Market {
        let priceChange = market.price * Double.random(in: -volatility...volatility)
        let newPrice = market.price + priceChange

        // Update 24h change
        let oldPrice = market.price - market.change24h
        let newChange24h = newPrice - oldPrice
        let newChangePercent = (newChange24h / oldPrice) * 100

        // Update high/low if necessary
        let newHigh = max(market.high24h, newPrice)
        let newLow = min(market.low24h, newPrice)

        // Update sparkline by shifting and adding new price
        var newSparkline = market.sparkline ?? []
        if !newSparkline.isEmpty {
            newSparkline.removeFirst()
            newSparkline.append(newPrice)
        }

        return Market(
            id: market.id,
            symbol: market.symbol,
            baseCurrency: market.baseCurrency,
            quoteCurrency: market.quoteCurrency,
            name: market.name,
            price: newPrice,
            change24h: newChange24h,
            changePercent24h: newChangePercent,
            volume24h: market.volume24h,
            high24h: newHigh,
            low24h: newLow,
            sparkline: newSparkline
        )
    }

    /// Updates position with new mark price and recalculates PnL
    static func updatePosition(_ position: Position, newMarkPrice: Double) -> Position {
        let priceDiff = position.side == .long ?
            (newMarkPrice - position.entryPrice) :
            (position.entryPrice - newMarkPrice)

        let unrealizedPnL = priceDiff * position.size
        let unrealizedPnLPercent = (priceDiff / position.entryPrice) * 100 * Double(position.leverage)

        return Position(
            id: position.id,
            symbol: position.symbol,
            side: position.side,
            size: position.size,
            entryPrice: position.entryPrice,
            markPrice: newMarkPrice,
            unrealizedPnL: unrealizedPnL,
            unrealizedPnLPercent: unrealizedPnLPercent,
            leverage: position.leverage,
            createdAt: position.createdAt
        )
    }

    /// Updates portfolio based on position changes
    static func updatePortfolio(_ portfolio: Portfolio, positions: [Position]) -> Portfolio {
        let totalUnrealizedPnL = positions.reduce(0) { $0 + $1.unrealizedPnL }
        let totalMarginUsed = positions.reduce(0) { $0 + $1.marginUsed }
        let equity = portfolio.initialCapital + portfolio.realizedPnL + totalUnrealizedPnL
        let availableBalance = equity - totalMarginUsed

        return Portfolio(
            equity: equity,
            availableBalance: availableBalance,
            usedMargin: totalMarginUsed,
            unrealizedPnL: totalUnrealizedPnL,
            realizedPnL: portfolio.realizedPnL,
            initialCapital: portfolio.initialCapital,
            lastUpdated: Date()
        )
    }
}
