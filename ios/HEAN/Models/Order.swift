//
//  Order.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

enum OrderSide: String, Codable {
    case buy = "BUY"
    case sell = "SELL"

    var displayName: String {
        rawValue
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        switch raw.lowercased() {
        case "buy":
            self = .buy
        case "sell":
            self = .sell
        default:
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "Unknown order side: \(raw)"
            ))
        }
    }
}

enum OrderType: String, Codable {
    case market = "MARKET"
    case limit = "LIMIT"
    case stopMarket = "STOP_MARKET"
    case stopLimit = "STOP_LIMIT"

    var displayName: String {
        switch self {
        case .market: return "Market"
        case .limit: return "Limit"
        case .stopMarket: return "Stop Market"
        case .stopLimit: return "Stop Limit"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        switch raw.uppercased() {
        case "MARKET": self = .market
        case "LIMIT": self = .limit
        case "STOP_MARKET": self = .stopMarket
        case "STOP_LIMIT": self = .stopLimit
        default: self = .market
        }
    }
}

enum OrderStatus: String, Codable {
    case new = "NEW"
    case pending = "PENDING"
    case partiallyFilled = "PARTIALLY_FILLED"
    case filled = "FILLED"
    case cancelled = "CANCELLED"
    case rejected = "REJECTED"
    case expired = "EXPIRED"

    var displayName: String {
        switch self {
        case .new: return "Active"
        case .pending: return "Pending"
        case .partiallyFilled: return "Partial"
        case .filled: return "Filled"
        case .cancelled: return "Cancelled"
        case .rejected: return "Rejected"
        case .expired: return "Expired"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        switch raw.uppercased() {
        case "NEW": self = .new
        case "PENDING": self = .pending
        case "PARTIALLY_FILLED": self = .partiallyFilled
        case "FILLED": self = .filled
        case "CANCELLED", "CANCELED": self = .cancelled
        case "REJECTED": self = .rejected
        case "EXPIRED": self = .expired
        default: self = .pending
        }
    }
}

struct Order: Identifiable, Codable {
    let id: String
    let symbol: String
    let side: OrderSide
    let type: OrderType?
    let status: OrderStatus
    let price: Double?
    let quantity: Double
    var filledQuantity: Double
    let createdAt: Date?
    var updatedAt: Date?
    let strategyId: String?

    var isActive: Bool {
        status == .new || status == .pending || status == .partiallyFilled
    }

    var isFilled: Bool {
        status == .filled
    }

    var fillPercent: Double {
        guard quantity > 0 else { return 0 }
        return (filledQuantity / quantity) * 100
    }

    var remainingQuantity: Double {
        quantity - filledQuantity
    }

    var formattedPrice: String {
        guard let price = price else { return "Market" }
        if price >= 1000 {
            return String(format: "$%.2f", price)
        } else if price >= 1 {
            return String(format: "$%.3f", price)
        } else {
            return String(format: "$%.5f", price)
        }
    }

    var formattedQuantity: String {
        String(format: "%.4f", quantity)
    }

    var formattedFilled: String {
        if filledQuantity == 0 {
            return "0%"
        }
        return String(format: "%.0f%%", fillPercent)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case symbol
        case side
        case type
        case status
        case price
        case quantity = "size"
        case filledQuantity = "filled_size"
        case createdAt = "timestamp"
        case updatedAt = "updated_at"
        case strategyId = "strategy_id"
        // Backend alternative key names
        case orderId = "order_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // id: try "id" first, then "order_id"
        if let id = try? container.decode(String.self, forKey: .id) {
            self.id = id
        } else if let ordId = try? container.decode(String.self, forKey: .orderId) {
            self.id = ordId
        } else {
            self.id = UUID().uuidString
        }

        self.symbol = try container.decode(String.self, forKey: .symbol)
        self.side = try container.decode(OrderSide.self, forKey: .side)
        self.type = try container.decodeIfPresent(OrderType.self, forKey: .type)
        self.status = try container.decodeIfPresent(OrderStatus.self, forKey: .status) ?? .pending
        self.price = try container.decodeIfPresent(Double.self, forKey: .price)
        self.quantity = try container.decodeIfPresent(Double.self, forKey: .quantity) ?? 0
        self.filledQuantity = try container.decodeIfPresent(Double.self, forKey: .filledQuantity) ?? 0
        self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)
        self.updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt)
        self.strategyId = try container.decodeIfPresent(String.self, forKey: .strategyId)
    }

    init(id: String, symbol: String, side: OrderSide, type: OrderType? = .market,
         status: OrderStatus = .pending, price: Double? = nil, quantity: Double = 0,
         filledQuantity: Double = 0, createdAt: Date? = nil, updatedAt: Date? = nil,
         strategyId: String? = nil) {
        self.id = id
        self.symbol = symbol
        self.side = side
        self.type = type
        self.status = status
        self.price = price
        self.quantity = quantity
        self.filledQuantity = filledQuantity
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.strategyId = strategyId
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(symbol, forKey: .symbol)
        try container.encode(side, forKey: .side)
        try container.encodeIfPresent(type, forKey: .type)
        try container.encode(status, forKey: .status)
        try container.encodeIfPresent(price, forKey: .price)
        try container.encode(quantity, forKey: .quantity)
        try container.encode(filledQuantity, forKey: .filledQuantity)
        try container.encodeIfPresent(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(strategyId, forKey: .strategyId)
    }
}
